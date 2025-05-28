import copy
import math
import time
from datetime import datetime

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import HeteroGraphConv
from dgl.nn.pytorch import SAGEConv
from torch.utils.data import Dataset
import os
import random
from utils import encoders
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
from torchmetrics import IoU
import matplotlib.pyplot as plt
import pandas as pd



# 边界框计算函数
def bounding_box_uvgrid(inp: torch.Tensor):
    pts = inp[..., :3].reshape((-1, 3))
    mask = inp[..., 6].reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces, :]
    return bounding_box_pointcloud(pts)

def bounding_box_pointcloud(pts: torch.Tensor):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return torch.tensor(box)

# 中心化和缩放函数
def center_and_scale_uvgrid(inp: torch.Tensor, return_center_scale=False):
    bbox = bounding_box_uvgrid(inp)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])
    center = 0.5 * (bbox[0] + bbox[1])
    inp[..., :3] -= center
    inp[..., :3] *= scale
    if return_center_scale:
        return inp, center, scale
    return inp

# 划分训练集、验证集和测试集
def split_dataset(dataset, test_ratio=0.15, val_ratio=0.15):
    random.seed(42)  # 设置随机种子，确保可重复性
    num_graphs = len(dataset)
    test_size = int(num_graphs * test_ratio)
    val_size = int(num_graphs * val_ratio)
    train_size = num_graphs - test_size - val_size

    indices = list(range(num_graphs))
    random.shuffle(indices)

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    return train_dataset, val_dataset, test_dataset

class HeteroGraphDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.graphs = []
        self._load_graphs()
        self.center_and_scale()

    def _load_graphs(self):
        file_list = os.listdir(self.data_dir)
        for file_name in tqdm(file_list, desc='Loading graphs'):
            if file_name.endswith('.bin'):
                graph_path = os.path.join(self.data_dir, file_name)
                graph = dgl.load_graphs(graph_path)[0][0]
                self.graphs.append(graph)

    def _geo_augment(self, graph):
        # 随机旋转（Z轴）
        if np.random.rand() < 0:
            theta = math.pi * 2 * np.random.rand()
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)

            # 旋转变换矩阵
            rot = torch.tensor([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ], dtype=torch.float32)

            face_data = graph.ndata['x']['face']
            original_shape = face_data.shape
            # 将 face_data 调整为二维张量，以便进行矩阵乘法
            flat_face_data = face_data.view(-1, original_shape[-1])

            # 检查列数是否足够
            if flat_face_data.shape[1] < 3:
                raise ValueError(f"graph.ndata['x']['face'] 列数不足 3，实际列数为 {flat_face_data.shape[1]}")
            rot = rot.to(flat_face_data.dtype)

            # 变换面特征
            try:
                flat_face_data[:, :3] = flat_face_data[:, :3] @ rot
            except RuntimeError as e:
                print(f"矩阵乘法出错，flat_face_data 形状: {flat_face_data.shape}, rot 形状: {rot.shape}")
                raise e

            # 将形状恢复为原始形状
            graph.ndata['x']['face'] = flat_face_data.view(original_shape)



        # 添加高斯噪声
        if np.random.rand() < 0:
            noise = torch.randn_like(graph.ndata['x']['face']) * 0.1
            graph.ndata['x']['face'] += noise[:, :graph.ndata['x']['face'].size(1)]

        return graph

    def __getitem__(self, idx):

        graph = copy.deepcopy(self.graphs[idx])  # 避免污染原始数据
        return self._geo_augment(graph)

    def __len__(self):
        return len(self.graphs)

    def center_and_scale(self):
        for i in range(len(self.graphs)):
            self.graphs[i].nodes['face'].data['x'], center, scale = center_and_scale_uvgrid(self.graphs[i].nodes['face'].data['x'], return_center_scale=True)

            edge_features = self.graphs[i].edges['adjacent_to'].data.get('x')
            if edge_features is not None and edge_features.numel() > 0:
                self.graphs[i].edges['adjacent_to'].data['x'][..., :3] -= center
                self.graphs[i].edges['adjacent_to'].data['x'][..., :3] *= scale

class RGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, rel_names):
        super(RGCN, self).__init__()
        # self.conv1 = HeteroGraphConv({
        #     rel: GraphConv(in_feats, hidden_size, )
        #     for rel in rel_names}, aggregate='sum')
        # self.conv2 = HeteroGraphConv({
        #     rel: GraphConv(hidden_size, hidden_size)
        #     for rel in rel_names}, aggregate='sum')

        # self.conv1 = HeteroGraphConv({
        #     'adjacent_to': SAGEConv(in_feats, hidden_size, 'mean', feat_drop=0.5),
        #     'belongs_to': SAGEConv(in_feats, hidden_size, 'mean', feat_drop=0.5)},
        #      aggregate='sum')


        self.conv1 = HeteroGraphConv({
            rel: SAGEConv(in_dim , hidden_dim, 'mean', feat_drop=0.5)
            for rel in rel_names}, aggregate='sum')

        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = HeteroGraphConv({
            rel: SAGEConv(hidden_dim , hidden_dim, 'mean', feat_drop=0.5)
            for rel in rel_names}, aggregate='sum')

        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, graph, inputs):
        # h1 = self.conv1(graph, inputs, mod_kwargs={
        #     'adjacent_to': {'edge_weight': graph.edges['adjacent_to'].data['x']}})
        # h1 = {k: self.bn1(v) for k, v in h1.items()}
        # h1 = {k: F.leaky_relu(v) for k, v in h1.items()}
        #
        # identity = h1
        # h2 = self.conv2(graph, h1, mod_kwargs={
        #     'adjacent_to': {'edge_weight': graph.edges['adjacent_to'].data['x']}})
        # h2 = {k: self.bn2(v) for k, v in h2.items()}
        # h2 = {k: F.leaky_relu(v + identity[k]) for k, v in h2.items()}
        #
        # # 特征融合
        # h_out = {k: self.fuse(torch.cat([h1[k], h2[k]], dim=1)) for k in h1}
        #
        h1 = self.conv1(graph, inputs)
        h1 = {k: self.bn1(v) for k, v in h1.items()}
        h1 = {k: F.leaky_relu(v) for k, v in h1.items()}

        identity = h1
        h2 = self.conv2(graph, h1)
        h2 = {k: self.bn2(v) for k, v in h2.items()}
        h2 = {k: F.leaky_relu(v + identity[k]) for k, v in h2.items()}

        # 特征融合
        h_out = {k: self.fuse(torch.cat([h1[k], h2[k]], dim=1)) for k in h1}
        return h_out

# class RGCN_GAT(nn.Module):
#     def __init__(self, in_feats, hidden_size, rel_names, node_types):
#         super(RGCN_GAT, self).__init__()
#         self.conv1 = HeteroGraphConv({
#             rel: GATConv(in_feats, in_feats // 8, num_heads=8, feat_drop=0.5, attn_drop=0.3, residual=True)
#             for rel in rel_names}, aggregate='sum')
#         self.conv2 = HeteroGraphConv({
#             rel: GATConv(in_feats, hidden_size, num_heads=1, feat_drop=0.5, attn_drop=0.3, residual=True)
#             for rel in rel_names}, aggregate='sum')
#
#         # self.conv1 = HeteroGraphConv({
#         #     'adjacent_to': EdgeGATConv(in_feats=in_feats, out_feats=in_feats // 8, edge_feats=64, num_heads=8, feat_drop=0.5, attn_drop=0.5),
#         #     'belongs_to': GATConv(in_feats, in_feats // 8, num_heads=8, feat_drop=0.5, attn_drop=0.5)  # 假设belongs_to无边特征
#         # }, aggregate='sum')
#         # self.conv2 = HeteroGraphConv({
#         #     'adjacent_to': EdgeGATConv(in_feats=in_feats, out_feats=hidden_size, edge_feats=64, num_heads=1, feat_drop=0.5, attn_drop=0.5),
#         #     'belongs_to': GATConv(in_feats, hidden_size, num_heads=1, feat_drop=0.5, attn_drop=0.5)
#         # }, aggregate='sum')
#
#
#         self.node_types = node_types
#         self.bn1 = nn.ModuleDict({
#             nt: nn.BatchNorm1d(in_feats) for nt in node_types
#         })
#         self.bn2 = nn.ModuleDict({
#             nt: nn.BatchNorm1d(hidden_size) for nt in node_types
#         })
#     def forward(self, graph, inputs):
#         # edge_feats = {
#         #     'adjacent_to': graph.edges['adjacent_to'].data['x']
#         # }
#         #
#         # print("Edge inputs:", edge_feats['adjacent_to'].shape)
#
#         h = self.conv1(graph, inputs)
#         # print("After conv1, feature shapes:")
#         # for k, v in h.items():
#         #     print(f"Node type {k}: {v.shape}")
#         # 调整特征形状以适应批归一化层
#         h = {k: v.view(v.size(0), -1) for k, v in h.items()}
#         h = {k: self.bn1[k](v) for k, v in h.items()}
#         h = {k: F.leaky_relu(v) for k, v in h.items()}
#         # print("Before conv2, feature shapes:")
#         # for k, v in h.items():
#         #     print(f"Node type {k}: {v.shape}")
#         h = self.conv2(graph, h)
#         h = {k: v.squeeze(1) if v.dim() == 3 else v for k, v in h.items()}
#         h = {k: self.bn2[k](v) for k, v in h.items()}
#         h = {k: F.leaky_relu(v) for k, v in h.items()}
#         return h.

class facenet(nn.Module):
    def __init__(self, in_dim, hidden_dim ):
        super(facenet, self).__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, 'mean', feat_drop=0.5)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, 'mean', feat_drop=0.5)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = self.bn1(h)
        h = F.leaky_relu(h)
        h = self.conv2(graph, h)
        h = self.bn2(h)
        h = F.leaky_relu(h)
        return h


class HeteroGraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names, node_types):
        super(HeteroGraphClassifier, self).__init__()

        self.rgcn = RGCN(in_dim  , hidden_dim,  rel_names)
        # self.rgcn = RGCN_GAT(in_dim * 2, hidden_dim, n_classes, rel_names, node_types)
        # self.rgcn = facenet(in_dim, hidden_dim)


        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim *2 , 256),  # 拼接全局特征
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

        self.vertex_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, in_dim ),
        )

        self.attr_encoder = nn.Sequential(
            nn.Linear(11, 16),  # 假设vertex原始特征维度
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, in_dim),
            # nn.BatchNorm1d(64),
            # nn.LeakyReLU()
        )
        self.curve_encoder = encoders.UVNetCurveEncoder(in_channels=6, output_dims=in_dim *2)
        self.face_encoder = encoders.UVNetSurfaceEncoder(in_channels=7, output_dims=in_dim )

    def forward(self, g):
        h = g.ndata['x']
        embd_face_feat = self.face_encoder(h['face'])

        # face_attr = self.attr_encoder(g.nodes['face'].data['attr'])
        # h['face'] = torch.cat([embd_face_feat, face_attr], dim=1)  # 拼接 indim * 2
        # h['vertex'] = h['vertex'].to(torch.float32)
        # h['vertex'] = self.vertex_encoder(h['vertex'])
        # g.edges['adjacent_to'].data['x'] = self.curve_encoder(g.edges['adjacent_to'].data['x'])
        # # 提取全局信息
        # # print(f"h['face'] shape: {h['face'].shape}, h['vertex'].shape: {h['vertex'].shape}, g.edges['adjacent_to'].data['x'].shape: {g.edges['adjacent_to'].data['x'].shape}")
        # h = self.rgcn(g, h)
        # g.ndata['x'] = h  # graph update
        #
        # mean_feat = dgl.mean_nodes(g, feat='x', ntype='face')
        # global_feat = mean_feat.view(mean_feat.size(0), -1)  # 全局特征
        # global_feat = global_feat.repeat_interleave(g.batch_num_nodes('face'), dim=0)  # 重复全局特征以匹配 face 节点数量
        # face_feat = torch.cat([h['face'], global_feat], dim=1).to(torch.float32)
        # # print(f"face_feat shape: {face_feat.shape}")
        # output_tensor = self.classifier(face_feat)
        # return output_tensor


        h['face'] = embd_face_feat
        h['vertex'] = h['vertex'].to(torch.float32)
        h['vertex'] = self.vertex_encoder(h['vertex'])


        # print(f"h['face'] shape: {h['face'].shape}, h['vertex'].shape: {h['vertex'].shape}, g.edges['adjacent_to'].data['x'].shape: {g.edges['adjacent_to'].data['x'].shape}")
        h = self.rgcn(g, h)
        g.ndata['x'] = h  # graph update

        mean_feat = dgl.mean_nodes(g, feat='x', ntype='face')
        global_feat = mean_feat.view(mean_feat.size(0), -1)  # 全局特征
        global_feat = global_feat.repeat_interleave(g.batch_num_nodes('face'), dim=0)  # 重复全局特征以匹配 face 节点数量
        face_feat = torch.cat([h['face'], global_feat], dim=1).to(torch.float32)
        # print(f"face_feat shape: {face_feat.shape}")
        output_tensor = self.classifier(face_feat)
        return output_tensor

        # output_tensor = self.classifier(h['face'])
        # h['face'] = embd_face_feat
        # subgraph = g.edge_type_subgraph([('face', 'adjacent_to', 'face')])
        # h['face'] = self.rgcn(subgraph, h['face'])
        # output_tensor = self.classifier(h['face'])
        # return output_tensor

def train_model(model, train_dataloader, val_dataloader, optimizer, epochs, device, checkpoint_dir):
    best_val_acc = 0
    best_checkpoint_path = None

    num_classes = 8
    train_iou = IoU(num_classes=num_classes, compute_on_step=False).to(device)
    val_iou = IoU(num_classes=num_classes, compute_on_step=False).to(device)

    val_accuracies = []  # 用于记录每个 epoch 的验证集准确率
    all_train_logs = []  # 用于存储训练日志信息


    current_date = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    # 创建以日期为名的子文件夹
    date_dir = os.path.join(checkpoint_dir, current_date)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)


    for epoch in tqdm(range(epochs), desc='Training epochs'):
        model.train()
        total_loss = 0
        total_correct = 0
        train_loss = 0
        train_iou.reset()

        for graph in tqdm(train_dataloader, desc=f'Epoch {epoch} Training', leave=False):
            graph = graph.to(device)
            labels = graph.nodes['face'].data['y'].to(device)

            logits = model(graph)
            logits = torch.squeeze(logits)

            assert logits.dim() == 2 and logits.shape[
                1] == num_classes, f"Expected logits shape (N, {num_classes}), got {logits.shape}"

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            _, predicted = torch.max(logits, 1)
            train_loss += loss.item()
            total_loss += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            predicted = predicted.to(device)

            train_iou.update(predicted, labels)

        train_accuracy = total_correct / total_loss
        train_iou_value = train_iou.compute()
        print(f'Epoch {epoch}: Training accuracy: {100 * train_accuracy:.2f} %')
        print(f'Average train loss: {train_loss / len(train_dataloader):.4f}')
        print(f'Epoch {epoch}: Training IoU: {train_iou_value:.4f}')



        # 记录模型参数日志
        log_file = os.path.join(date_dir, 'model_params.log')
        with open(log_file, 'w') as f:
            f.write(f"indim: 64+2\n")
            f.write(f"epochs: {epochs}\n")
            f.write(f"rgcn_conv1: {model.rgcn.conv1}\n")
            f.write(f"rgcn_conv2: {model.rgcn.conv2}\n")

        model.eval()
        val_correct = 0
        val_total = 0
        val_iou.reset()
        val_loss = 0
        with torch.no_grad():
            for graph in tqdm(val_dataloader, desc=f'Epoch {epoch} Validation', leave=False):
                graph = graph.to(device)
                labels = graph.nodes['face'].data['y'].to(device)
                logits = model(graph)
                logits = torch.squeeze(logits)
                assert logits.dim() == 2 and logits.shape[
                    1] == num_classes, f"Expected logits shape (N, {num_classes}), got {logits.shape}"

                loss = F.cross_entropy(logits, labels)
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                predicted = predicted.to(device)
                val_iou.update(predicted, labels)

        val_accuracy = val_correct / val_total
        val_iou_value = val_iou.compute()
        print(f'Epoch {epoch}: Validation accuracy: {100 * val_accuracy:.2f} %')
        print(f'Epoch {epoch}: Validation IoU: {val_iou_value:.4f}')
        # 使用验证集损失更新学习率
        scheduler.step(val_loss / len(val_dataloader))

        val_accuracies.append(val_accuracy)


        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            checkpoint = {
                'epoch': epoch,
               'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_acc,
            }

            with open(log_file, 'a') as f:
                f.write(f"Training accuracy: {100 * train_accuracy:.2f} \n")
                f.write(f"Validation accuracy: {100 * val_accuracy:.2f} \n")


            checkpoint_path = os.path.join(date_dir, 'checkpoint.pth')
            torch.save(checkpoint, checkpoint_path)
            best_checkpoint_path = checkpoint_path
            print(f'Checkpoint saved at epoch {epoch} with validation accuracy {100 * best_val_acc:.2f} %')

        # 绘制验证集准确率曲线
    plt.plot(range(epochs), val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Epoch')
    plt.savefig(os.path.join(checkpoint_dir, 'validation_accuracy_curve.png'))
    plt.show()

    return model, log_file, best_checkpoint_path

def test_model(model, test_dataloader, device, log_file):
    model.eval()
    num_classes = 8  # 根据你的输出维度修改
    test_iou = IoU(num_classes=num_classes, compute_on_step=False).to(device)
    test_iou.reset()

    test_correct = 0
    test_total = 0
    test_loss = 0
    with torch.no_grad():
        for graph in tqdm(test_dataloader, desc='Testing'):
            graph = graph.to(device)
            labels = graph.nodes['face'].data['y'].to(device)
            logits = model(graph)

            logits = torch.squeeze(logits)
            assert logits.dim() == 2 and logits.shape[
                1] == num_classes, f"Expected logits shape (N, {num_classes}), got {logits.shape}"

            loss = F.cross_entropy(logits, labels)
            test_loss += loss.item()
            _, predicted = torch.max(logits, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            predicted = predicted.to(device)

            test_iou.update(predicted, labels)

    test_accuracy = test_correct / test_total
    test_iou_value = test_iou.compute()
    print(f'Accuracy of the network on the test graphs: {100 * test_accuracy:.2f} %')
    print(f'Average test loss: {test_loss / len(test_dataloader):.4f}')
    print(f'Test IoU: {test_iou_value:.4f}')

    with open(log_file, 'a') as f:
        f.write(f"Test accuracy: {100 * test_accuracy:.2f} \n")
        f.write(f"Average test loss: {test_loss / len(test_dataloader):.4f} \n")
        f.write(f"Test IoU: {test_iou_value:.4f} \n")

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()

    # size_all_mb = (param_size + buffer_size) / 1024**2
    size_all_mb = param_size / 1024**2
    return size_all_mb

if __name__ == '__main__':

    # 假设数据存放在 'data' 目录下
    data_dir = 'D:\Data_and_code\graph_with_label_vecfeat_faceattr'
    dataset = HeteroGraphDataset(data_dir)

    # 划分训练集和测试集
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    # augmented_train_dataset = []
    # for graph in train_dataset:
    #     augmented_graph = copy.deepcopy(graph)
    #     augmented_graph = dataset._geo_augment(augmented_graph)
    #     augmented_train_dataset.append(graph)
    #     augmented_train_dataset.append(augmented_graph)
    # print(f"Number of augmented training graphs: {len(augmented_train_dataset)}")

    train_dataloader = GraphDataLoader(
        train_dataset,
        # augmented_train_dataset,
        batch_size=960,
        drop_last=False,
        shuffle=True
    )

    val_dataloader = GraphDataLoader(
        val_dataset,
        batch_size=960,
        drop_last=False,
        shuffle=False
    )
    test_dataloader = GraphDataLoader(
        test_dataset,
        batch_size=960,
        drop_last=False,
        shuffle=False
    )

    indim = 64
    # 定义隐藏层大小
    hidden_size = 128
    # 定义输出维度
    out_size = 8
    rel_names = ['adjacent_to', 'belongs_to']
    node_types = ['vertex', 'face']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_runs = 5
    test_accuracies = []
    test_losses = []
    test_ious = []

    for run in range(num_runs):
        print(f"Run {run + 1} of {num_runs}")

        # 创建模型实例
        model = HeteroGraphClassifier(indim, hidden_size, out_size, rel_names, node_types)
        model.to(device)

        # 定义优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7,
                                                               min_lr=1e-7)

        checkpoint_dir = 'D:\Data_and_code\LearnML\learn_360seg\checkpoints'

        start = time.time()
        # 训练模型
        model, log_file, best_checkpoint_path = train_model(model, train_dataloader, val_dataloader, optimizer,
                                                            epochs=140,
                                                            device=device, checkpoint_dir=checkpoint_dir)
        # 测试模型
        # 加载最佳检查点
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        num_classes = 8  # 根据你的输出维度修改
        test_iou = IoU(num_classes=num_classes, compute_on_step=False).to(device)
        test_iou.reset()

        test_correct = 0
        test_total = 0
        test_loss = 0
        with torch.no_grad():
            for graph in tqdm(test_dataloader, desc='Testing'):
                graph = graph.to(device)
                labels = graph.nodes['face'].data['y'].to(device)
                logits = model(graph)

                logits = torch.squeeze(logits)
                assert logits.dim() == 2 and logits.shape[
                    1] == num_classes, f"Expected logits shape (N, {num_classes}), got {logits.shape}"

                loss = F.cross_entropy(logits, labels)
                test_loss += loss.item()
                _, predicted = torch.max(logits, 1)

                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                predicted = predicted.to(device)

                test_iou.update(predicted, labels)

        test_accuracy = test_correct / test_total
        test_iou_value = test_iou.compute()
        test_loss_avg = test_loss / len(test_dataloader)

        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss_avg)
        test_ious.append(test_iou_value.item())

        print(f'Accuracy of the network on the test graphs: {100 * test_accuracy:.2f} %')
        print(f'Average test loss: {test_loss_avg:.4f}')
        print(f'Test IoU: {test_iou_value:.4f}')

        with open(log_file, 'a') as f:
            f.write(f"Run {run + 1} Test accuracy: {100 * test_accuracy:.2f} \n")
            f.write(f"Run {run + 1} Average test loss: {test_loss_avg:.4f} \n")
            f.write(f"Run {run + 1} Test IoU: {test_iou_value:.4f} \n")

        end = time.time()
        print(f'Time used for run {run + 1}: {(end - start) / 60} min')

    # 计算标准差
    accuracy_std = np.std(test_accuracies)
    loss_std = np.std(test_losses)
    iou_std = np.std(test_ious)

    print(f"Test accuracy standard deviation: {100 * accuracy_std:.2f} %")
    print(f"Test loss standard deviation: {loss_std:.4f}")
    print(f"Test IoU standard deviation: {iou_std:.4f}")








    # class_counts = torch.zeros(out_size)
    # for graph in train_dataset:
    #     labels = graph.nodes['face'].data['y']
    #     for label in labels:
    #         class_counts[label] += 1
    #
    # total_samples = class_counts.sum()
    # class_weights = total_samples / (out_size * class_counts)
    # class_weights = class_weights.to(device)

    # 创建模型实例
    # model = HeteroGraphClassifier(indim, hidden_size, out_size, rel_names, node_types)
    # model.to(device)
    #
    # # 定义优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3, amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-7)
    # # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #
    # checkpoint_dir = 'D:\Data_and_code\LearnML\learn_360seg\checkpoints'
    #
    # start = time.time()
    # # 训练模型
    # model, log_file, best_checkpoint_path = train_model(model, train_dataloader, val_dataloader, optimizer, epochs=140,
    #                                                     device=device, checkpoint_dir=checkpoint_dir)
    # # 测试模型
    # # 加载最佳检查点
    # checkpoint = torch.load(best_checkpoint_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # test_model(model, test_dataloader, device, log_file)
    #
    # model_size = get_model_size(model)
    # print(f"模型参数规模大小: {model_size:.2f} MB")
    #
    # end = time.time()
    # print(f'Time used: {(end - start) / 60} min')