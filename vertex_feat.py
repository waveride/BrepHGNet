import numpy as np
from dgl.data.utils import load_graphs
import pathlib
import torch
import dgl
from tqdm import tqdm

def shape_vec(tensor, target_dim=64):
    # average pooling
    result = []
    for i in range(tensor.shape[0]):
        vector = tensor[i]
        original_length = vector.size(0)
        if original_length > target_dim:
            # 计算填充所需的元素数量

            num_segments = (original_length // target_dim) + 1
            pad_size = target_dim * num_segments - original_length

            padded_vector = torch.cat((vector, vector[:pad_size]))
            segments = padded_vector.view(target_dim, -1)
            # 计算平均值
            avg_vector = torch.mean(segments, dim=1)
            result.append(avg_vector)
        else:
            raise ValueError("The dimension of the tensor should be greater than target_dim.")
            # 堆叠结果
    result_tensor = torch.stack(result)
    return result_tensor

def load_g_with_vec_feat():
    dir_path = r"D:\Data_and_code"
    path = pathlib.Path(dir_path)

    file_path = path.joinpath('graph_with_label')
    output_folder = path.joinpath('graph_with_label_vec_feat')
    output_folder.mkdir(exist_ok=True)

    for file in tqdm(file_path.glob('*.bin'), desc="Processing graphs"):
        graphs = load_graphs(str(file))[0][0]
        # num_faces = graphs.num_nodes('face')
        # num_vertices = graphs.num_nodes('vertex')
        # # print("num_nodes:", num_faces, num_vertices)



        face_xyz = graphs.nodes['face'].data['x'].view(-1, 100, 7)[:, :, :3]
        vec_xyz = graphs.nodes['vertex'].data['x']

        distances = torch.cdist(vec_xyz,face_xyz.view(-1, 3))

        graphs.nodes['vertex'].data['x'] = shape_vec(distances)

        output_file_path = output_folder.joinpath(file.name)
        dgl.data.utils.save_graphs(str(output_file_path), [graphs])


if __name__ == '__main__':
    load_g_with_vec_feat()
    # dir_path = r"D:\DatasetResult\360_seg"
    # path = pathlib.Path(dir_path)
    #
    # file_path = path.joinpath('graph_mymethod')
    # label_path = path.joinpath('breps').joinpath('seg')
    #
    # graph_file = sorted([file for file in file_path.glob('*.bin')])
    # label_file = sorted([file for file in label_path.glob('*.seg')])
    #
    # # print(len(graph_file), graph_file[0], len(label_file), label_file[0], sep='\n')
    #
    # output_folder = path.joinpath('graph_with_label')
    # output_folder.mkdir(exist_ok=True)
    #
    # for file in tqdm(graph_file, desc="Processing graphs"):
    #
    #     label = np.loadtxt(label_path.joinpath(file.stem).with_suffix('.seg'), dtype=int, ndmin=1)
    #
    #     graphs = load_graphs(str(file))[0][0]
    #
    #     if graphs.num_nodes('face') != label.shape[0]:
    #         print(file.stem, "number of nodes not match")
    #         continue
    #
    #     graphs.nodes['face'].data['y'] = torch.tensor(label).long()
    #
    #     output_file_path = output_folder.joinpath(file.name)
    #     dgl.data.utils.save_graphs(str(output_file_path), [graphs])


