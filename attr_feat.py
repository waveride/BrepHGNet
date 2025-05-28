import multiprocessing
import os

from occwl.graph import face_adjacency
from occwl.solid import Solid
from sklearn.preprocessing import OneHotEncoder
import dgl
import torch
import numpy as np
import pathlib

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.TopExp import TopExp_Explorer
from dgl import load_graphs
from tqdm import tqdm
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import topods_Face, topods_Edge
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_LinearProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from functools import partial
import signal
from multiprocessing import Pool, Manager, cpu_count
from itertools import repeat
from OCC.Core.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface, GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola, GeomAbs_Parabola, GeomAbs_BSplineCurve, GeomAbs_BezierCurve, GeomAbs_OtherCurve
)


# 面类型映射
face_type_mapping = {
    GeomAbs_Plane: 0,
    GeomAbs_Cylinder: 1,
    GeomAbs_Cone: 2,
    GeomAbs_Sphere: 3,
    GeomAbs_BezierSurface: 4,
    GeomAbs_BSplineSurface: 5,
    GeomAbs_SurfaceOfRevolution: 6,
    GeomAbs_SurfaceOfExtrusion: 7,
    GeomAbs_OffsetSurface: 8,
    GeomAbs_OtherSurface: 9
}

edge_type_mapping = {
    GeomAbs_Line: 0,
    GeomAbs_Circle: 1,
    GeomAbs_Ellipse: 2,
    GeomAbs_Hyperbola: 3,
    GeomAbs_Parabola: 4,
    GeomAbs_BSplineCurve: 5,
    GeomAbs_BezierCurve: 6,
    GeomAbs_OtherCurve: 7
}


def calculate_face_type_and_area(face):
    """
    计算面的类型和面积
    """
    surf = topods_Face(face)
    face_type = BRepAdaptor_Surface(surf).GetType()
    face_type_id = face_type_mapping.get(face_type, -1)

    props = GProp_GProps()
    brepgprop_SurfaceProperties(topods_Face(face), props)
    area = round(props.Mass(), 3)

    return face_type_id, area


def calculate_edge_type_and_length(edge):
    """
    计算边的类型和长度
    """
    edge = topods_Edge(edge)
    edge_type = BRepAdaptor_Curve(edge).GetType()
    edge_type_id = edge_type_mapping.get(edge_type, -1)

    props = GProp_GProps()
    brepgprop_LinearProperties(topods_Edge(edge), props)
    length = round(props.Mass(), 3)

    return edge_type_id, length

def read_stp_file(file_path):
    file_path_str = str(file_path)
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path_str)
    if status == IFSelect_RetDone:
        reader.TransferRoots()
        shape = reader.Shape()
        return shape
    else:
        raise ValueError("无法正确读取STP文件")

def get_faces(shape):
    faces = []

    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    while face_exp.More():
        face = face_exp.Current()
        faces.append(face)
        face_exp.Next()

    return faces

def get_edges(shape):
    edges = []

    edge_exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_exp.More():
        edge = edge_exp.Current()
        edges.append(edge)
        edge_exp.Next()

    return edges

def one_hot_encode(labels, num_classes=9):
    """
    独热编码函数
    """
    one_hot = torch.zeros(len(labels), num_classes)
    for i, label in enumerate(labels):
        if label != -1:
            one_hot[i, label] = 1
    return one_hot

def min_max_normalize(data):
    """
    最小 - 最大归一化函数
    """
    min_val = torch.min(data)
    max_val = torch.max(data)
    if max_val - min_val == 0:
        return torch.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def process_one_file(brep_file_path, graph_file_path, output_folder):
    shape = read_stp_file(brep_file_path)
    solid = Solid(shape)
    face_adj_graph = face_adjacency(solid)
    edges = list(face_adj_graph.edges())

    faces = get_faces(shape)
    # edges = get_edges(shape)

    face_types = []
    areas = []
    for face in faces:
        face_type_id, area = calculate_face_type_and_area(face)
        face_types.append(face_type_id)
        areas.append(area)

    edge_types = []
    lengths = []
    for edge in edges:
        edge_type_id, length = calculate_edge_type_and_length(edge)
        edge_types.append(edge_type_id)
        lengths.append(length)

    graph = load_graphs(str(graph_file_path))[0][0]

    # 独热编码面类型
    face_num_classes = len(face_type_mapping)
    face_types_one_hot = one_hot_encode(face_types, face_num_classes)

    edge_num_classes = len(edge_type_mapping)
    edge_types_one_hot = one_hot_encode(edge_types, edge_num_classes)

    # 最小 - 最大归一化面积
    areas_tensor = torch.tensor(areas, dtype=torch.float32)
    areas_normalized = min_max_normalize(areas_tensor).unsqueeze(1)

    # 最小 - 最大归一化边长
    lengths_tensor = torch.tensor(lengths, dtype=torch.float32)
    lengths_normalized = min_max_normalize(lengths_tensor).unsqueeze(1)

    # 拼接处理后的属性
    attr_tensor = torch.cat((face_types_one_hot, areas_normalized), dim=1)
    edge_attr_tensor = torch.cat((edge_types_one_hot, lengths_normalized), dim=1)

    # print(attr_tensor.shape)

    # 将处理后的 2D 向量存储到异构图的 face 节点的 attr 属性中
    graph.nodes['face'].data['attr'] = attr_tensor
    graph.edges['adjacent_to'].data['attr'] = edge_attr_tensor

    # 保存修改后的异构图
    file_name = os.path.basename(graph_file_path)
    output_file_path = os.path.join(output_folder, file_name)
    dgl.data.utils.save_graphs(str(output_file_path), [graph])

def process_all_files(brep_folder, graph_folder, output_folder):
    # 创建输出文件夹
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 遍历 BREP 文件夹中的所有文件
    brep_files = sorted([os.path.join(brep_folder, f) for f in os.listdir(brep_folder) if f.endswith('.stp')])
    for brep_file in tqdm(brep_files, desc="Processing files"):
        file_name = os.path.basename(brep_file).replace('.stp', '.bin')
        graph_file = os.path.join(graph_folder, file_name)
        if os.path.exists(graph_file):
            process_one_file(brep_file, graph_file, output_folder)

# 创建一个处理任务的函数，固定 graph_folder 和 output_folder
    task_func = partial(process_one_file, graph_folder=graph_folder, output_folder=output_folder)

    # 使用多进程池并行处理
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(task_func, brep_files), total=len(brep_files), desc="Processing files"))

if __name__ == "__main__":
    brep_folder = r"D:\Data_and_code\s2.0.1\breps\step"
    graph_folder = r"D:\Data_and_code\graph_with_label_vec_feat"
    output_folder = r"D:\Data_and_code\graph_with_label_vec_face_edgeattr"

    process_all_files(brep_folder, graph_folder, output_folder)


    # file_path = r"D:\Data_and_code\s2.0.1\breps\step/16550_e88d6986_1.stp"
    # face_types, areas = process_one_file(file_path)
    # print(face_types)
    # print(areas)
