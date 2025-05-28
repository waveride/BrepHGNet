import pathlib
import networkx as nx
import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_FACE
import dgl
import torch
from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Face
from OCC.Core._TopoDS import topods_Vertex
from matplotlib import pyplot as plt
from occwl.graph import face_adjacency, vertex_adjacency
from occwl.solid import Solid
from occwl.uvgrid import ugrid, uvgrid
import argparse
from multiprocessing.pool import Pool
from tqdm import tqdm
import signal
from itertools import repeat



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


def get_faces_and_vertices(shape):
    faces = []
    unique_vertices = []
    visited_xyzs = []

    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    while face_exp.More():
        face = face_exp.Current()
        faces.append(face)
        face_exp.Next()

    vertex_exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    while vertex_exp.More():
        vertex = vertex_exp.Current()
        point = BRep_Tool.Pnt(topods_Vertex(vertex))
        # 使用顶点坐标作为唯一标识
        x = round(point.X(), 5)
        y = round(point.Y(), 5)
        z = round(point.Z(), 5)

        vertex_key = (x, y, z)
        if vertex_key not in visited_xyzs:
            visited_xyzs.append(vertex_key)
            unique_vertices.append(vertex)
        vertex_exp.Next()


    return faces, unique_vertices, visited_xyzs

def is_vertex_on_face(vertexs, faces):
    """
    判断顶点是否在面上
    :param vertex: 顶点
    :param face: 面
    :return:返回顶点在面的索引
    """


    face_nodes = []
    vertex_nodes = []

    for i, face in enumerate(faces):
        vertex_in_face = []
        is_vex = TopExp_Explorer(face, TopAbs_VERTEX)
        while is_vex.More():
            vex = is_vex.Current()
            vertex_in_face.append(vex)
            is_vex.Next()

        for j, vertex in enumerate(vertexs):
            if vertex in vertex_in_face:
                face_nodes.append(int(i))
                vertex_nodes.append(int(j))


    return vertex_nodes,face_nodes

def get_heterograph(vertexs, faces, shape, vertexs_xyz, curv_num_u_samples = 10, curv_num_v_samples = 10, surf_num_u_samples=10, surf_num_v_samples=10):
    """
    获取异构图
    :param faces: 面列表
    :param vertexs: 顶点列表
    :return: 异构图
    """
    solid = Solid(shape)
    face_adj_graph = face_adjacency(solid)


    face_adj_edges = list(face_adj_graph.edges())
    vertex_nodes = torch.tensor(vertexs, dtype=torch.int64)
    face_nodes = torch.tensor(faces, dtype=torch.int64)



    graph_face_feat = []

    for face_idx in face_adj_graph.nodes():
        # 获取B-rep面
        face = face_adj_graph.nodes[face_idx]["face"]

        # Compute UV-grids
        points = uvgrid(
            face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        normals = uvgrid(
            face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # Concatenate channel-wise to form face feature tensor
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat.append(face_feat)
    graph_face_feat = np.asarray(graph_face_feat)
    # print(graph_face_feat.shape)

    # Compute the U-grids for edges   点坐标和切线坐标
    graph_edge_feat = []
    for edge_idx in face_adj_graph.edges():
        # Get the B-rep edge
        edge = face_adj_graph.edges[edge_idx]["edge"]
        # Ignore dgenerate edges, e.g. at apex of cone
        if not edge.has_curve():
            continue
        # Compute U-grids
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        # Concatenate channel-wise to form edge feature tensor
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    graph_vertex_feat = np.asarray(vertexs_xyz)


    graph_data = {
        ('face', 'adjacent_to', 'face'): (torch.tensor([e[0] for e in face_adj_edges], dtype=torch.int64),
                                          torch.tensor([e[1] for e in face_adj_edges], dtype=torch.int64)),
        ('vertex', 'belongs_to', 'face'): (vertex_nodes, face_nodes)

    }

    # print("图数据:")
    # for key, (src, dst) in graph_data.items():
    #     print(f"{key}: src 数据类型: {src.dtype}, dst 数据类型: {dst.dtype}")

    g = dgl.heterograph(graph_data)
    g.nodes['face'].data['x'] = torch.from_numpy(graph_face_feat)
    g.nodes['vertex'].data['x'] = torch.from_numpy(graph_vertex_feat)
    g.edges['adjacent_to'].data['x'] = torch.from_numpy(graph_edge_feat)


    return g

def process_one_shape(arguments):
    """
    处理单个stp文件
    """
    fn, args = arguments
    fn_stem = fn.stem
    output_path = pathlib.Path(args.output)
    shape = read_stp_file(fn)  # Assume there's one solid per file
    faces, vertexs, visited_xyzs = get_faces_and_vertices(shape)
    idx = is_vertex_on_face(vertexs, faces)

    # print(f"Vertex nodes type: {type(idx[0])}, Face nodes type: {type(idx[1])}")
    # print(f"Vertex nodes sample: {idx[0][:5]}, Face nodes sample: {idx[1][:5]}")

    graph = get_heterograph(idx[0], idx[1], shape, visited_xyzs, curv_num_u_samples=args.curv_num_u_samples, surf_num_u_samples=args.surf_num_u_samples, surf_num_v_samples=args.surf_num_v_samples)

    dgl.data.utils.save_graphs(str(output_path.joinpath(fn_stem + ".bin")), [graph])


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def process_all_shapes(args):
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    step_files = list(input_path.glob("*.st*p"))
    pool = Pool(processes=args.num_processes, initializer=initializer)
    try:
        results = list(tqdm(pool.imap(process_one_shape, zip(step_files, repeat(args))), total=len(step_files)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    print(f"Processed {len(results)} files.")

def main():
    parser = argparse.ArgumentParser(
        "Convert solid models to face-adjacency graphs with UV-grid features"
    )
    parser.add_argument("input", type=str, help="Input folder of STEP files")
    parser.add_argument("output", type=str, help="Output folder of DGL graph BIN files")
    parser.add_argument(
        "--curv_num_u_samples", type=int, default=10, help="Number of samples on each curve"
    )
    parser.add_argument(
        "--surf_num_u_samples",
        type=int,
        default=10,
        help="Number of samples on each surface along the u-direction",
    )
    parser.add_argument(
        "--surf_num_v_samples",
        type=int,
        default=10,
        help="Number of samples on each surface along the v-direction",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes to use",
    )
    args = parser.parse_args()
    process_all_shapes(args)




if __name__ == '__main__':
    main()

    # file_path = "D:\DatasetResult/360_seg/breps\step/16550_e88d6986_0.stp"
    # shape = read_stp_file(file_path)
    # shape_test = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
    #
    # faces, vertexs, visited_xyzs = get_faces_and_vertices(shape)
    # idx= is_vertex_on_face(vertexs, faces)
    # print(idx, visited_xyzs)
    #
    # g = get_heterograph(idx[0], idx[1], shape_test, visited_xyzs)
    # print(g, g.nodes['vertex'].data['x'],sep='\n')
    # # visualize_heterograph(g)


