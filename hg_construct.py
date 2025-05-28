

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.STEPControl import STEPControl_AsIs
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt, gp_Vec
import dgl
import torch
from OCC.Core.TopoDS import topods_Face, topods_Vertex, topods_Edge, topods_Wire
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere




def read_step_file(file_name):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(file_name)

    if status == 1:  # 检查文件是否成功读取
        step_reader.TransferRoots()
        shape = step_reader.OneShape()


        return shape
    else:
        raise Exception("Error: Can't read file.")


def get_unique_faces(shape):
    """从形状中获取所有唯一的面"""
    unique_faces = set()
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)

    while face_explorer.More():
        face = topods_Face(face_explorer.Current())
        face_hash = face.HashCode(1 << 24)  # 生成唯一的哈希码
        unique_faces.add(face_hash)
        face_explorer.Next()

    return unique_faces


def get_unique_vertices(shape):
    """从形状中获取所有唯一的顶点"""
    unique_vertices = set()
    vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)

    while vertex_explorer.More():
        vertex = topods_Vertex(vertex_explorer.Current())
        vertex_hash = vertex.HashCode(1 << 24)  # 生成唯一的哈希码
        unique_vertices.add(vertex_hash)
        vertex_explorer.Next()

    return unique_vertices




def build_heterograph(shape):
    # 获取所有唯一的面和顶点
    unique_faces = get_unique_faces(shape)
    unique_vertices = get_unique_vertices(shape)

    print(f'该模型共有 {len(unique_faces)} 个面, {len(unique_vertices)} 个顶点')

    # 映射面和顶点到索引
    face_to_idx = {face_hash: idx for idx, face_hash in enumerate(unique_faces)}
    vertex_to_idx = {vertex_hash: idx for idx, vertex_hash in enumerate(unique_vertices)}

    # 初始化边列表
    # face_interact_face_edges = ([], [])
    vertex_subject_face_edges = ([], [])
    vertex_interact_vertex_edges = ([], [])

    # 遍历所有面，构建  ('vertex', 'subject', 'face') 边
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods_Face(face_explorer.Current())
        face_hash = face.HashCode(1 << 24)
        face_idx = face_to_idx[face_hash]

        print(f"Processing face {face_idx}")

        # 获取面的顶点，并确保不重复记录
        vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
        processed_vertices = set()
        while vertex_explorer.More():
            vertex = topods_Vertex(vertex_explorer.Current())
            vertex_hash = vertex.HashCode(1 << 24)
            vertex_idx = vertex_to_idx[vertex_hash]
            if vertex_hash not in processed_vertices:
                vertex_subject_face_edges[0].append(vertex_idx)
                vertex_subject_face_edges[1].append(face_idx)
                processed_vertices.add(vertex_hash)
            vertex_explorer.Next()



        # # 获取面的相邻面
        # edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        # while edge_explorer.More():
        #     edge = topods_Edge(edge_explorer.Current())
        #     adjacent_faces = []
        #     face_explorer_edge = TopExp_Explorer(edge, TopAbs_FACE)
        #     while face_explorer_edge.More():
        #         adjacent_face = topods_Face(face_explorer_edge.Current())
        #         adjacent_face_hash = adjacent_face.HashCode(1 << 24)
        #         if adjacent_face_hash != face_hash:
        #             adjacent_faces.append(adjacent_face_hash)
        #         face_explorer_edge.Next()
        #
        #     for adjacent_face_hash in adjacent_faces:
        #         if adjacent_face_hash in face_to_idx:
        #             adjacent_face_idx = face_to_idx[adjacent_face_hash]
        #             face_interact_face_edges[0].append(face_idx)
        #             face_interact_face_edges[1].append(adjacent_face_idx)

        face_explorer.Next()

    # 遍历所有顶点，构建 ('vertex', 'interact_vertex', 'vertex') 边
    vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = topods_Vertex(vertex_explorer.Current())
        vertex_hash = vertex.HashCode(1 << 24)
        vertex_idx = vertex_to_idx[vertex_hash]

        print(f"Processing vertex {vertex_idx}")

        # 获取顶点的相邻顶点
        edge_explorer = TopExp_Explorer(vertex, TopAbs_EDGE)
        while edge_explorer.More():
            edge = topods_Edge(edge_explorer.Current())
            vertex_explorer_edge = TopExp_Explorer(edge, TopAbs_VERTEX)
            while vertex_explorer_edge.More():
                adjacent_vertex = topods_Vertex(vertex_explorer_edge.Current())
                adjacent_vertex_hash = adjacent_vertex.HashCode(1 << 24)
                if adjacent_vertex_hash != vertex_hash and adjacent_vertex_hash in vertex_to_idx:
                    adjacent_vertex_idx = vertex_to_idx[adjacent_vertex_hash]
                    vertex_interact_vertex_edges[0].append(vertex_idx)
                    vertex_interact_vertex_edges[1].append(adjacent_vertex_idx)
                vertex_explorer_edge.Next()

        vertex_explorer.Next()

    # 构建异构图
    hg = dgl.heterograph({

        ('vertex', 'subject', 'face'): vertex_subject_face_edges,
        ('vertex', 'interact_vertex', 'vertex'): vertex_interact_vertex_edges
    })

    return hg

if __name__ == '__main__':
    file_path = 'D:\DatasetResult/360_seg/breps\step/16550_e88d6986_0.stp'  # 替换为你的STEP文件路径
    shape = read_step_file(file_path)
    shape_test = BRepPrimAPI_MakeBox(1, 1, 1).Shape()

    hg = build_heterograph(shape)

    print("Heterogeneous Graph:")
    print(hg)



    #
    # graph_data = {
    #     ('vertex', 'subject', 'face'):(face_vertex_id[1], face_vertex_id[0])
    # }
    #
    # # print(vertex_id)
    #
    #
    # g.nodes['face'].data['d'] = torch.ones(19, 1)
    #
    # print(face_vertex_id)

    # graph_data = {
    #     ('face', 'interact_face', 'face'): (th.tensor([0, 1]), th.tensor([1, 2])),
    #     ('vertex', 'subject', 'face'): (th.tensor([0, 1]), th.tensor([2, 3])),
    #     ('vertex', 'interact_vertex', 'vertex'): (th.tensor([1]), th.tensor([2]))
    # }
    #
    # g = dgl.heterograph(graph_data)
    #
    # print(g.num_nodes())

