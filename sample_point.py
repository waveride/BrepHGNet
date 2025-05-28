import math
import random

import dgl
import numpy as np
import torch
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.BRepLProp import BRepLProp_CLProps, BRepLProp_SLProps
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.GProp import GProp_GProps
from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.GeomLProp import GeomLProp_SLProps
from occwl.compound import Compound
from occwl.graph import face_adjacency
from occwl.io import load_step




def calculate_area(shape):
    props = GProp_GProps()
    brepgprop_SurfaceProperties(shape, props)
    return props.Mass()

def get_normal_at_uv(face, u, v):
    """获取指定UV坐标处的法向量"""

    face = BRep_Tool.Surface(face)
    props = GeomLProp_SLProps(face, 1, 1e-6)
    props.SetParameters(u, v)
    normal = props.Normal()
    # print(normal)
    return normal.X(), normal.Y(), normal.Z()

def uniform_sample_uv(u_min, u_max, v_min, v_max, n_points):
    """在给定的UV参数范围内均匀采样点"""
    samples = []
    n = int(math.sqrt(n_points))
    du = (u_max - u_min) / n - 1
    dv = (v_max - v_min) / n - 1
    for i in range(n):
        u = u_min + i * du
        for j in range(n):
            v = v_min + j * dv
            samples.append((u, v))
    return samples[:n_points]

def random_sample_uv(u_min, u_max, v_min, v_max, num_points):
    """在给定的UV参数范围内随机采样点"""

    samples = []
    for _ in range(num_points):
        u = random.uniform(u_min, u_max)
        v = random.uniform(v_min, v_max)
        samples.append((u, v))
    return samples


def sample_pts(stp_dir, number_pts=1024):

    '''
    1.读取stp模型，获取每个面
    2.计算面积
    3.输入采样总数，按面积采样
    4.将采样点坐标法向，加载到图节点上
    '''

    # occwl构建graph
    compound = Compound.load_from_step(stp_dir)
    solid = next(compound.solids())
    graph = face_adjacency(solid)

    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))


    # occ读面采点
    shape = read_step_file(stp_dir)
    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    faces = []
    face_areas = []
    while explorer.More():
        face = explorer.Current()

        faces.append(face)
        area = calculate_area(face)
        face_areas.append(area)
        explorer.Next()

    total_area = sum(face_areas)

    # 计算每个面应该采样的点数
    point_distribution = [max(1, int((area / total_area) * number_pts)) for area in face_areas]

    # 处理剩余点，平均分配至每个面
    remaining_points = number_pts - sum(point_distribution)
    if remaining_points > 0:
        num_faces = len(face_areas)
        avg_additional_points = remaining_points // num_faces
        extra_points = remaining_points % num_faces

        # 分配额外的点
        additional_points = [avg_additional_points + (1 if i < extra_points else 0) for i in range(num_faces)]

        # 更新点数分布
        point_distribution = [pd + ap for pd, ap in zip(point_distribution, additional_points)]

    i = 1
    # 采样点
    sampled_points = []
    for face, num_points in zip(faces, point_distribution):
        points_perface = []
        points_missface = []
        miss_pts = 0

        if num_points > 0:
            u_min, u_max, v_min, v_max = breptools_UVBounds(face)
            uv_samples = uniform_sample_uv(u_min, u_max, v_min, v_max, num_points)

            for u, v in uv_samples:
                pnt = BRep_Tool().Surface(face).Value(u, v)
                x, y, z = pnt.X(), pnt.Y(), pnt.Z()
                x, y, z = round(x, 3), round(y, 3), round(z, 3)

                nx, ny, nz = get_normal_at_uv(face, u, v)
                nx, ny, nz = round(nx, 3), round(ny, 3), round(nz, 3)

                points_perface.append([x, y, z, nx, ny, nz])


            # 随机采样
            if len(points_perface) < num_points:

                miss_pts = num_points - len(points_perface)

                random_uvsample = random_sample_uv(u_min, u_max, v_min, v_max, miss_pts)
                for u, v in random_uvsample:
                    pnt = BRep_Tool().Surface(face).Value(u, v)
                    ran_x, ran_y, ran_z = round(pnt.X(), 3), round(pnt.Y(), 3), round(pnt.Z(), 3)


                    rnx, rny, rnz = get_normal_at_uv(face, u, v)
                    rnx, rny, rnz = round(rnx, 3), round(rny, 3), round(rnz, 3)

                    points_missface.append([ran_x, ran_y, ran_z, rnx, rny, rnz])

            points_perface.append(points_missface)


        sampled_points.append(points_perface)




        print(f'该模型共{len(faces)}个面，第{i}个面理论采样：{num_points}，均匀采样：{len(points_perface)}，随机采样：{miss_pts}，实际采样：{len(points_perface) + miss_pts} ')
        i = i + 1
    face_feat = [np.array(face_points) for face_points in sampled_points]


    dgl_graph.ndata["x"] = torch.from_numpy(face_feat)


    return dgl_graph, point_distribution, sampled_points


def process_model():
    """
    1.construct a graph for stp model, face to node, edge to edge

    """
    pass


if __name__ == '__main__':
    testpath = 'D:\DatasetResult/360_seg/breps\step/16550_e88d6986_1.stp'
    graph, pt_dis, pts = sample_pts(testpath)
    print(graph, pt_dis, len(pts))



