import torch
from matplotlib import pyplot as plt
import time

def get_rotated_vertices(center, width, height, angle):
    theta = torch.deg2rad(angle)
    b = torch.cos(theta) * 0.5
    a = torch.sin(theta) * 0.5
    return torch.stack([
        center[:, 0] - a * height - b * width, center[:, 1] + b * height - a * width,
        center[:, 0] + a * height - b * width, center[:, 1] - b * height - a * width,
        center[:, 0] + a * height + b * width, center[:, 1] - b * height + a * width,
        center[:, 0] - a * height + b * width, center[:, 1] + b * height + a * width
    ], dim=-1).view(-1, 4, 2)

def rectangle(boxes1, boxes2):
    # 计算左上角最小值
    lt = torch.min(boxes1[:, None, :, :], boxes2[None, :, :, :])  # 形状为 (N, M, 4, 2)
    lt = lt.min(dim=2).values  # 沿第3维度（4个顶点）取最小值，形状为 (N, M, 2)

    # 计算右下角最大值
    rb = torch.max(boxes1[:, None, :, :], boxes2[None, :, :, :])  # 形状为 (N, M, 4, 2)
    rb = rb.max(dim=2).values  # 沿第3维度（4个顶点）取最大值，形状为 (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    c = wh[:, :, 0] ** 2 + wh[:, :, 1] ** 2
    return wh , c

def draw_rectangles(vertices1, vertices2, lt, rb):
    plt.figure()
    ax = plt.gca()

    # 绘制第一个集合的矩形
    for i in range(vertices1.size(0)):
        rect_patch = plt.Polygon(vertices1[i].cpu().numpy(), fill=None, edgecolor='r')
        ax.add_patch(rect_patch)

    # 绘制第二个集合的矩形
    for i in range(vertices2.size(0)):
        rect_patch = plt.Polygon(vertices2[i].cpu().numpy(), fill=None, edgecolor='b')
        ax.add_patch(rect_patch)

    # 绘制生成的最小包围矩形
    for i in range(lt.size(0)):
        for j in range(lt.size(1)):
            rect_patch = plt.Polygon(torch.tensor([
                [lt[i, j, 0], lt[i, j, 1]],
                [rb[i, j, 0], lt[i, j, 1]],
                [rb[i, j, 0], rb[i, j, 1]],
                [lt[i, j, 0], rb[i, j, 1]]
            ]).cpu().numpy(), fill=None, edgecolor='g', linestyle='--')
            ax.add_patch(rect_patch)

    plt.xlim(min(torch.min(vertices1[:, :, 0]).item(), torch.min(vertices2[:, :, 0]).item()) - 10, 
             max(torch.max(vertices1[:, :, 0]).item(), torch.max(vertices2[:, :, 0]).item()) + 10)
    plt.ylim(min(torch.min(vertices1[:, :, 1]).item(), torch.min(vertices2[:, :, 1]).item()) - 10, 
             max(torch.max(vertices1[:, :, 1]).item(), torch.max(vertices2[:, :, 1]).item()) + 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def generalized_grasp_box_cost(grasps1, grasps2, out_angle, tgt_angle):
    # start = time.time()
    # 计算顶点
    vertices1 = get_rotated_vertices(grasps1[:, :2], grasps1[:, 2], grasps1[:, 3], out_angle.squeeze())
    vertices2 = get_rotated_vertices(grasps2[:, :2], grasps2[:, 2], grasps2[:, 3], tgt_angle.squeeze())

    # 计算x^2 + y^2
    d = grasps1[:, :2].unsqueeze(1) - grasps2[:, :2].unsqueeze(0)
    dist_squared = (d ** 2).sum(dim=-1)

    # 计算最小包围矩形
    wh, c = rectangle(vertices1, vertices2)
    #计算角度差
    # angle_diff = out_angle - tgt_angle
    angle_diff = out_angle - tgt_angle.T
    angle_diff = torch.deg2rad(angle_diff)
    #计算w、h的差值的平方
    # w_diff = (grasps1[:, 2] - grasps2[:, 2]) ** 2
    # h_diff = (grasps1[:, 3] - grasps2[:, 3]) ** 2

    grasps1_expanded = grasps1.unsqueeze(1)  # 形状变为 [200, 1, 4]
    grasps2_expanded = grasps2.unsqueeze(0)  # 形状变为 [1, 32, 4]

    diff = (grasps1_expanded - grasps2_expanded) ** 2
    w_diff = diff[:, :, 2]
    h_diff = diff[:, :, 3]

    #计算w、h的平方与lt-rb的差值的平方的比值
    ratio = w_diff / wh[:, :, 0] ** 2
    ratio += h_diff / wh[:, :, 1] ** 2
    #计算角度损失与中心距离损失系数
    alapha = 2 - dist_squared / c
    gamma = 2
    #计算总损失
    loss = gamma * dist_squared / c + alapha * (torch.sin(angle_diff))**2 + ratio 
    # end = time.time()
    # print(f"Time: {end - start}")

    # 绘制矩形
    # draw_rectangles(vertices1, vertices2, lt, rb)
    return loss

# 测试程序
# def test():
#     # 示例输入
#     grasps1 = torch.tensor([[10, 10, 5, 2], [20, 20, 4, 3]], dtype=torch.float32, device='cuda')  # 形状 (2, 4)
#     grasps2 = torch.tensor([[15, 15, 6, 3], [25, 25, 5, 4]], dtype=torch.float32, device='cuda')  # 形状 (2, 4)
#     out_angle = torch.tensor([[30], [45]], dtype=torch.float32, device='cuda')  # 形状 (2, 1)
#     tgt_angle = torch.tensor([[60], [90]], dtype=torch.float32, device='cuda')  # 形状 (2, 1)

#     # 调用函数
#     dist_squared = generalized_grasp_box_iou(grasps1, grasps2, out_angle, tgt_angle)

#     # 验证计算是否正确
#     print("验证距离平方计算：")
#     print(dist_squared)

# test()







































# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# def get_rotated_vertices(center, width, height, angle):
#     theta = torch.deg2rad(angle)
#     b = torch.cos(theta) * 0.5
#     a = torch.sin(theta) * 0.5
#     return torch.stack([
#         center[:, 0] - a * height - b * width, center[:, 1] + b * height - a * width,
#         center[:, 0] + a * height - b * width, center[:, 1] - b * height - a * width,
#         center[:, 0] + a * height + b * width, center[:, 1] - b * height + a * width,
#         center[:, 0] - a * height + b * width, center[:, 1] + b * height + a * width
#     ], dim=-1).view(-1, 4, 2)

# def polygon_area(vertices):
#     x = vertices[:, :, 0]
#     y = vertices[:, :, 1]
#     return 0.5 * torch.abs(torch.sum(x * torch.roll(y, 1, dims=1) - y * torch.roll(x, 1, dims=1), dim=1))

# def edge_intersection(p1, p2, q1, q2):
#     A = p2 - p1
#     B = q1 - q2
#     C = p1 - q1
#     det = A[0] * B[1] - A[1] * B[0]
#     t = (C[0] * B[1] - C[1] * B[0]) / det
#     u = (C[0] * A[1] - C[1] * A[0]) / det
#     return (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1)

# def compute_intersection(vertices1, vertices2):
#     intersections = torch.zeros(vertices1.size(0), vertices2.size(0), device=vertices1.device)
#     for i in range(vertices1.size(0)):
#         for j in range(vertices2.size(0)):
#             for k in range(4):
#                 for l in range(4):
#                     if edge_intersection(vertices1[i, k], vertices1[i, (k+1)%4], vertices2[j, l], vertices2[j, (l+1)%4]):
#                         intersections[i, j] += 1
#     return intersections

# def compute_iou_batch(vertices1, vertices2):
#     inter_area = compute_intersection(vertices1, vertices2)
#     area1 = polygon_area(vertices1)
#     area2 = polygon_area(vertices2)
#     union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
#     ious = inter_area / union_area
#     return ious

# def get_minimum_bounding_box(vertices1, vertices2):
#     all_vertices = torch.cat((vertices1.view(-1, 2), vertices2.view(-1, 2)), dim=0)
#     min_x, _ = torch.min(all_vertices[:, 0], dim=0)
#     max_x, _ = torch.max(all_vertices[:, 0], dim=0)
#     min_y, _ = torch.min(all_vertices[:, 1], dim=0)
#     max_y, _ = torch.max(all_vertices[:, 1], dim=0)
#     return torch.tensor([
#         [min_x, min_y],
#         [max_x, min_y],
#         [max_x, max_y],
#         [min_x, max_y]
#     ], device=vertices1.device)

# def draw_rectangles(vertices1, vertices2, bounding_box):
#     plt.figure()
#     ax = plt.gca()
#     for i in range(vertices1.size(0)):
#         rect1_patch = plt.Polygon(vertices1[i].cpu().numpy(), fill=None, edgecolor='r')
#         ax.add_patch(rect1_patch)
#     for j in range(vertices2.size(0)):
#         rect2_patch = plt.Polygon(vertices2[j].cpu().numpy(), fill=None, edgecolor='b')
#         ax.add_patch(rect2_patch)
#     bounding_box_patch = plt.Polygon(bounding_box.cpu().numpy(), fill=None, edgecolor='g', linestyle='--')
#     ax.add_patch(bounding_box_patch)
#     plt.xlim(torch.min(vertices1[:, :, 0]).item() - 10, torch.max(vertices1[:, :, 0]).item() + 10)
#     plt.ylim(torch.min(vertices1[:, :, 1]).item() - 10, torch.max(vertices1[:, :, 1]).item() + 10)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.show()

# def generalized_grasp_box_iou(grasps1, grasps2, out_angle, tgt_angle):

#     # 计算顶点
#     vertices1 = get_rotated_vertices(grasps1[:, :2], grasps1[:, 2], grasps1[:, 3], out_angle.squeeze())
#     vertices2 = get_rotated_vertices(grasps2[:, :2], grasps2[:, 2], grasps2[:, 3], tgt_angle.squeeze())
#     # 计算IoU
#     iou = compute_iou_batch(vertices1, vertices2)
#     print(f"IoU: {iou}")

#     # 计算最小非偏转外接矩形
#     bounding_box = get_minimum_bounding_box(vertices1, vertices2)
#     return iou
#     # 绘制结果
#     # draw_rectangles(vertices1, vertices2, bounding_box)

