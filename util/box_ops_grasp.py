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

    lt = torch.min(boxes1[:, None, :, :], boxes2[None, :, :, :])  
    lt = lt.min(dim=2).values
    rb = torch.max(boxes1[:, None, :, :], boxes2[None, :, :, :])  
    rb = rb.max(dim=2).values 

    wh = (rb - lt).clamp(min=0) 
    c = wh[:, :, 0] ** 2 + wh[:, :, 1] ** 2
    return wh , c

def draw_rectangles(vertices1, vertices2, lt, rb):
    plt.figure()
    ax = plt.gca()

    for i in range(vertices1.size(0)):
        rect_patch = plt.Polygon(vertices1[i].cpu().numpy(), fill=None, edgecolor='r')
        ax.add_patch(rect_patch)

    for i in range(vertices2.size(0)):
        rect_patch = plt.Polygon(vertices2[i].cpu().numpy(), fill=None, edgecolor='b')
        ax.add_patch(rect_patch)

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

    vertices1 = get_rotated_vertices(grasps1[:, :2], grasps1[:, 2], grasps1[:, 3], out_angle.squeeze())
    vertices2 = get_rotated_vertices(grasps2[:, :2], grasps2[:, 2], grasps2[:, 3], tgt_angle.squeeze())

    d = grasps1[:, :2].unsqueeze(1) - grasps2[:, :2].unsqueeze(0)
    dist_squared = (d ** 2).sum(dim=-1)

    wh, c = rectangle(vertices1, vertices2)

    angle_diff = out_angle - tgt_angle.T
    angle_diff = torch.deg2rad(angle_diff)

    grasps1_expanded = grasps1.unsqueeze(1) 
    grasps2_expanded = grasps2.unsqueeze(0) 

    diff = (grasps1_expanded - grasps2_expanded) ** 2
    w_diff = diff[:, :, 2]
    h_diff = diff[:, :, 3]

    ratio = w_diff / wh[:, :, 0] ** 2
    ratio += h_diff / wh[:, :, 1] ** 2

    alapha = 2 - dist_squared / c
    gamma = 2

    loss = gamma * dist_squared / c + alapha * (torch.sin(angle_diff))**2 + ratio 

    return loss