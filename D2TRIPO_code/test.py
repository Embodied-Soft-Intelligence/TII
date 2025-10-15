import cv2
import os
import numpy as np

# filename='/home/pmh/data/VMRD/data/grasps/train/00443.txt'
# imgfile='/home/pmh/data/VMRD/data/images/train/00443.jpg'
# im=cv2.imread(imgfile)
# with open(filename, 'r') as f:
# # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。
# # 然后将每个元素中的不同信息提取出来
#     lines = f.readlines()
#     for line in lines:
#         location = line.split(' ')
#         x0 = float(location[0])
#         y0 = float(location[1])
#         x1 = float(location[2])
#         y1 = float(location[3])
#         x2 = float(location[4])
#         y2 = float(location[5])
#         x3 = float(location[6])
#         y3 = float(location[7])
#         newBox = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
#         color = (4, 250, 7)
#         point = np.array(newBox).astype(int)
#         box = np.int32(point)
        
#         # Draw the box contour on the image
#         cv2.drawContours(im, [box], 0, (0, 0, 255), 2)

#         # Label each point with its index
#         # for idx, (x, y) in enumerate(point):
#         #     cv2.circle(im, (x, y), 5, (0, 255, 0), -1)  # Draw a circle at each point
#         #     cv2.putText(im, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Add index text
# img_name = os.path.basename(imgfile)
# # output_path = os.path.join(output_folder, img_name)
# cv2.imwrite('/home/pmh/code/dab_pred_weight_deformable/test.jpg', im)
# # print(f"Processed image saved to: {output_path}")
import math
from shapely.geometry import Polygon
import torch
from oriented_iou_loss import cal_iou
def get_rotated_box_vertices(x, y, w, h, alpha):
    """计算旋转矩形的四个顶点坐标"""
    # 中心点
    cx, cy = x, y
    # 半宽和半高
    hw, hh = w / 2, h / 2
    alpha=math.radians(alpha)
    # 旋转角度的余弦和正弦值
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    
    # 四个顶点坐标
    vertices = [
        (cx - hw * cos_alpha + hh * sin_alpha, cy - hw * sin_alpha - hh * cos_alpha),
        (cx + hw * cos_alpha + hh * sin_alpha, cy + hw * sin_alpha - hh * cos_alpha),
        (cx + hw * cos_alpha - hh * sin_alpha, cy + hw * sin_alpha + hh * cos_alpha),
        (cx - hw * cos_alpha - hh * sin_alpha, cy - hw * sin_alpha + hh * cos_alpha)
    ]
    return vertices
def calculate_iou(box1, box2):
    """计算两个旋转矩形框的IoU"""
    # 获取两个旋转矩形的顶点坐标
    vertices1 = get_rotated_box_vertices(*box1)
    vertices2 = get_rotated_box_vertices(*box2)
    
    # 创建两个多边形
    poly1 = Polygon(vertices1)
    poly2 = Polygon(vertices2)
    
    # 计算交集和并集面积
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    
    # 计算IoU
    iou = inter_area / union_area
    return iou
# tensor([256.1250, 308.5475, 206.3574,  78.7260,  -5.8539], device='cuda:0')
# tensor([272.4833, 376.2501, 217.8436, 103.1684,   0.0000], device='cuda:0')
from rbbox_overlaps import rbbx_overlaps 
a=calculate_iou([100, 100, 100, 100,0],[100, 100, 100, 100,45])
c=rbbx_overlaps(np.array([[100, 100, 100, 100,0]],dtype=np.float32),np.array([[100, 100, 100, 100,45]],dtype=np.float32))
b=cal_iou(torch.tensor([[100, 100, 100, 100,0],[256.1250, 308.5475, 206.3574,  78.7260,  -5.8539]],device='cuda:0').view(1,2,-1),torch.tensor([[100, 100, 100, 100,45],[272.4833, 376.2501, 217.8436, 103.1684,   0.0000]],device='cuda:0').view(1,2,-1))
print(a)
print(b[0])
print(c)

from mmcv.ops import box_iou_rotated
d=box_iou_rotated(torch.tensor([[100., 100., 100., 100.,0.]],device='cuda:0').view(1,1,-1),torch.tensor([[100., 100., 100., 100.,45.]],device='cuda:0').view(1,1,-1))
print(d)