import os
import cv2
import numpy as np
from main import get_args_parser as get_main_args_parser
import torch
from main import get_args_parser as get_main_args_parser
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from datasets import build_dataset
from models.DAB_DETR import build_DABDETR
from datasets.data_prefetcher import data_prefetcher
from util import box_ops
from torch import nn
import math 
from shapely.geometry import Polygon
from rbbox_overlaps import rbbx_overlaps # type: ignore
import datasets.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import torchvision.transforms as trans
from models.dab_deformable_detr import build_dab_deformable_detr

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
def get_all_rotated_boxes_vertices(boxes_grasp):
    """计算批次中所有旋转矩形的顶点坐标"""
    batch_size, num_boxes, _ = boxes_grasp.shape
    all_vertices = []
    for i in range(batch_size):
        vertices_list = []
        for j in range(num_boxes):
            x, y, w, h, alpha = boxes_grasp[i, j]
            vertices = get_rotated_box_vertices(x, y, w, h, alpha)
            vertices_list.append(vertices)
        all_vertices.append(vertices_list)
    return all_vertices

def load_model(model_path , args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] 当前使用{}做推断".format(device))
    model, _, _ = build_dab_deformable_detr(args)
    model.cuda()
    model.eval()
    state_dict = torch.load(model_path) # <-----------修改加载模型的路径
    # model.load_state_dict(state_dict["model"])
    missing_keys, unexpected_keys = model.load_state_dict(state_dict["model"], strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    model.to(device)
    print("load model sucess")
    return model

device=torch.device('cuda')
main_args = get_main_args_parser().parse_args()
    
    # 加载模型
model = load_model('/home/pmh/code/dab_pred_weight_deformable/results/checkpoint0135_79.pth', main_args)
model.eval()
imgfile = '/home/pmh/data/VMRD/data/images/train/00410.jpg'
# filename = 'D:/Desktop/data/grasp/train/00077.txt'
im = cv2.imread(imgfile)
img = Image.open(imgfile).convert('RGB')
a=trans.ToTensor()
# sample=torch.tensor(im).permute(2,0,1).unsqueeze(0).to(torch.device('cuda')).float()/255
sample=a(img).unsqueeze(0).to(torch.device('cuda'))

sample=F.normalize(sample, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
outputs = model(sample)
out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
out_logits_grasp=outputs['pred_logits']
# out_bbox_grasp=torch.cat((outputs['pred_boxes'], outputs['pred_angles']), dim=2)
out_angle_grasp=outputs['pred_angles']
prob_angles_grasp=out_angle_grasp.sigmoid()
angel_indice=torch.argmax(prob_angles_grasp, dim=2, keepdim=True)
angle=(angel_indice-8)*10
out_bbox_grasp=torch.cat((outputs['pred_boxes'],angle), dim=2)

prob = out_logits.sigmoid()
prob_grasp = out_logits_grasp.sigmoid()
topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
keep=topk_values>0.5
topk_values_grasp, topk_indexes_grasp = torch.topk(prob_grasp.view(out_logits.shape[0], -1), 100, dim=1)
scores = topk_values
scores_grasp=topk_values_grasp
topk_boxes = topk_indexes // out_logits.shape[2]
topk_boxes_grasp = topk_indexes_grasp // out_logits_grasp.shape[2]
labels = topk_indexes % out_logits.shape[2]
labels_grasp=topk_indexes_grasp %  out_logits_grasp.shape[2]
boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
boxes_grasp=torch.gather(out_bbox_grasp, 1, topk_boxes_grasp.unsqueeze(-1).repeat(1,1,5))

boxes_grasp_xy=torch.tensor(get_all_rotated_boxes_vertices(boxes_grasp)).to(device).view(1,100,8)
img_h, img_w = torch.tensor(sample.shape[2]),torch.tensor(sample.shape[3])
scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).unsqueeze(0).to(device)
boxes = boxes * scale_fct
scale_fct_grasp=torch.tensor([img_w, img_h, img_w, img_h,img_w, img_h, img_w, img_h]).unsqueeze(0).unsqueeze(0).to(device)
boxes_grasp_xy=boxes_grasp_xy * scale_fct_grasp
results = [{'scores': s, 'labels': l, 'boxes': b,'scores_grasp': s_grasp, 'labels_grasp': l_grasp,'boxes_grasp':b_grasp} 
                   for s, l, b, s_grasp, l_grasp, b_grasp in zip(scores, labels, boxes, scores_grasp, labels_grasp, boxes_grasp_xy)]

for i in range(20):
    location = boxes_grasp_xy[0][i]
    x0 = float(location[0])
    y0 = float(location[1])
    x1 = float(location[2])
    y1 = float(location[3])
    x2 = float(location[4])
    y2 = float(location[5])
    x3 = float(location[6])
    y3 = float(location[7])
    newBox = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    color = (4, 250, 7)
    point = np.array(newBox).astype(int)
    box_grasp = np.int32(point)
    
    # Draw the box contour on the image
    cv2.drawContours(im, [box_grasp], 0, (0, 0, 255), 2)
    scores=results[0]['scores_grasp'][i]
    labels=results[0]['labels_grasp'][i]
    scores='{:.2f}'.format(scores.tolist())
    text=str(scores)+' '+str(labels.tolist())
    # Label each point with its index
    # for idx, (x, y) in enumerate(point):
    #     cv2.circle(im, (x, y), 5, (0, 255, 0), -1)  # Draw a circle at each point
    cv2.putText(im, text, box_grasp[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Add index text
    
    # box=boxes[0][i]
    # a0=float(box[0])
    # b0=float(box[1])
    # a1=float(box[2])
    # b1=float(box[3])
    # newBox1=[[a0,b0],[a0,b1],[a1,b1],[a1,b0]]
    # point1 = np.array(newBox1).astype(int)
    # bbox = np.int32(point1)
    # cv2.drawContours(im, [bbox], 0, (0, 255, 255), 2)

# Display the image with the points labeled
# cv2.imshow('img', im)
cv2.imwrite('/home/pmh/code/dab_pred_weight_deformable/test.jpg', im)
cv2.waitKey(0)
cv2.destroyAllWindows()
