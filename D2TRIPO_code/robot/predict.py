import sys
sys.path.append('/home/pmh/Code/dab_deformable_resnet_5d_Rotate_Point_QK_nobias/')
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as trans
import torchvision.transforms.functional as F
from main import get_args_parser as get_main_args_parser
from models.DAB_DETR import build_DABDETR
from util import box_ops
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from models.dab_deformable_detr import build_dab_deformable_detr
from shapely.geometry import Polygon
from torch import nn
from realsenseD435 import Camera
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from UR_Robot import UR_Robot
from threading import Thread
# 绘制旋转矩形的函数
def draw_rotated_rectangle(image, x, y, w, h, angle, color=(151, 169, 91), thickness=2):
    # 创建旋转矩形
    rect = ((x, y), (w, h), angle)
    # 获取旋转矩形的四个顶点
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # 将顶点坐标转换为整数
    # 绘制旋转矩形
    cv2.polylines(image, [box], isClosed=True, color=color, thickness=thickness)

def draw_rectangle(image, xmin, ymin, w, h, idx, lable_id, color=(138, 239, 242), thickness=2):
    # 绘制普通矩形
    label=['banana','watch','pliers','toothbrush','wrench',
            'towel','knife','badminton','tape','charger',
            'pen','box','umbrella','apple','wallet',
            'shaver','paper','cup','mobile phone','wrist developer', 
            'bottle','headset','toothpaste','glasses','remote controller',
            'card','mouse','cans','screwdriver','notebook',
            'stapler']
    text=str(idx)+' '+label[lable_id]
    xmin, ymin, w, h = int(xmin), int(ymin), int(w), int(h)
    cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), color, thickness)
    cv2.putText(image, text, (xmin, ymin), cv2.FONT_HERSHEY_DUPLEX, 0.6, (43, 44, 51), 2)
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_adjs=outputs['pred_adj']
        out_logits_grasp=outputs['pred_logits_grasp']
        out_angle_grasp=outputs['pred_angles_grasp']
        
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob_angles_grasp=out_angle_grasp.sigmoid()
        angel_indice=torch.argmax(prob_angles_grasp, dim=2, keepdim=True)
        angle=(angel_indice-8)*10-5
        out_bbox_grasp=torch.cat((outputs['pred_boxes_grasp'],angle), dim=2)
        topk_values_angles_grasp, topk_indexes_angles_grasp = torch.topk(prob_angles_grasp.view(out_angle_grasp.shape[0], -1), 100, dim=1)
        prob = out_logits.sigmoid()
        prob_grasp = out_logits_grasp.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        keep=topk_values>0.4
        topk_values_grasp, topk_indexes_grasp = torch.topk(prob_grasp.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        scores_grasp=topk_values_grasp
        # scores_angle_grasp=topk_values_angles_grasp
        topk_boxes = topk_indexes // out_logits.shape[2]
        topk_boxes_grasp = topk_indexes_grasp // out_logits_grasp.shape[2]
        topk_angles_grasp = topk_indexes_angles_grasp // out_angle_grasp.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        labels_grasp=topk_indexes_grasp %  out_logits_grasp.shape[2]
        # labels_angles_grasp=topk_indexes_angles_grasp %  out_angle_grasp.shape[2]
        adjs=[]
        for i in range(len(topk_boxes)):
            keep_topk_boxes=topk_boxes[i][keep[i]]
            keep_labels=labels[i][keep[i]]
            adj=out_adjs[i][keep_topk_boxes][:,keep_topk_boxes]
            adj=torch.sigmoid(adj)
            adjs.append((keep_labels,adj))
        
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        boxes_grasp=torch.gather(out_bbox_grasp, 1, topk_boxes_grasp.unsqueeze(-1).repeat(1,1,5))
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        angle_fct=torch.ones(1).to('cuda:0')
        scale_fct_grasp= torch.stack([img_w, img_h, img_w, img_h,angle_fct], dim=1)
        boxes_grasp=boxes_grasp* scale_fct_grasp[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b,'scores_grasp': s_grasp, 'labels_grasp': l_grasp,'boxes_grasp':b_grasp} 
                   for s, l, b, s_grasp, l_grasp, b_grasp in zip(scores, labels, boxes, scores_grasp, labels_grasp, boxes_grasp)]
        
        return results,adjs
def get_rotated_box_vertices(x, y, w, h, alpha):
    """计算旋转矩形的四个顶点坐标"""
    cx, cy = x, y
    hw, hh = w / 2, h / 2
    alpha = math.radians(alpha)
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)

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


def load_model(model_path, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] 当前使用{}做推断".format(device))
    model, _, _ = build_dab_deformable_detr(args)
    model.cuda()
    model.eval()
    state_dict = torch.load(model_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict["model"], strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    model.to(device)
    print("load model success")
    return model
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

def find_best_grasp(all_dt_grasp,all_dt):
    iou_gra_gt=[]
    all_bbox=[]
    idx_gt=[]
    all_idx_gt=[]
    e=[]
    for i in range(len(all_dt)):
        bbox0 = all_dt[i]['bbox']  # 假设 bbox0 是一个 tensor，形状为 (4,)
        bbox=[0]*4
        bbox[0]=bbox0[0]+bbox0[2]/2
        bbox[1]=bbox0[1]+bbox0[3]/2
        bbox[2]=bbox0[2]
        bbox[3]=bbox0[3]
        bbox.append(0)            # 在 list 的末尾添加一个 0        
        all_bbox.append(bbox)            # 收集到列表 b 中
    # all_bbox = np.array(all_bbox,dtype=np.float32)             # 最终将列表 b 转换为 NumPy 数组，形状为 (n, 5)

    for i in range(len(all_dt_grasp)):
        for j in range(len(all_dt)):
            if all_dt_grasp[i]['image_id']==all_dt[j]['image_id'] and all_dt_grasp[i]['category_id_grasp']==all_dt[j]['category_id']:
            # if all_dt_grasp[i]['image_id']==all_gt[j]['image_id'] :
                # bbox = np.array([all_bbox[j]])
                bbox=all_bbox[j]
                # bbox_grasp=np.array([all_dt_grasp[i]['bbox_grasp']],dtype=np.float32)
                bbox_grasp=all_dt_grasp[i]['bbox_grasp']
                # ious = rbbx_overlaps(bbox,bbox_grasp)
                ious=calculate_iou(bbox,bbox_grasp)
                iou_gra_gt.append(ious)
                idx_gt.append(j)   
        if iou_gra_gt:      
            index_of_max_value = np.argmax(iou_gra_gt)
            all_idx_gt.append(idx_gt[index_of_max_value])
        else:
            all_idx_gt.append(None)
        iou_gra_gt=[]
        idx_gt=[]   
        # a={'image_id':all_gt[index_of_max_value]['image_id'],'category_id':all_gt[index_of_max_value]['category_id'],}
    score=[]
    best_grasp=[]
    for i in range(len(all_dt)):
        if i in all_idx_gt:
            indices = [j for j, x in enumerate(all_idx_gt) if x == i]  # 查找该数的索引
            for _ , y in  enumerate(indices):
                score.append(all_dt_grasp[y]['score_grasp'])
            tmp=max(score)
            max_idx=score.index(tmp)
            best_grasp.append(all_dt_grasp[indices[max_idx]])
            score=[]
        else:
            e.append(i)
    return best_grasp
def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
def prepare_for_coco_detection(predictions):
    coco_results = []
    grasp_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes_grasp = prediction["boxes_grasp"].tolist()
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        scores_grasp = prediction["scores_grasp"].tolist()
        labels = prediction["labels"].tolist()
        labels_grasp = prediction["labels_grasp"].tolist()
        grasp_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id_grasp": labels_grasp[i],
                    "bbox_grasp": box,
                    "score_grasp": scores_grasp[i],
                }
                for i, box in enumerate(boxes_grasp)
            ]
        )
        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results,grasp_results
def draw_adj(matrix,img_name,output_folder2):
    import matplotlib.pyplot as plt
    # 创建热力图
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    # plt.colorbar()  # 添加颜色条
    plt.grid(color='red', linewidth=1, linestyle='-', alpha=0.7)

    # 设置 x 轴和 y 轴的刻度
    plt.xticks(ticks=np.arange(0, matrix.shape[1]), labels=np.arange(1, matrix.shape[1] + 1))
    plt.yticks(ticks=np.arange(0, matrix.shape[0]), labels=np.arange(1, matrix.shape[0] + 1))

    # 显示网格线
    plt.gca().set_xticks(np.arange(0.5, matrix.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(0.5, matrix.shape[0], 1), minor=True)
    # 绘制小刻度的网格线
    plt.gca().grid(which='minor', color='blue', linestyle='-', linewidth=2)

    # 隐藏主刻度的网格线
    plt.gca().grid(which='major', linestyle='None')  # 不绘制主刻度的网格线
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')  # 可选：如果需要设置标签位置
    base_name = os.path.splitext(img_name)[0]
    png_name = base_name + '.png'
    output_filepath = os.path.join(output_folder2, png_name)
    
    # 保存热力图
    plt.savefig(output_filepath)
    plt.close()

    print(f"热力图已保存到：{output_filepath}")
def grasp_order(a):
    A=[]
    B=[]
    b=np.eye(len(a))
    for _ in range(len(a)-1):
        if a.any()==0:
            # print('No relationship detected')
            B=list(range(len(a)))
            break
        else:
            b=np.dot(b,a)
            if not b.any()==0:
                A.append(b)
            else:
                break
    if A:
        c=np.argwhere(A[-1]>=1)
        for _ , x in enumerate(c):
            B.append(x[0])
    if B:
        return B
    else:
        return [0]
def extract_max_depth_in_rotated_rect(depth_image, rect):
    """
    提取深度图像中旋转矩形框所包含区域的最大深度值。

    :param depth_image: 深度图像，numpy 数组，形状为 (480, 640)。
    :param rect: 旋转矩形框，格式为 (x, y, w, h, alpha)，
                 其中 x, y 为中心坐标，w, h 为宽和高，alpha 为旋转角度（以度为单位）。
    :return: 矩形框内的最大深度值。
    """
    # 解构矩形框参数
    x, y, w, h, alpha = rect[0]

    # 将角度转换为弧度
    alpha_rad = np.deg2rad(alpha)

    # 生成旋转矩形框的顶点
    rect_vertices = cv2.boxPoints(((x, y), (w, h), alpha))
    rect_vertices = np.int0(rect_vertices)  # 转换为整数坐标

    # 创建一个掩码图像，大小与深度图像相同
    mask = np.zeros_like(depth_image, dtype=np.uint8)

    # 在掩码上绘制旋转矩形框
    cv2.fillPoly(mask, [rect_vertices], 255)

    # 使用掩码提取深度图像中的感兴趣区域
    masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)

    # 获取矩形框区域的最大深度值
    min_depth = np.min(masked_depth[np.nonzero(masked_depth)])

    return min_depth
def visualize_matrix(matrix, color_0=[211/255,235/255,248/255], color_1=[246/255,174/255,69/255], edge_color=[139/255,139/255,140/255], 
                     edge_width=5, font_size=12, save_path=True):
    """
    可视化0-1矩阵的函数，支持RGB颜色定义，并调整网格和坐标显示。
    
    参数:
        - matrix: numpy.ndarray, 包含0和1的二维矩阵
        - color_0: tuple, 表示0对应的RGB颜色，默认白色 (1, 1, 1)
        - color_1: tuple, 表示1对应的RGB颜色，默认黑色 (0, 0, 0)
        - edge_color: tuple, 方格网格线的RGB颜色，默认灰色 (0.5, 0.5, 0.5)
        - edge_width: int, 网格线和外围边框的宽度
        - font_size: int, 索引数字的字体大小
        - save_path: str, 保存图片的文件路径（包含文件名和扩展名，例如 'output/matrix_plot.png'）
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("输入的矩阵必须是二维的 numpy.ndarray")
    
    # 创建自定义颜色映射
    cmap = ListedColormap([color_0, color_1])
    
    # 绘制矩阵
    plt.figure(figsize=(matrix.shape[1] * 0.7, matrix.shape[0] * 0.7))
    plt.imshow(matrix, cmap=cmap, interpolation="none")

    # 添加网格线
    plt.gca().set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    plt.gca().grid(which="minor", color=edge_color, linestyle='-', linewidth=edge_width, zorder=0)
    
    # 添加外围边框（略微调整宽度以匹配视觉效果）
    rect = patches.Rectangle(
        (-0.5, -0.5),  # 左下角坐标
        matrix.shape[1],  # 矩阵宽度
        matrix.shape[0],  # 矩阵高度
        linewidth=edge_width * 1.5,  # 稍微调整，使视觉上匹配
        edgecolor=edge_color,
        facecolor='none',
        zorder=1  # 确保在网格线上方绘制
    )
    plt.gca().add_patch(rect)

    # 设置坐标轴
    plt.xticks(ticks=np.arange(matrix.shape[1]), labels=np.arange(matrix.shape[1]), fontsize=font_size)
    plt.yticks(ticks=np.arange(matrix.shape[0]), labels=np.arange(matrix.shape[0]), fontsize=font_size)

    # 将 x 轴坐标放到上方
    plt.tick_params(axis='x', labeltop=True, labelbottom=False)
    
    # 移除外部边框
    plt.tick_params(which="minor", size=0)
    plt.tick_params(axis='both', which='major', length=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    # 调整画布比例
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        folder = os.path.dirname(save_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)  # 创建文件夹
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
def process_image(model, color_img):
    # img = Image.open(imgfile).convert('RGB')
    image = color_img
    a = trans.ToTensor()
    sample = a(image).unsqueeze(0).to(torch.device('cuda'))
    sample = F.normalize(sample, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    orig_target_sizes=torch.tensor([[sample.shape[2],sample.shape[3]]],device='cuda:0')
    outputs = model(sample)
    postprocessors = PostProcess()
    results,adjs = postprocessors(outputs, orig_target_sizes)
    # res = {int(imgfile[-9:-4]): results[0]}
    res = {0: results[0]}
    results,results_grasp = prepare_for_coco_detection(res)
    dt = []
    dt_grasp=[]
    dt_adj=[output for output in adjs]
    grasp=[]
    for i in range(len(results)):
        if results[i]["score"] > 0.4:
            dt.append(results[i])
    for j in range(len(results_grasp)):
        if results_grasp[j]["score_grasp"] > 0.15:
            dt_grasp.append(results_grasp[j])
    best_dt_grasp=find_best_grasp(dt_grasp,dt)
    matrix = dt_adj[0][1].cpu().numpy()
    binary_matrix = (matrix > 0.5).astype(int)
    Grasp_Order=grasp_order(binary_matrix)
    if best_dt_grasp and Grasp_Order:
        grasp.append(best_dt_grasp[Grasp_Order[0]]['bbox_grasp'])
    else:
        grasp=[]
    # image= cv2.imread(imgfile)
    for bbox in grasp:
        # bbox = item['bbox_grasp']
        x, y, w, h, angle = bbox  # 解包为旋转矩形的参数
        draw_rotated_rectangle(image, x=x, y=y, w=w, h=h, angle=angle)
    i=0
    for item in dt:       
        bbox = item['bbox']
        lable_id=item['category_id']
        xmin, ymin, w, h= bbox  # 解包为旋转矩形的参数
        draw_rectangle(image, xmin=xmin, ymin=ymin, w=w, h=h, idx=i, lable_id=lable_id)
        i+=1
    return image,grasp,binary_matrix


    # img_name = os.path.basename(imgfile)
    # output_path = os.path.join(output_folder1, img_name)
    # cv2.imwrite(output_path, image)
    # print(f"Processed image saved to: {output_path}")

    # # 示例矩阵 (你可以直接使用从 PyTorch Tensor 转换得到的 numpy 数组)


    # draw_adj(binary_matrix,img_name,output_folder2)

def grasp2act(grasp,depth_img):
    destination=[]
    x,y,w,h,a=grasp[0]
    Z = depth_img[int(y)][int(x)] * robot.cam_depth_scale
    X = np.multiply(x-robot.cam_intrinsics[0][2],Z/robot.cam_intrinsics[0][0])
    Y = np.multiply(y-robot.cam_intrinsics[1][2],Z/robot.cam_intrinsics[1][1])
    # if Z == 0:
    #     return
    correction_Z = extract_max_depth_in_rotated_rect(depth_img, grasp)
    # X0 = np.multiply(0-robot.cam_intrinsics[0][2],correction_Z /robot.cam_intrinsics[0][0])
    # Y0 = np.multiply(0-robot.cam_intrinsics[1][2],correction_Z /robot.cam_intrinsics[1][1])  
    W = np.multiply(w,correction_Z /robot.cam_intrinsics[0][0])

    grasp_point = np.asarray([X,Y,correction_Z])
    grasp_point.shape = (3,1)
    grasp_home = [0, -0.3, 0.3, -np.pi, 0, np.pi]  # you can change me
    tool_orientation = [-np.pi, 0, np.pi]
    R_BE = robot.rpy2R(grasp_home[3:])
    # R_BE = robot.rpy2R([-np.pi, 0, np.pi])
    t_BE = np.asarray(grasp_home[:3]).reshape(3,1)*1000
    # t_BE = np.asarray([0, -0.3, 0.4]).reshape(3,1)
    T_BE = np.concatenate((np.concatenate((R_BE, t_BE), axis=1),np.array([[0, 0, 0, 1]])), axis=0)
    # T_EB = np.linalg.inv(T_BE)
    # bias = np.zeros_like(T_BE)
    # bias[0:3,3:]= np.array([[0],[0],[-155]])
    T_EC = robot.cam_pose 
    camera2robot = np.dot(T_BE,T_EC)
    target_position = np.dot(camera2robot[0:3,0:3],grasp_point) + camera2robot[0:3,3:] 
    # target_position = target_position[0:3,0]/1000+[0,0,0.16]
    target_position = target_position[0:3,0]/1000
    if target_position[2] < 0.01:
        target_position[2] = 0.01
    else:
        pass
    # angle_deflection=[0,0,a/180*np.pi]
    tool_orientation[2]-=a/180*np.pi
    b=np.append(grasp_home[:3],tool_orientation)
    robot.move_j_p(b)
    destination=np.append(target_position,tool_orientation)
    # destination[0]-=0.02
    # open_size=W/100+0.2
    open_size=0.6
    open_pos = int(-259*open_size +225) 
    robot.gripper.move_and_wait_for_pos(open_pos, 120, 100)
    robot.move_j_p(destination)
    put_depth=destination[2]
    time.sleep(1.2)
    robot.gripper.move_and_wait_for_pos(225, 120, 20)
    time.sleep(1.2)
    destination[2]+=0.1
    robot.move_j_p(destination)
    destination[:3]=[-0.35, -0.1, 0.15]
    # box_position = [0.25,0,0.25,-np.pi,0,np.pi]
    robot.move_j_p(destination)
    # box_position[2]=put_depth
    # robot.move_j_p(box_position)
    robot.gripper.move_and_wait_for_pos(0, 120, 10)
    time.sleep(1.2)
    robot.move_j_p(grasp_home)

def grasp_action():
    device = torch.device('cuda')
    main_args = get_main_args_parser().parse_args()
    
    # Load model
    model = load_model('/home/pmh/Code/dab_deformable_resnet_5d_Rotate_Point_QK_nobias/checkpoint0110.pth', main_args)
    model.eval()
    output_folder="/home/pmh/Code/dab_deformable_resnet_5d_Rotate_Point_QK_nobias/robot/result"

    # User options (change me)
    # --------------- Setup options ---------------

    grasp_home = [0, -0.3, 0.3, -np.pi, 0, np.pi]  # you can change me
    robot.move_j_p(grasp_home)
    time.sleep(10)
    # robot.open_gripper()

    # Slow down robot
    robot.joint_acc = 1.1
    robot.joint_vel = 0.5
    i=0
    while True:
        # camera_color_img, camera_depth_img,colorizer_depth= robot.get_camera_frame()
        camera_color_img, camera_depth_img = robot.get_camera_data()
        image=np.asanyarray(camera_color_img)
        with torch.no_grad():
            color_image,grasp,binary_matrix = process_image(model,image)
        if grasp:
            img_name=f'{int(i)}.jpg'
            adj_name=f'adj{int(i)}.jpg'
            output_path=os.path.join(output_folder, img_name)
            adj_path=os.path.join(output_folder, adj_name)
            cv2.imwrite(output_path, color_image)
            # draw_adj(binary_matrix,adj_name,output_folder)
            visualize_matrix(binary_matrix,save_path=adj_path)
            grasp2act(grasp,camera_depth_img)
            # break
        else:
            pass
        i+=1
def save_video():
    video_path = f'/home/pmh/Code/dab_deformable_resnet_5d_Rotate_Point_QK_nobias/robot/video/{int(time.time())}.mp4'
    fps, w, h = 30, 640, 480
    mp4 = cv2.VideoWriter_fourcc(*'mp4v') # 视频格式
    wr  = cv2.VideoWriter(video_path, mp4, fps, (w, h), isColor=True)
    flag_V =0
    while True:
        # camera_color_img, camera_depth_img = robot.get_camera_data()
        color_img, depth_img = robot.get_camera_data()
        cv2.imshow('RealSense',color_img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s') :
            flag_V = 1
        if flag_V == 1:
            wr.write(color_img)                # 保存RGB图像帧
            print('...录制视频中...')
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            print('...录制结束/直接退出...')
            break
    wr.release()
    print(f'，若保存视频，则视频保存在：{video_path}')
    cv2.destroyAllWindows()
# import time 
if __name__ == "__main__":
    device = torch.device('cuda')
    main_args = get_main_args_parser().parse_args()
    
    # Load model
    model = load_model('/home/pmh/Code/dab_deformable_resnet_5d_Rotate_Point_QK_nobias/checkpoint0110.pth', main_args)
    model.eval()
    output_folder="/home/pmh/Code/dab_deformable_resnet_5d_Rotate_Point_QK_nobias/robot/result"

    # User options (change me)
    # --------------- Setup options ---------------
    tcp_host_ip = '192.168.56.11' # IP and port to robot arm as TCP client (UR5)
    tcp_port = 30003
    tool_orientation = [-np.pi, 0, np.pi]
    # ---------------------------------------------

    # # # # # Move robot to home pose
    robot = UR_Robot(tcp_host_ip,tcp_port,is_use_robotiq85=True,is_use_camera=True)
    # robot.open_gripper()

    # Slow down robot
    device = torch.device('cuda')
    main_args = get_main_args_parser().parse_args()

    # User options (change me)
    # --------------- Setup options ---------------

    grasp_home = [0, -0.3, 0.3, -np.pi, 0, np.pi]  # you can change me
    robot.move_j_p(grasp_home)
    time.sleep(1)
    # robot.open_gripper()

    # Slow down robot
    robot.joint_acc = 1.1
    robot.joint_vel = 0.5
    i=0
    while True:
        # camera_color_img, camera_depth_img,colorizer_depth= robot.get_camera_frame()
        camera_color_img, camera_depth_img = robot.get_camera_data()
        image=np.asanyarray(camera_color_img)
        with torch.no_grad():
            color_image,grasp,binary_matrix = process_image(model,image)
        if grasp:
            img_name=f'{int(i)}.jpg'
            adj_name=f'adj{int(i)}.jpg'
            output_path=os.path.join(output_folder, img_name)
            adj_path=os.path.join(output_folder, adj_name)
            cv2.imwrite(output_path, color_image)
            # draw_adj(binary_matrix,adj_name,output_folder)
            visualize_matrix(binary_matrix,save_path=adj_path)
            grasp2act(grasp,camera_depth_img)
            # break
        else:
            # break
            pass
        i+=1
    # tcp_host_ip = '192.168.56.11' # IP and port to robot arm as TCP client (UR5)
    # tcp_port = 30003
    # tool_orientation = [-np.pi, 0, np.pi]
    # robot = UR_Robot(tcp_host_ip,tcp_port,is_use_robotiq85=True,is_use_camera=True)
    # thread1 = Thread(target=grasp_action)  # 线程1：执行任务打印4个a
    # # thread2 = Thread(target=save_video)  # 线程2：执行任务打印2个b
    
    # thread1.start()  # 线程1开始
    # # thread2.start()  # 线程2开始
    
    # thread1.join()  # 等待线程1结束
    # thread2.join()  # 等待线程2结束


    # video_path = f'/home/pmh/Code/dab_deformable_resnet_5d_Rotate_Point_QK_nobias/robot/{int(time.time())}.mp4'
    # fps, w, h = 30, 640, 480
    # mp4 = cv2.VideoWriter_fourcc(*'mp4v') # 视频格式
    # wr  = cv2.VideoWriter(video_path, mp4, fps, (w, h), isColor=True)
    # flag_V =0
    # while True:
    #     # camera_color_img, camera_depth_img = robot.get_camera_data()
    #     color_img, depth_img = robot.get_camera_data()
    #     cv2.imshow('RealSense',color_img)
    #     key = cv2.waitKey(1)
    #     if key & 0xFF == ord('s') :
    #         flag_V = 1
    #     if flag_V == 1:
    #         wr.write(color_img)                # 保存RGB图像帧
    #         print('...录制视频中...')
    #     if key & 0xFF == ord('q') or key == 27:
    #         cv2.destroyAllWindows()
    #         print('...录制结束/直接退出...')
    #         break
    # wr.release()
    # print(f'，若保存视频，则视频保存在：{video_path}')
    # cv2.destroyAllWindows()
        # image=np.asanyarray(camera_color_img)
        # with torch.no_grad():
        #     color_image,grasp= process_image(model,image)
        #     img_name=str(0)
        #     output_path=os.path.join(output_folder, img_name)
        #     cv2.imwrite(output_path, color_image)
        # if grasp:
        #     grasp2act(grasp,camera_depth_img)
        # else:
        #     break
        

    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # # Start streaming
    # pipeline.start(config)
    # try:
    #     while True:
    #         # Wait for a coherent pair of frames: depth and color
    #         frames = pipeline.wait_for_frames()
    #         start=time.time()
    #         depth_frame = frames.get_depth_frame()
    #         color_frame = frames.get_color_frame()
    #         if not depth_frame or not color_frame:
    #             continue
    #         # Convert images to numpy arrays
 
    #         depth_image = np.asanyarray(depth_frame.get_data())

    #         raw_color_image = np.asanyarray(color_frame.get_data())
    #         with torch.no_grad():
    #             color_image,_ = process_image(model,raw_color_image)
    #         # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    #         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    #         # Stack both images horizontally
    #         images = np.hstack((color_image, depth_colormap))
    #         # Show images
    #         cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    #         end = time.time()
    #         fps=1/(end-start)        
    #         cv2.putText(images, "FPS {0}".format(float('%.1f' % (fps))), (500, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),3)
    #         cv2.imshow('RealSense', images)
    #         key = cv2.waitKey(1)
    #         # Press esc or 'q' to close the image window
    #         if key & 0xFF == ord('q') or key == 27:
    #             cv2.destroyAllWindows()
    #             break
    # finally:
    #     # Stop streaming
    #     pipeline.stop()
