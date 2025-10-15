from main import get_args_parser as get_main_args_parser
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from datasets import build_dataset
from models.DAB_DETR import build_DABDETR
from models.dab_deformable_detr import build_dab_deformable_detr
from datasets.data_prefetcher import data_prefetcher
from util import box_ops
from torch import nn
import math 
from shapely.geometry import Polygon
from rbbox_overlaps import rbbx_overlaps # type: ignore
from oriented_iou_loss import cal_iou

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
        keep=topk_values>0.5
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

def load_model(model_path , args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] 当前使用{}做推断".format(device))
    model, _, _ =  build_dab_deformable_detr(args)
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

def compute_iou(box1, box2):
    # 计算IoU
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def convert_to_xywh1(boxes):
    cx, cy, w, h = boxes.unbind(1)
    return torch.stack((cx-w/2, cy-h/2, w, h), dim=1)

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

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
def prepare(targets):
    coco_results = []
    for gt in targets:
        if len(targets) == 0:
            continue
        
        original_id=gt['image_id']
        boxes = gt["boxes"]
        w,h=gt['orig_size'][1],gt['orig_size'][0]
        boxes=convert_to_xywh1(boxes)
        boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        labels = gt["labels"].tolist()
        coco_results.extend(
            [
                {
                    "image_id": int(original_id),
                    "category_id": labels[k],
                    "bbox": box,
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results
def prepare_grasp(targets):
    grasp_results = []
    for gt in targets:
        if len(targets) == 0:
            continue
        
        original_id=gt['image_id']
        boxes_cxcy=gt['grasp_points'].mean(dim=1)
        # boxes=torch.cat((boxes_cxcy,gt['grasp_widths'].unsqueeze(1), gt['grasp_heights'].unsqueeze(1), (gt['grasp_angles_cls'].unsqueeze(1)-8)*10), dim=1)
        boxes=torch.cat((boxes_cxcy,gt['grasp_widths'].unsqueeze(1), gt['grasp_heights'].unsqueeze(1), gt['grasp_angles'].unsqueeze(1)), dim=1)
        # w,h=gt['orig_size'][1],gt['orig_size'][0]
        # boxes=convert_to_xywh1(boxes)
        # boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        w,h=gt['orig_size'][1],gt['orig_size'][0]
        boxes = boxes * torch.tensor([w, h, w, h,1], device=boxes.device)
        labels = gt["grasp_classes"].tolist()
        grasp_results.extend(
            [
                {
                    "image_id": int(original_id),
                    "category_id_grasp": labels[k],
                    "bbox_grasp": box,
                }
                for k, box in enumerate(boxes)
            ]
        )
    return grasp_results

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

def calculate_angle_diff(angle1, angle2):
    """计算两个旋转矩形框的IoU"""
    # 获取两个旋转矩形的顶点坐标
    if abs(angle1-angle2)>90:
        angle_diff=180-abs(angle1-angle2)
    else:
        angle_diff=abs(angle1-angle2)
    return angle_diff
def calculate_adj(all_gt_adj, all_dt_adj):
    all_TP = 0
    all_FP = 0
    all_FN = 0
    all_TR_gt, all_TR_dt = 0, 0
    IA = 0
    Ii = [0, 0, 0, 0]
    IAi = [0, 0, 0, 0]
    
    for k in range(len(all_dt_adj)):
        gt_adj = {}
        dt_adj = {}
        TP, FP, FN, TR_gt, TR_dt = 0, 0, 0, 0, 0
        
        # Initialize ground truth adjacency matrix with flag
        for i in range(len(all_gt_adj[k][0])):
            for j in range(len(all_gt_adj[k][0])):
                gt_adj.update({(all_gt_adj[k][0][i], all_gt_adj[k][0][j]): {'value': all_gt_adj[k][1][i][j], 'flag': True}})
        
        # Initialize detected adjacency matrix
        for i in range(len(all_dt_adj[k][0])):
            for j in range(len(all_dt_adj[k][0])):
                dt_adj.update({(all_dt_adj[k][0][i], all_dt_adj[k][0][j]): all_dt_adj[k][1][i][j]})
        
        # Match detected adjacency with ground truth adjacency
        for key_dt in dt_adj.keys():
            if dt_adj[key_dt] > 0.5:
                TR_dt += 1  # Count the detected positive connections
                for key_gt in gt_adj.keys():
                # Match with ground truth
                    if key_dt == key_gt and dt_adj[key_dt] > 0.5 and gt_adj[key_gt]['value'] == 1 and gt_adj[key_gt]['flag']:
                        TP += 1  # True positive
                        gt_adj[key_gt]['flag'] = False  # Mark as matched
                        break
        # Count ground truth positive connections
        for key_gt in gt_adj.keys():
            if gt_adj[key_gt]['value'] == 1:
                TR_gt += 1  # Total ground truth positive connections
        
        # Calculate false positives and false negatives
        FP = TR_dt - TP  # False positives
        FN = TR_gt - TP  # False negatives
        
        # Check for perfect matches
        if FP == 0 and FN == 0:
            IA += 1
        
        # Accuracy per image size group
        for i in range(4):
            if len(all_gt_adj[k][0]) == i + 2:
                Ii[i] += 1
                if FP == 0 and FN == 0:
                    IAi[i] += 1

        # Accumulate results across all images
        all_TP += TP
        all_FP += FP
        all_FN += FN
        all_TR_dt += TR_dt
        all_TR_gt += TR_gt
    
    # Calculate precision, recall, and accuracy metrics
    precision = all_TP / (all_TR_dt + 1e-6)
    recall = all_TP / all_TR_gt
    Image_Accuracy = IA / len(all_dt_adj)
    Image_Accuracyi = []
    for i in range(4):
        Image_Accuracyi.append(IAi[i] / (Ii[i] + 1e-6))
    
    return precision, recall, Image_Accuracy, Image_Accuracyi

# def find_best_grasp(all_dt_grasp,all_gt):
#     a=[]
#     b=[]
#     for i in range(len(all_gt)):
#         for j in range(len(all_dt_grasp)):
#             if all_gt[i]['image_id'] == all_dt_grasp[j]['image_id'] and all_gt[i]['category_id'] == all_dt_grasp[j]['category_id_grasp']:
#                 a.append(all_dt_grasp[j])
#         a.sort(key=lambda x: x['score_grasp'], reverse=True)  
#         if not a==[]: 
#             b.append(a[0])
#         a=[]
#     return b

import numpy as np
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
            # if all_dt_grasp[i]['image_id']==all_dt[j]['image_id'] :
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
import os
def main():
    weights_path = '/root/autodl-tmp/results'
    txt_path=os.path.join(weights_path,'evaluate.txt')
    files = os.listdir(weights_path)

    for file in files:
        if not file[-3:] == 'pth':
            continue
        file_path = os.path.join(weights_path, file)
        main_args = get_main_args_parser().parse_args()
        # 加载模型
        model = load_model(files, main_args)
        model.eval()
        # coco_evaluator = CocoEvaluator(base_ds, iou_types)
        # 构建数据集和数据加载器
        dataset = build_dataset(image_set='val')
        data_loader = DataLoader(dataset, batch_size=1, collate_fn=utils.collate_fn)

        # 数据预取器
        prefetcher = data_prefetcher(data_loader, device='cuda', prefetch=True)
        samples, targets = prefetcher.next()

        all_dt = []  # 初始化用于存储所有 dt 的列表
        all_gt = []  # 初始化用于存储所有 gt 的列表
        all_dt_grasp = []  # 初始化用于存储所有 dt 的列表
        all_gt_grasp = []  # 初始化用于存储所有 gt 的列表   
        all_dt_adj=[]
        all_gt_adj=[]
        for _ in range(len(data_loader)):
            with torch.no_grad():
                # 预测
                outputs = model(samples)
                postprocessors = {'bbox': PostProcess()}
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results,adjs = postprocessors['bbox'](outputs, orig_target_sizes)
                res = {target['image_id'].item(): output for target, output in zip(targets, results)}
                results,results_grasp = prepare_for_coco_detection(res)
                gt = prepare(targets)
                gt_grasp=prepare_grasp(targets)
                gt_adj=[(output['labels'],output['adj']) for output in targets]
                dt = []
                dt_grasp=[]
                dt_adj=[output for output in adjs]
                for i in range(len(results)):
                    if results[i]["score"] > 0.3:
                        dt.append(results[i])
                for j in range(len(results_grasp)):
                    if results_grasp[j]["score_grasp"] > 0.2:
                        dt_grasp.append(results_grasp[j])
                # 收集当前批次的 dt 和 gt
                all_dt.extend(dt)
                all_dt_grasp.extend(dt_grasp)
                all_dt_adj.extend(dt_adj)
                all_gt.extend(gt)
                all_gt_grasp.extend(gt_grasp)
                all_gt_adj.extend(gt_adj)
                samples, targets = prefetcher.next()

        # 按照 score 对所有 dt 进行排序
        all_dt.sort(key=lambda x: x['score'], reverse=True)
        all_dt_grasp.sort(key=lambda x: x['score_grasp'], reverse=True)
        all_dt_grasp=find_best_grasp(all_dt_grasp,all_dt)
        OP,OR,IA,IAi=calculate_adj(all_gt_adj,all_dt_adj)
        print(f'OP: {OP}, OR: {OR}, IA: {IA}')
        print(f'IA2: {IAi[0]}, IA3: {IAi[1]}, IA4: {IAi[2]},IA5: {IAi[3]}')
        used_all_gt_indices = set()  # 用于跟踪已匹配的 gt 索引
        TP, FP, FN = 0, 0, 0
        for j in range(len(all_dt)):
            for i in range(len(all_gt)):
                # if i in used_all_gt_indices:
                #     continue  # 跳过已经匹配的 gt
                if all_gt[i]['image_id'] == all_dt[j]['image_id'] and all_gt[i]['category_id'] == all_dt[j]['category_id']:
                    iou = compute_iou(all_gt[i]['bbox'], all_dt[j]['bbox'])
                    if iou > 0.5:
                        # TP += 1
                        used_all_gt_indices.add(i)  # 标记这个 gt 为已匹配
                        break  # 找到匹配的 gt 后，跳出内层循环
                    else:
                        continue
        TP = len(used_all_gt_indices)
        FP = len(all_dt) - TP
        FN = len(all_gt) - TP
        # 打印整个数据加载器的总结果
        print(f'Total TP: {TP}, FP: {FP}, FN: {FN}')
    # 打印整个数据加载器的总结果
        precision, recall, f1_score=calculate_metrics(TP, FP, FN)
        print(f'precision_det: {precision}, recall_det: {recall}, f1_score: {f1_score}')


        used_all_gt_grasp_indices = set()  # 用于跟踪已匹配的 gt 索引

        TP1, FP1, FN1 = 0, 0, 0
        for j in range(len(all_dt_grasp)):
            for i in range(len(all_gt_grasp)):
                # if i in used_all_gt_grasp_indices:
                #      continue  # 跳过已经匹配的 gt
                # if all_gt_grasp[i]['image_id'] == all_dt_grasp[j]['image_id'] and all_gt_grasp[i]['category_id_grasp'] == all_dt_grasp[j]['category_id_grasp']:
                if all_gt_grasp[i]['image_id'] == all_dt_grasp[j]['image_id']:
                    # a=torch.tensor(all_gt_grasp[i]['bbox_grasp'],device='cuda:0').clone()
                    # b= torch.tensor(all_dt_grasp[j]['bbox_grasp'],device='cuda:0').clone()
                    # ious = cal_iou(a.view(1,1,-1), b.view(1,1,-1))
                    iou = calculate_iou(all_gt_grasp[i]['bbox_grasp'], all_dt_grasp[j]['bbox_grasp'])
                    # iou=ious[0].item()
                    angle_diff=calculate_angle_diff(all_gt_grasp[i]['bbox_grasp'][4], all_dt_grasp[j]['bbox_grasp'][4])
                    # if iou > 0.25 and abs(all_gt_grasp[i]['bbox_grasp'][4]-all_dt_grasp[j]['bbox_grasp'][4])<30:
                    if iou >0.25 and angle_diff<30:
                        TP1 += 1
                        # used_all_gt_grasp_indices.add(i)  # 标记这个 gt 为已匹配
                        break  # 找到匹配的 gt 后，跳出内层循环
                    else:
                        continue
        FP1 = len(all_dt_grasp) - TP1
        FN1 = len(all_gt) - TP1 
        # 打印整个数据加载器的总结果
        print(f'Total TP_grasp: {TP1}, FP_grasp: {FP1}, FN_grasp: {FN1}')
    # 打印整个数据加载器的总结果
        precision_grasp, recall_grasp, f1_score_grasp=calculate_metrics(TP1, FP1, FN1)
        print(f'precision_grasp: {precision_grasp}, recall_grasp: {recall_grasp}, f1_score_grasp: {f1_score_grasp}')
        with open(txt_path,'a') as f:
                f.write(file)
                f.write("\n")
                f.write(f'OP: {OP}, OR: {OR}, IA: {IA}, IA2: {IAi[0]}, IA3: {IAi[1]}, IA4: {IAi[2]},IA5: {IAi[3]}\n')
                f.write(f'precision_det: {precision}, recall_det: {recall}, f1_score: {f1_score}\n')
                f.write(f'precision_grasp: {precision_grasp}, recall_grasp: {recall_grasp}, f1_score_grasp: {f1_score_grasp}\n')                              
                f.write("\n")


if __name__ == "__main__":
    main()
