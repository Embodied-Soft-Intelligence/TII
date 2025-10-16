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
from models.dab_deformable_detr import build_dab_deformable_detr
from shapely.geometry import Polygon
from torch import nn
from realsenseD435 import Camera
import pyrealsense2 as rs
import numpy as np
import cv2
import time
# 绘制旋转矩形的函数
def draw_rotated_rectangle(image, x, y, w, h, angle, color=(0, 255, 0), thickness=2):
    # 创建旋转矩形
    rect = ((x, y), (w, h), angle)
    # 获取旋转矩形的四个顶点
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # 将顶点坐标转换为整数
    # 绘制旋转矩形
    cv2.polylines(image, [box], isClosed=True, color=color, thickness=thickness)

def draw_rectangle(image, xmin, ymin, w, h, idx, lable_id, color=(255, 0, 0), thickness=2):
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
    cv2.putText(image, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
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
        keep=topk_values>0.3
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
            print('No relationship detected')
            B=np.array(range(len(a)))
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
    return B
def extract_max_depth_in_rotated_rect(depth_image, rect):
    """
    提取深度图像中旋转矩形框所包含区域的最大深度值。

    :param depth_image: 深度图像，numpy 数组，形状为 (480, 640)。
    :param rect: 旋转矩形框，格式为 (x, y, w, h, alpha)，
                 其中 x, y 为中心坐标，w, h 为宽和高，alpha 为旋转角度（以度为单位）。
    :return: 矩形框内的最大深度值。
    """
    # 解构矩形框参数
    x, y, w, h, alpha = rect

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
    max_depth = np.max(masked_depth[np.nonzero(masked_depth)])

    return max_depth
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
        if results[i]["score"] > 0.3:
            dt.append(results[i])
    for j in range(len(results_grasp)):
        if results_grasp[j]["score_grasp"] > 0.2:
            dt_grasp.append(results_grasp[j])
    best_dt_grasp=find_best_grasp(dt_grasp,dt)
    matrix = dt_adj[0][1].cpu().numpy()
    binary_matrix = (matrix > 0.5).astype(int)
    Grasp_Order=grasp_order(binary_matrix)
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = f"{current_time}.jpg"
    output_folder2 = '/home/pmh/Code/dab_deformable_resnet_5d_Rotate_Point_QK_nobias/adj'
    
    # cv2.imwrite(output_path, image)
    # print(f"Processed image saved to: {output_path}")

    # 示例矩阵 (你可以直接使用从 PyTorch Tensor 转换得到的 numpy 数组)


    draw_adj(binary_matrix,img_name,output_folder2)



    if len(binary_matrix)==len(best_dt_grasp):
        for i in Grasp_Order:
            grasp.append(best_dt_grasp[i]['bbox_grasp'])
    # image= cv2.imread(imgfile)
    for item in best_dt_grasp:
        bbox = item['bbox_grasp']
        x, y, w, h, angle = bbox  # 解包为旋转矩形的参数
        draw_rotated_rectangle(image, x=x, y=y, w=w, h=h, angle=angle)
    i=0
    for item in dt:
        i+=1
        bbox = item['bbox']
        lable_id=item['category_id']
        xmin, ymin, w, h= bbox  # 解包为旋转矩形的参数
        draw_rectangle(image, xmin=xmin, ymin=ymin, w=w, h=h, idx=i, lable_id=lable_id)
    return image,grasp


    # img_name = os.path.basename(imgfile)
    # output_path = os.path.join(output_folder1, img_name)
    # cv2.imwrite(output_path, image)
    # print(f"Processed image saved to: {output_path}")

    # # 示例矩阵 (你可以直接使用从 PyTorch Tensor 转换得到的 numpy 数组)


    # draw_adj(binary_matrix,img_name,output_folder2)


# import time 
if __name__ == "__main__":
    device = torch.device('cuda')
    main_args = get_main_args_parser().parse_args()
    
    # Load model
    model = load_model('/home/pmh/Code/dab_deformable_resnet_5d_Rotate_Point_QK_nobias/checkpoint0110.pth', main_args)
    model.eval()
    # title = "DT-DETR Inference"
    # camera=Camera()
    # cap=cv2.VideoCapture(0)
    # color_img, depth_img = camera.get_data()
    # with torch.no_grad():
    #     while cap.isOpened():
    #         # 从视频中读取一帧
    #         success, frame = cap.read()
    #         if success:
    #             # 在框架上运行 YOLOv8 推理
    #             image = process_image(frame)
                # 显示带标注的框架
     # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            start=time.time()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
 
            depth_image = np.asanyarray(depth_frame.get_data())

            raw_color_image = np.asanyarray(color_frame.get_data())
            with torch.no_grad():
                color_image,_ = process_image(model,raw_color_image)
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            end = time.time()
            fps=1/(end-start)        
            cv2.putText(images, "FPS {0}".format(float('%.1f' % (fps))), (500, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),3)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
