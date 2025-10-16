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


def process_image(model, imgfile, output_folder):
    img = Image.open(imgfile).convert('RGB')
    a = trans.ToTensor()
    sample = a(img).unsqueeze(0).to(torch.device('cuda'))
    sample = F.normalize(sample, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    outputs = model(sample)
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    out_logits_grasp = outputs['pred_logits']
    out_angle_grasp=outputs['pred_angles']
    prob_angles_grasp=out_angle_grasp.sigmoid()
    angel_indice=torch.argmax(prob_angles_grasp, dim=2, keepdim=True)
    angle=(angel_indice-8)*10-5
    out_bbox_grasp=torch.cat((outputs['pred_boxes'],angle), dim=2)
    # out_bbox_grasp = torch.cat((outputs['pred_boxes_grasp'], outputs['pred_angles_grasp']), dim=2)

    prob = out_logits.sigmoid()
    prob_grasp = out_logits_grasp.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
    topk_values_grasp, topk_indexes_grasp = torch.topk(prob_grasp.view(out_logits.shape[0], -1), 100, dim=1)
    scores = topk_values
    scores_grasp = topk_values_grasp
    topk_boxes = topk_indexes // out_logits.shape[2]
    topk_boxes_grasp = topk_indexes_grasp // out_logits_grasp.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    labels_grasp = topk_indexes_grasp % out_logits_grasp.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
    boxes_grasp = torch.gather(out_bbox_grasp, 1, topk_boxes_grasp.unsqueeze(-1).repeat(1, 1, 5))

    boxes_grasp_xy = torch.tensor(get_all_rotated_boxes_vertices(boxes_grasp)).to(device).view(1, 100, 8)
    img_h, img_w = torch.tensor(sample.shape[2]), torch.tensor(sample.shape[3])
    scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).unsqueeze(0).to(device)
    boxes = boxes * scale_fct
    scale_fct_grasp = torch.tensor([img_w, img_h, img_w, img_h, img_w, img_h, img_w, img_h]).unsqueeze(0).unsqueeze(0).to(device)
    boxes_grasp_xy = boxes_grasp_xy * scale_fct_grasp
    results = [{'scores': s, 'labels': l, 'boxes': b,'scores_grasp': s_grasp, 'labels_grasp': l_grasp,'boxes_grasp':b_grasp} 
                   for s, l, b, s_grasp, l_grasp, b_grasp in zip(scores, labels, boxes, scores_grasp, labels_grasp, boxes_grasp_xy)]
    im = cv2.imread(imgfile)
    
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
        
        cv2.drawContours(im, [box_grasp], 0, (0, 0, 255), 2)
        scores=results[0]['scores_grasp'][i]
        labels=results[0]['labels_grasp'][i]
        scores='{:.2f}'.format(scores.tolist())
        text=str(scores)+' '+str(labels.tolist())
        cv2.putText(im, text, box_grasp[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Add index text
    
        # box = boxes[0][i]
        # a0 = float(box[0])
        # b0 = float(box[1])
        # a1 = float(box[2])
        # b1 = float(box[3])
        # newBox1 = [[a0, b0], [a0, b1], [a1, b1], [a1, b0]]
        # point1 = np.array(newBox1).astype(int)
        # bbox = np.int32(point1)
        # cv2.drawContours(im, [bbox], 0, (0, 255, 255), 2)

    # Save the image to the output folder
    img_name = os.path.basename(imgfile)
    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, im)
    print(f"Processed image saved to: {output_path}")
def process_tgt(filename,imgfile,output_folder):
    im = cv2.imread(imgfile)
    with open(filename, 'r') as f:
    # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。
    # 然后将每个元素中的不同信息提取出来
        lines = f.readlines()
        for line in lines:
            location = line.split(' ')
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
            box = np.int32(point)
            
            # Draw the box contour on the image
            cv2.drawContours(im, [box], 0, (0, 0, 255), 2)

            # Label each point with its index
            # for idx, (x, y) in enumerate(point):
            #     cv2.circle(im, (x, y), 5, (0, 255, 0), -1)  # Draw a circle at each point
            #     cv2.putText(im, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Add index text
    img_name = os.path.basename(imgfile)
    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, im)
    print(f"Processed image saved to: {output_path}")
if __name__ == "__main__":
    device = torch.device('cuda')
    main_args = get_main_args_parser().parse_args()
    
    # Load model
    model = load_model('/home/pmh/code/dab_pred_weight_deformable/results/checkpoint0048.pth', main_args)
    model.eval()

    # Define input and output folders
    input_folder = '/home/pmh/data/VMRD/data/images/val'
    output_folder1 = '/home/pmh/code/dab_pred_weight_deformable/val_outputimg'
    label_folder = '/home/pmh/data/VMRD/Grasps'
    output_folder2 = '/home/pmh/code/dab_pred_weight_deformable/val_tgtimg'
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)

    # Process each image in the input folder
    for imgfile in os.listdir(input_folder):
        img_path = os.path.join(input_folder, imgfile)
        name = imgfile.rsplit('.')[0]+'.txt'
        label_path=os.path.join(label_folder, name)
        if os.path.isfile(img_path) and imgfile.endswith(('.jpg', '.png', '.jpeg')):
            process_image(model, img_path, output_folder1)
            process_tgt(label_path,img_path,output_folder2)
