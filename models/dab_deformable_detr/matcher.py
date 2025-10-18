# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from oriented_iou_loss import cal_eiou

class HungarianMatcher(nn.Module):

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):

        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class HungarianMatcher_Grasp(nn.Module):

    def __init__(self, 
                 cost_grasp_class: float = 1, 
                 cost_grasp_angle: float = 1, 
                 cost_grasp_box: float = 1, 
                 focal_alpha = 0.25,
                 cost_grasp_iou = 1):

        super().__init__()
        self.cost_grasp_class = cost_grasp_class
        self.cost_grasp_angle = cost_grasp_angle
        self.cost_grasp_box = cost_grasp_box 
        self.cost_grasp_iou = cost_grasp_iou
        assert cost_grasp_class != 0 or self.cost_grasp != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):

        bs, num_queries = outputs["pred_logits_grasp"].shape[:2]

        out_prob_grasp = outputs["pred_logits_grasp"].flatten(0, 1).sigmoid() 
        out_bbox_grasp = outputs["pred_boxes_grasp"].flatten(0, 1) 
        out_angle_grasp = outputs["pred_angles_grasp"].flatten(0, 1).sigmoid() 

        tgt_ids = torch.cat([v["grasp_classes"] for v in targets])
        tgt_coord = torch.cat([v["grasp_points"] for v in targets])
        tgt_width = torch.cat([v["grasp_widths"] for v in targets]).unsqueeze(1)
        tgt_height = torch.cat([v["grasp_heights"] for v in targets]).unsqueeze(1)
        tgt_angle_cls = torch.cat([v["grasp_angles_cls"] for v in targets])
        tgt_angle= torch.cat([v["grasp_angles"] for v in targets])
        tgt_center = tgt_coord.mean(dim = 1)  # [44, 2]

        tgt_xywh = torch.cat((tgt_center, tgt_width, tgt_height), dim=1)  

        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob_grasp ** gamma) * (-(1 - out_prob_grasp + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob_grasp) ** gamma) * (-(out_prob_grasp + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        alpha = self.focal_alpha
        gamma = 2.0   
        neg_cost_class_angle = (1 - alpha) * (out_angle_grasp ** gamma) * (-(1 - out_angle_grasp + 1e-8).log())
        pos_cost_class_angle = alpha * ((1 - out_angle_grasp) ** gamma) * (-(out_angle_grasp + 1e-8).log())
        cost_grasp_angle = pos_cost_class_angle[:, tgt_angle_cls] - neg_cost_class_angle[:, tgt_angle_cls]


        max_indices = torch.argmax(out_angle_grasp, dim=1)
        real_angle = (max_indices.unsqueeze(1) - 8) * 10 -5

        out_dim=out_bbox_grasp.shape[0]
        tgt_dim=tgt_xywh.shape[0]
        out_grasp=torch.cat((out_bbox_grasp*10000, real_angle), dim=1).unsqueeze(1).expand(out_dim,tgt_dim,5)
        tgt_grasp=torch.cat((tgt_xywh*10000, tgt_angle.unsqueeze(1)), dim=1).unsqueeze(0).expand(out_dim,tgt_dim,5)

        eiou_loss=cal_eiou(out_grasp,tgt_grasp)

        C = self.cost_grasp_class * cost_class + self.cost_grasp_angle * cost_grasp_angle + self.cost_grasp_iou * eiou_loss 

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["grasp_classes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou),  \
            HungarianMatcher_Grasp(
            cost_grasp_class = args.set_cost_grasp_class,
            cost_grasp_angle = args.set_cost_grasp_angle, 
            cost_grasp_box = args.set_cost_grasp_box,
            focal_alpha=args.focal_alpha,
            cost_grasp_iou = args.set_cost_grasp_iou
            )
