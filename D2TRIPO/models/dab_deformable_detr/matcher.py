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

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from oriented_iou_loss import cal_eiou

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
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
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, 
                 cost_grasp_class: float = 1, 
                 cost_grasp_angle: float = 1, 
                 cost_grasp_box: float = 1, 
                 focal_alpha = 0.25,
                 cost_grasp_iou = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_grasp_class = cost_grasp_class
        self.cost_grasp_angle = cost_grasp_angle
        self.cost_grasp_box = cost_grasp_box 
        self.cost_grasp_iou = cost_grasp_iou
        assert cost_grasp_class != 0 or self.cost_grasp != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits_grasp"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob_grasp = outputs["pred_logits_grasp"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox_grasp = outputs["pred_boxes_grasp"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_angle_grasp = outputs["pred_angles_grasp"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, 1]


        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["grasp_classes"] for v in targets])
        tgt_coord = torch.cat([v["grasp_points"] for v in targets])
        # tgt_angle = torch.cat([v["grasp_angles"] for v in targets]).unsqueeze(1)
        tgt_width = torch.cat([v["grasp_widths"] for v in targets]).unsqueeze(1)
        tgt_height = torch.cat([v["grasp_heights"] for v in targets]).unsqueeze(1)
        tgt_angle_cls = torch.cat([v["grasp_angles_cls"] for v in targets])
        tgt_angle= torch.cat([v["grasp_angles"] for v in targets])
        # 计算矩形的中心点坐标
        tgt_center = tgt_coord.mean(dim = 1)  # [44, 2]

        tgt_xywh = torch.cat((tgt_center, tgt_width, tgt_height), dim=1)  # 根据数据确认tgt_width和tgt_height是否需要进行不同的处理

        # cost_grasp_angle = generalized_grasp_box_cost(out_bbox_grasp, tgt_xywh, out_angle_grasp, tgt_angle)


        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob_grasp ** gamma) * (-(1 - out_prob_grasp + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob_grasp) ** gamma) * (-(out_prob_grasp + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the angle cost
        alpha = self.focal_alpha
        gamma = 2.0   
        neg_cost_class_angle = (1 - alpha) * (out_angle_grasp ** gamma) * (-(1 - out_angle_grasp + 1e-8).log())
        pos_cost_class_angle = alpha * ((1 - out_angle_grasp) ** gamma) * (-(out_angle_grasp + 1e-8).log())
        cost_grasp_angle = pos_cost_class_angle[:, tgt_angle_cls] - neg_cost_class_angle[:, tgt_angle_cls]


        max_indices = torch.argmax(out_angle_grasp, dim=1)
        real_angle = (max_indices.unsqueeze(1) - 8) * 10 -5
        # bg_indices = (max_indices == 18).nonzero(as_tuple=True)[0]
        # real_tgt_angle_cls = (tgt_angle_cls.unsqueeze(1) - 8) * 10

        out_dim=out_bbox_grasp.shape[0]
        tgt_dim=tgt_xywh.shape[0]
        out_grasp=torch.cat((out_bbox_grasp*10000, real_angle), dim=1).unsqueeze(1).expand(out_dim,tgt_dim,5)
        tgt_grasp=torch.cat((tgt_xywh*10000, tgt_angle.unsqueeze(1)), dim=1).unsqueeze(0).expand(out_dim,tgt_dim,5)
        # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(out_bbox_grasp, tgt_xywh, p=1)
        eiou_loss=cal_eiou(out_grasp,tgt_grasp)
        # ious=torch.tensor(rbbx_overlaps(out_grasp,tgt_grasp)).to(out_prob_grasp.device)
        # ious[bg_indices, :] = 0
        # print('______________________________________________________________________________________________________________________________')
        # print("IOUS Max:", ious.max())
        # print("IOUS Min:", ious.min())
        # print('______________________________________________________________________________________________________________________________')
        # Final cost matrix
        # C = self.cost_grasp_box * cost_bbox + self.cost_grasp_class * cost_class + self.cost_grasp_angle * cost_grasp_angle + self.cost_grasp_iou * eiou_loss 
        C = self.cost_grasp_class * cost_class + self.cost_grasp_angle * cost_grasp_angle + self.cost_grasp_iou * eiou_loss 
        # C = self.cost_grasp_box * cost_bbox + self.cost_grasp_class * cost_class + self.cost_grasp_angle * cost_grasp_angle 
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
