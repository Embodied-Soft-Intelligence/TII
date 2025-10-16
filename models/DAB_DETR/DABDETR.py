# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------




import os

import math
from typing import Dict
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
from .transformer import build_transformer


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    return loss.mean(1).sum() / num_boxes


class DABDETR(nn.Module):
    """ This is the DAB-DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_dec_layers,
                    aux_loss=False, 
                    iter_update=True,
                    query_dim=4, 
                    bbox_embed_diff_each_layer=False,
                    random_refpoints_xy=False,
                    ref_5d=False,
                    bbox_embed_diff_each_layer_grasp=False,
                    iter_adj=False
                    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            iter_update: iterative update of boxes
            query_dim: query dimension. 2 for point and 4 for box.
            bbox_embed_diff_each_layer: dont share weights of prediction heads. Default for False. (shared weights.)
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
            

        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.d_model = hidden_dim
        # self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.class_embed_grasp = nn.Linear(hidden_dim, num_classes)
        # self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        self.bbox_embed_diff_each_layer_grasp = bbox_embed_diff_each_layer_grasp

        # if bbox_embed_diff_each_layer:
        #     self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for i in range(num_dec_layers)])
        # else:
        #     self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # grasp and angles 回归头
        if bbox_embed_diff_each_layer_grasp:
            self.bbox_embed_grasp = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for i in range(num_dec_layers)])
            self.angle_embed_grasp = nn.ModuleList([MLP(hidden_dim, hidden_dim, 1, 3) for i in range(num_dec_layers)])
        else:
            self.bbox_embed_grasp = MLP(hidden_dim, hidden_dim, 4, 3)
            self.angle_embed_grasp = MLP(hidden_dim, hidden_dim, 1, 3)

        # adj 是否迭代更新
        # if iter_adj:
        #     self.adj_proj = nn.ModuleList([MLP(hidden_dim + 4, hidden_dim + 4, hidden_dim, 3) for i in range(num_dec_layers)])
        #     self.adj_iter_proj = nn.ModuleList([MLP(num_queries, num_queries, hidden_dim, 3) for i in range(num_dec_layers)])
        # else:
        #     self.adj_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # setting query dim
        self.query_dim = query_dim
        assert query_dim in [2, 4]

        # self.refpoint_embed = nn.Embedding(num_queries, query_dim)
        # self.random_refpoints_xy = random_refpoints_xy
        # if random_refpoints_xy:
        #     # import ipdb; ipdb.set_trace()
        #     self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
        #     self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
        #     self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if ref_5d:
            # grasp embedding 5D
            self.refpoint_embed_grasp = nn.Embedding(num_queries, 5)
        else:
            self.refpoint_embed_grasp = nn.Embedding(num_queries, query_dim)

        if random_refpoints_xy:
            self.refpoint_embed_grasp.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed_grasp.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed_grasp.weight.data[:, :2])
            self.refpoint_embed_grasp.weight.data[:, :2].requires_grad = False

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.iter_update = iter_update
        self.iter_adj = iter_adj

        # if self.iter_update:
        #     self.transformer.decoder.bbox_embed = self.bbox_embed

        # init prior_prob setting for focal loss
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init grasp_class_embed
        prior_prob_grasp = 0.01
        bias_value_grasp = -math.log((1 - prior_prob_grasp) / prior_prob_grasp)
        self.class_embed_grasp.bias.data = torch.ones(num_classes) * bias_value_grasp

        # import ipdb; ipdb.set_trace()
        # init bbox_embed
        # if bbox_embed_diff_each_layer:
        #     for bbox_embed in self.bbox_embed:
        #         nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        #         nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        # else:
        #     nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        #     nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if bbox_embed_diff_each_layer_grasp:
            for bbox_embed_grasp in self.bbox_embed_grasp:
                nn.init.constant_(bbox_embed_grasp.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed_grasp.layers[-1].bias.data, 0) 
            # for angle_embed_grasp in self.angle_embed_grasp:
            #     nn.init.constant_(angle_embed_grasp.layers[-1].weight.data, 0)
            #     nn.init.constant_(angle_embed_grasp.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed_grasp.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed_grasp.layers[-1].bias.data, 0)

            # nn.init.constant_(self.angle_embed_grasp.layers[-1].weight.data, 0)
            # nn.init.constant_(self.angle_embed_grasp.layers[-1].bias.data, 0)
        
        # if iter_adj:
        #     for adj_proj in self.adj_proj:
        #         nn.init.constant_(adj_proj.layers[-1].weight.data, 0)
        #         nn.init.constant_(adj_proj.layers[-1].bias.data, 0)
        #     for adj_iter_proj in self.adj_iter_proj:
        #         nn.init.constant_(adj_iter_proj.layers[-1].weight.data, 0)
        #         nn.init.constant_(adj_iter_proj.layers[-1].bias.data, 0)
        # else:
        #     nn.init.constant_(self.adj_proj.layers[-1].weight.data, 0)
        #     nn.init.constant_(self.adj_proj.layers[-1].bias.data, 0)


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        # default pipeline
        # embedweight = self.refpoint_embed.weight
        embedweight = None
        embedweight_grasp = self.refpoint_embed_grasp.weight

        # hs, reference, hs_grasp, reference_grasp, memory = self.transformer(self.input_proj(src), mask, embedweight, pos[-1], embedweight_grasp)
        hs_grasp, reference_grasp= self.transformer(self.input_proj(src), mask, embedweight, pos[-1], embedweight_grasp)
        
        # if not self.bbox_embed_diff_each_layer:
        #     reference_before_sigmoid = inverse_sigmoid(reference)
        #     tmp = self.bbox_embed(hs)
        #     tmp[..., :self.query_dim] += reference_before_sigmoid
        #     outputs_coord = tmp.sigmoid()
        # else:
        #     reference_before_sigmoid = inverse_sigmoid(reference)
        #     outputs_coords = []
        #     for lvl in range(hs.shape[0]):
        #         tmp = self.bbox_embed[lvl](hs[lvl])
        #         tmp[..., :self.query_dim] += reference_before_sigmoid[lvl]
        #         outputs_coord = tmp.sigmoid()
        #         outputs_coords.append(outputs_coord)
        #     outputs_coord = torch.stack(outputs_coords)

        if not self.bbox_embed_diff_each_layer_grasp:
            reference_grasp_before_sigmoid = inverse_sigmoid(reference_grasp)
            tmp_grasp = self.bbox_embed_grasp(hs_grasp)
            tmp_grasp[..., :self.query_dim] += reference_grasp_before_sigmoid
            outputs_grasp_coord = tmp_grasp.sigmoid()
            outputs_grasp_angle = self.angle_embed_grasp(hs_grasp)
        else:
            reference_grasp_before_sigmoid = inverse_sigmoid(reference_grasp)
            outputs_grasp_coords = []
            outputs_grasp_angles = []
            for lvl in range(hs_grasp.shape[0]):
                tmp_grasp = self.bbox_embed_grasp[lvl](hs_grasp[lvl])
                tmp_grasp[..., :self.query_dim] += reference_grasp_before_sigmoid[lvl]
                outputs_grasp_coord = tmp_grasp.sigmoid()
                outputs_grasp_coords.append(outputs_grasp_coord)
                outputs_grasp_angle = self.angle_embed_grasp(hs_grasp[lvl])
                outputs_grasp_angles.append(outputs_grasp_angle)

            outputs_grasp_angle = torch.stack(outputs_grasp_angles)
            outputs_grasp_coord = torch.stack(outputs_grasp_coords)

        # [300, 2, 256] -> [2, 300, 256]
        # memory = memory.permute(1, 0, 2)
        # outputs_adjs = []
        # adj 是否迭代更新
        # if self.iter_adj:
        #     for lvl in range(hs_grasp.shape[0]):
        #         if lvl == 0:
        #             adj_cat = torch.cat([memory, outputs_coord[lvl]], dim=2)
        #             adj = self.adj_proj[lvl](adj_cat)
        #             outputs_adj = torch.matmul(adj, adj.transpose(1, 2))
        #             outputs_adj = outputs_adj / self.d_model
        #         else:
        #             outputs_adj = self.adj_iter_proj[lvl](outputs_adj)
        #             adj_cat = torch.cat([outputs_adj, outputs_coord[lvl]], dim=2)
        #             adj = self.adj_proj[lvl](adj_cat)
        #             outputs_adj = torch.matmul(adj, adj.transpose(1, 2))
        #             outputs_adj = outputs_adj / self.d_model
        #         outputs_adjs.append(outputs_adj)
        #     outputs_adj = torch.stack(outputs_adjs) 
        # else:
        #     # [2, 300, 256] -> [300, 2, 256]

        #     adj = self.adj_proj(memory)
        #     last_outputs_coord = outputs_coord[-1]
        #     adj_cat = torch.cat([adj, last_outputs_coord], dim=2)
        #     outputs_adj = torch.matmul(adj_cat, adj_cat.transpose(1, 2))
        #     outputs_adj = outputs_adj / self.d_model
            
        # outputs_class = self.class_embed(hs)
        outputs_class_grasp = self.class_embed_grasp(hs_grasp)

        if self.iter_adj:
            out = { 
                    # 'pred_logits': outputs_class[-1], 
                    # 'pred_boxes': outputs_coord[-1],
                    'pred_logits_grasp': outputs_class_grasp[-1],
                    'pred_boxes_grasp': outputs_grasp_coord[-1],
                    'pred_angles_grasp': outputs_grasp_angle[-1],
                    # 'pred_adj': outputs_adj[-1]
                }
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(
                                                        # outputs_class, 
                                                        # outputs_coord,
                                                        outputs_class_grasp,
                                                        outputs_grasp_coord,
                                                        outputs_grasp_angle,
                                                        # outputs_adj
                                                        )
        else:
            out = { 
                    # 'pred_logits': outputs_class[-1], 
                    # 'pred_boxes': outputs_coord[-1],
                    'pred_logits_grasp': outputs_class_grasp[-1],
                    'pred_boxes_grasp': outputs_grasp_coord[-1],
                    'pred_angles_grasp': outputs_grasp_angle[-1],
                    # 'pred_adj': outputs_adj
                }
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(
                                                        # outputs_class, 
                                                        # outputs_coord,
                                                        outputs_class_grasp,
                                                        outputs_grasp_coord,
                                                        outputs_grasp_angle
                                                        )
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class_grasp, outputs_grasp_coord, outputs_grasp_angle):

        return [{   
                    'pred_logits_grasp': a,
                    'pred_boxes_grasp': b,
                    'pred_angles_grasp': c,

                }
                for a, b, c in zip(
                                    outputs_class_grasp[:-1],
                                    outputs_grasp_coord[:-1],
                                    outputs_grasp_angle[:-1],
                                 )]


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, weight_dict, focal_alpha, losses, matcher_grasp = None, losses_aux = None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses
        self.losses_aux = losses_aux
        self.focal_alpha = focal_alpha
        self.matcher_grasp = matcher_grasp

    def get_rotated_vertices(self, center, width, height, angle):
        theta = torch.deg2rad(angle)
        b = torch.cos(theta) * 0.5
        a = torch.sin(theta) * 0.5
        return torch.stack([
            center[:, 0] - a * height - b * width, center[:, 1] + b * height - a * width,
            center[:, 0] + a * height - b * width, center[:, 1] - b * height - a * width,
            center[:, 0] + a * height + b * width, center[:, 1] - b * height + a * width,
            center[:, 0] - a * height + b * width, center[:, 1] + b * height + a * width
        ], dim=-1).view(-1, 4, 2)
    
    def rectangle(self, boxes1, boxes2):
        # 计算左上角最小值
        lt = torch.min(boxes1, boxes2)  # 形状为 (N, 4, 2)
        lt = lt.min(dim=1).values  # 沿第1维度（4个顶点）取最小值，形状为 (N, 2)

        # 计算右下角最大值
        rb = torch.max(boxes1, boxes2)  # 形状为 (N, 4, 2)
        rb = rb.max(dim=1).values  # 沿第1维度（4个顶点）取最大值，形状为 (N, 2)

        wh = (rb - lt).clamp(min=0)  # [N, 2]
        c = wh[:, 0] ** 2 + wh[:, 1] ** 2
        return wh, c


    
    def loss_grasp_class(self, outputs, targets, num_grasp_boxes, indices_grasp):

        assert 'pred_logits_grasp' in outputs

        src_logits = outputs['pred_logits_grasp']

        idx = self._get_src_permutation_idx(indices_grasp)
        target_classes_o = torch.cat([t["grasp_classes"][J] for t, (_, J) in zip(targets, indices_grasp)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_grasp_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce_grasp': loss_ce}

        return losses

    def loss_grasp_bbox(self, outputs, targets, num_grasp_boxes, indices_grasp):

        assert 'pred_boxes_grasp' in outputs

        idx = self._get_src_permutation_idx(indices_grasp)
        src_boxes = outputs['pred_boxes_grasp'][idx]
        # tgt_center = tgt_coord.view(-1, 4, 2).mean(dim=1)  # [N, 2]
        target_boxes = torch.cat([t['grasp_points'][i].view(-1, 4, 2).mean(dim=1) for t, (_, i) in zip(targets, indices_grasp)], dim=0)

        # # 获取 grasp_points 的 wh 部分
        # src_width = outputs['pred_widths_grasp'][idx]
        # src_height = outputs['pred_heights_grasp'][idx]

        target_w = torch.cat([t['grasp_widths'][i].unsqueeze(-1) for t, (_, i) in zip(targets, indices_grasp)], dim=0)
        target_h = torch.cat([t['grasp_heights'][i].unsqueeze(-1) for t, (_, i) in zip(targets, indices_grasp)], dim=0)

        # 将 xy 和 wh 拼接在一起
        # src_combined = torch.cat((src_boxes, src_width, src_height), dim=1)
        target_combined = torch.cat((target_boxes, target_w, target_h), dim=1)

        loss_bbox = F.l1_loss(src_boxes, target_combined, reduction='none')

        losses = {}
        # losses['loss_bbox_grasp'] = loss_bbox.sum() / num_grasp_boxes

        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        #     box_ops.box_cxcywh_to_xyxy(src_boxes),
        #     box_ops.box_cxcywh_to_xyxy(target_boxes)))
        # losses['loss_giou'] = loss_giou.sum() / num_boxes

        # Extract grasp angles
        out_angle = outputs['pred_angles_grasp'][idx]  # Predicted angles
        tgt_angle = torch.cat([t['grasp_angles'][i].unsqueeze(-1) for t, (_, i) in zip(targets, indices_grasp)], dim=0)  # Target angles, reshaped to [32, 1]

        # 计算角度损失
        # angle_loss = F.l1_loss(out_angle, tgt_angle, reduction='none')
        # losses['loss_angle_grasp'] = angle_loss.sum() / num_grasp_boxes

        angle_diff = out_angle - tgt_angle

        # 修改大于90度和小于-90度的角度差异
        # angle_diff = torch.where(angle_diff > 90, angle_diff - 180, angle_diff)
        # angle_diff = torch.where(angle_diff < -90, angle_diff + 180, angle_diff)
        angle_diff = torch.deg2rad(angle_diff)

        src_vertices = self.get_rotated_vertices(src_boxes[:, :2], src_boxes[:, 2], src_boxes[:, 3], out_angle.squeeze())
        # out_vertices = self.get_rotated_vertices(target_combined[:, :2], target_combined[:, 2], target_combined[:, 3], tgt_angle.squeeze())

        tgt_vertices = torch.cat([t['grasp_points'][i] for t, (_, i) in zip(targets, indices_grasp)], dim=0)

        # 计算最小外接矩形
        wh, c = self.rectangle(src_vertices, tgt_vertices)

        src_width = src_boxes[:, 2].unsqueeze(-1)
        src_height = src_boxes[:, 3].unsqueeze(-1)

        diff_w = (src_width - target_w) ** 2
        diff_h = (src_height - target_h) ** 2

        ratio = diff_w.squeeze(-1) / wh[:, 0] ** 2 + diff_h.squeeze(-1) / wh[:, 1] ** 2

        d = src_boxes[:, :2] - target_boxes
        distance = (d ** 2).sum(dim = -1)

        alapha = 2 - distance / c
        gamma = 2

        loss = gamma * distance / c + alapha * (torch.sin(angle_diff.squeeze(-1))) ** 2 + ratio
        losses['loss_grasp'] = loss.sum() / num_grasp_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, num_grasp_boxes, indices_grasp, **kwargs):

        loss_map = {
            'grasp_class':self.loss_grasp_class,
            'grasp_bbox':self.loss_grasp_bbox          
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        # return loss_map[loss](outputs, targets, indices, num_boxes, 
        #                     num_grasp_boxes, indices_grasp, **kwargs)
        return loss_map[loss](outputs, targets, num_grasp_boxes, indices_grasp, **kwargs)



    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)
        indices_grasp = self.matcher_grasp(outputs_without_aux, targets)

        # if return_indices:
        #     indices0_copy = indices
        #     indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_boxes = sum(len(t["labels"]) for t in targets)
        # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        num_grasp_boxes = sum(len(t["grasp_classes"]) for t in targets)
        num_grasp_boxes = torch.as_tensor([num_grasp_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_grasp_boxes)
        num_grasp_boxes = torch.clamp(num_grasp_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, num_grasp_boxes, indices_grasp, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_grasp = self.matcher_grasp(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, num_grasp_boxes, indices_grasp, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits_grasp'], outputs['pred_boxes_grasp']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_DABDETR(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 32 if args.dataset_file != 'coco' else 32
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DABDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_dec_layers=args.dec_layers,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        ref_5d = args.ref_5d,
        bbox_embed_diff_each_layer_grasp=args.bbox_embed_diff_each_layer_grasp,
        iter_adj=args.iter_adj
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher_grasp = build_matcher(args)

    weight_dict = {
                #    'loss_ce': args.cls_loss_coef, 
                #    'loss_bbox': args.bbox_loss_coef, 
                #    'loss_adj':args.adj_loss_coef, 
                   'loss_grasp':args.grasp_loss_coef, 
                   'loss_ce_grasp':args.grasp_ce_loss_coef, 
                   }
    # weight_dict['loss_giou'] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            if args.iter_adj:
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            else:
                temp_weight_dict = {k: v for k, v in weight_dict.items() if k != 'loss_adj'}
                aux_weight_dict.update({k + f'_{i}': v for k, v in temp_weight_dict.items()})
    
        weight_dict.update(aux_weight_dict)

    losses = ['grasp_class', 'grasp_bbox']
    losses_aux = ['labels', 'boxes', 'cardinality', 'grasp_class', 'grasp_bbox']

    criterion = SetCriterion(num_classes, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses, 
                             matcher_grasp = matcher_grasp, losses_aux = losses_aux)
    criterion.to(device)

    postprocessors = {'bbox': PostProcess(num_select=args.num_select)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
