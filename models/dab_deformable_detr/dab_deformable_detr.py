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


import os
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy
from oriented_iou_loss import cal_eiou

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DABDeformableDETR(nn.Module):

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=True, two_stage=False,
                 use_dab=True, 
                 num_patterns=0,
                 random_refpoints_xy=False
                 ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.d_model = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.adj1_linear = nn.Linear(hidden_dim, hidden_dim)
        self.adj2_linear = nn.Linear(hidden_dim, hidden_dim)
        self.adj_embed = MLP(2*hidden_dim, hidden_dim, 1, 3)
        self.adj_query = nn.Linear(hidden_dim, hidden_dim)
        self.adj_key = nn.Linear(hidden_dim, hidden_dim)
        self.WG = nn.Linear(2*hidden_dim, 1)

        self.class_embed_grasp = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed_grasp = MLP(hidden_dim, hidden_dim, 4, 3)
        self.angle_embed_grasp = nn.Linear(hidden_dim, 19)

        self.num_feature_levels = num_feature_levels
        self.use_dab = use_dab
        self.num_patterns = num_patterns
        self.random_refpoints_xy = random_refpoints_xy
        if not two_stage:
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
            else:
                self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries, 4)
                self.tgt_embed_grasp = nn.Embedding(num_queries, hidden_dim)
                self.refpoint_embed_grasp = nn.Embedding(num_queries, 4)
                self.refangle_embed_grasp = nn.Embedding(num_queries, 19)

                if random_refpoints_xy:
                    self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
                    self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                    self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.num_patterns > 0:
            self.patterns_embed = nn.Embedding(self.num_patterns, hidden_dim)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed_grasp.bias.data = torch.ones(num_classes) * bias_value

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        nn.init.constant_(self.bbox_embed_grasp.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed_grasp.layers[-1].bias.data, 0)

        nn.init.constant_(self.angle_embed_grasp.weight.data, 0)
        nn.init.constant_(self.angle_embed_grasp.bias.data, 0)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        num_pred_grasp = (transformer.decoder_grasp.num_layers + 1) if two_stage else transformer.decoder_grasp.num_layers

        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed

            self.adj1_linear= nn.ModuleList([self.adj1_linear for _ in range(num_pred)])
            self.adj2_linear = nn.ModuleList([self.adj2_linear for _ in range(num_pred)])
            self.adj_embed = _get_clones(self.adj_embed , num_pred)

            self.adj_query= nn.ModuleList([self.adj_query for _ in range(num_pred)])
            self.adj_key = nn.ModuleList([self.adj_key for _ in range(num_pred)])
            self.WG = _get_clones(self.WG, num_pred)

        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if with_box_refine:
            self.class_embed_grasp = _get_clones(self.class_embed_grasp, num_pred_grasp)
            self.bbox_embed_grasp = _get_clones(self.bbox_embed_grasp, num_pred_grasp)
            self.angle_embed_grasp = _get_clones(self.angle_embed_grasp, num_pred_grasp)
            nn.init.constant_(self.bbox_embed_grasp[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder_grasp.bbox_embed = self.bbox_embed_grasp
            self.transformer.decoder_grasp.angle_embed = self.angle_embed_grasp

        else:
            nn.init.constant_(self.bbox_embed_grasp.layers[-1].bias.data[2:], -2.0)
            self.class_embed_grasp = nn.ModuleList([self.class_embed_grasp for _ in range(num_pred_grasp)])
            self.bbox_embed_grasp = nn.ModuleList([self.bbox_embed_grasp for _ in range(num_pred_grasp)])
            self.angle_embed_grasp = nn.ModuleList([self.angle_embed_grasp for _ in range(num_pred_grasp)])
            self.transformer.decoder_grasp.bbox_embed = None

        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.two_stage:
            query_embeds = None
        elif self.use_dab:
            
            if self.num_patterns == 0:
                tgt_embed = self.tgt_embed.weight        
                refanchor = self.refpoint_embed.weight      
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)

                tgt_embed_grasp = self.tgt_embed_grasp.weight        
                refanchor_grasp = self.refpoint_embed_grasp.weight  
                refangle_grasp = self.refangle_embed_grasp.weight
                query_embeds_grasp = torch.cat((tgt_embed_grasp, refanchor_grasp, refangle_grasp), dim=1)
            
            else:
                tgt_embed = self.tgt_embed.weight          
                pat_embed = self.patterns_embed.weight     
                tgt_embed = tgt_embed.repeat(self.num_patterns, 1)
                pat_embed = pat_embed[:, None, :].repeat(1, self.num_queries, 1).flatten(0, 1) 
                tgt_all_embed = tgt_embed + pat_embed
                refanchor = self.refpoint_embed.weight.repeat(self.num_patterns, 1)      
                query_embeds = torch.cat((tgt_all_embed, refanchor), dim=1)
        else:
            query_embeds = self.query_embed.weight
        
        hs, init_reference, inter_references, \
        hs_grasp, init_reference_grasp, inter_references_grasp, \
        init_reference_out_angle_grasp, inter_references_out_angle_grasp,\
        q, k,\
        enc_outputs_class, enc_outputs_coord_unact = \
        self.transformer(srcs, masks, pos, query_embeds, query_embeds_grasp)


        outputs_classes = []
        outputs_coords = []
        outputs_adjs = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            decoder_outputs_adj_1 = self.adj1_linear[lvl](hs[lvl]).unsqueeze(2)
            decoder_outputs_adj_2 = self.adj2_linear[lvl](hs[lvl]).unsqueeze(1)
            decoder_outputs_adj_1 = decoder_outputs_adj_1.expand(-1, 300, 300, -1)
            decoder_outputs_adj_2 = decoder_outputs_adj_2.expand(-1, 300, 300, -1)
            decoder_outputs_adj = torch.cat((decoder_outputs_adj_1, decoder_outputs_adj_2), dim=-1)

            decoder_outouts_query = self.adj_query[lvl](q[lvl]).unsqueeze(2)
            decoder_outouts_key = self.adj_key[lvl](k[lvl]).unsqueeze(1)
            decoder_outouts_query_boardcast = decoder_outouts_query.expand(-1, 300, 300, -1)
            decoder_outouts_key_boardcast = decoder_outouts_key.expand(-1, 300, 300, -1)
            decoder_outouts_query_key = torch.cat((decoder_outouts_query_boardcast, decoder_outouts_key_boardcast), dim=-1)

            ga=self.WG[lvl](decoder_outouts_query_key).sigmoid()
            gz=self.WG[lvl](decoder_outputs_adj).sigmoid()

            output_adj_end = gz*decoder_outputs_adj + ga*decoder_outouts_query_key
            outputs_adj = self.adj_embed[lvl](output_adj_end).squeeze(-1)

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_adjs.append(outputs_adj)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_adj = torch.stack(outputs_adjs)

        outputs_classes_grasp = []
        outputs_coords_grasp = []
        outputs_angles_grasp = []

        for lvl in range(hs_grasp.shape[0]):
            if lvl == 0:
                reference_grasp = init_reference_grasp

            else:
                reference_grasp = inter_references_grasp[lvl - 1]
            reference_grasp = inverse_sigmoid(reference_grasp)
            outputs_class_grasp = self.class_embed_grasp[lvl](hs_grasp[lvl])
            outputs_angle_grasp = self.angle_embed_grasp[lvl](hs_grasp[lvl])

            tmp_grasp = self.bbox_embed_grasp[lvl](hs_grasp[lvl])
            if reference_grasp.shape[-1] == 4:
                tmp_grasp += reference_grasp
            else:
                assert reference_grasp.shape[-1] == 2
                tmp_grasp[..., :2] += reference_grasp
            outputs_coord_grasp = tmp_grasp.sigmoid()
            outputs_classes_grasp.append(outputs_class_grasp)
            outputs_angles_grasp.append(outputs_angle_grasp)    
            outputs_coords_grasp.append(outputs_coord_grasp)
        outputs_class_grasp = torch.stack(outputs_classes_grasp)
        outputs_coord_grasp = torch.stack(outputs_coords_grasp)
        outputs_angle_grasp = torch.stack(outputs_angles_grasp)

        out = {
                'pred_logits': outputs_class[-1], 
                'pred_boxes': outputs_coord[-1],
                'pred_logits_grasp': outputs_class_grasp[-1], 
                'pred_boxes_grasp': outputs_coord_grasp[-1],
                'pred_angles_grasp': outputs_angle_grasp[-1],
                'pred_adj': outputs_adj[-1]
               }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_grasp, outputs_coord_grasp, outputs_angle_grasp, outputs_adj)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}


        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_grasp, outputs_coord_grasp, outputs_angle_grasp, output_adj):

        return [{'pred_logits': a, 'pred_boxes': b, 'pred_logits_grasp': c, 'pred_boxes_grasp': d, 'pred_angles_grasp': e, 'pred_adj': f}
                for a, b, c, d, e, f in zip(
                                         outputs_class[:-1], 
                                         outputs_coord[:-1], 
                                         outputs_class_grasp[:-1], 
                                         outputs_coord_grasp[:-1],
                                         outputs_angle_grasp[:-1],
                                         output_adj[:-1]
                                         )]


class SetCriterion(nn.Module):

    def __init__(self, num_classes, weight_dict, losses, focal_alpha=0.25, matcher = None, matcher_grasp = None):

        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        # self.aux_losses = ['labels', 'boxes', 'grasp_class', 'grasp_bbox', 'grasp_angle']
        self.focal_alpha = focal_alpha
        self.matcher_grasp = matcher_grasp


    def loss_labels(self, outputs, targets, num_boxes, indices, num_grasp_boxes, indices_grasp, log=True):

        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses


    def loss_boxes(self, outputs, targets, num_boxes, indices, num_grasp_boxes, indices_grasp):

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    def loss_grasp_class(self, outputs, targets, num_boxes, indices, num_grasp_boxes, indices_grasp):

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
        losses = {'loss_grasp_ce': loss_ce}

        return losses

    def loss_grasp_angle(self, outputs, targets, num_boxes, indices, num_grasp_boxes, indices_grasp):
        
        assert 'pred_angles_grasp' in outputs

        src_logits = outputs['pred_angles_grasp']

        idx = self._get_src_permutation_idx(indices_grasp)
        target_classes_o = torch.cat([t["grasp_angles_cls"][J] for t, (_, J) in zip(targets, indices_grasp)])
        target_classes = torch.full(src_logits.shape[:2], 19,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o 

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1) 

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_grasp_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_grasp_angle': loss_ce}

        return losses

    def angle_focal_loss(self, inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):

        prob = inputs
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_boxes



    def loss_grasp_bbox(self, outputs, targets, num_boxes, indices, num_grasp_boxes, indices_grasp):

        assert 'pred_boxes_grasp' in outputs

        idx = self._get_src_permutation_idx(indices_grasp)
        src_boxes = outputs['pred_boxes_grasp'][idx]
        target_boxes = torch.cat([t['grasp_points'][i].view(-1, 4, 2).mean(dim=1) for t, (_, i) in zip(targets, indices_grasp)], dim=0)

        target_w = torch.cat([t['grasp_widths'][i].unsqueeze(-1) for t, (_, i) in zip(targets, indices_grasp)], dim=0)
        target_h = torch.cat([t['grasp_heights'][i].unsqueeze(-1) for t, (_, i) in zip(targets, indices_grasp)], dim=0)

        target_combined = torch.cat((target_boxes, target_w, target_h), dim=1)

        losses = {}

        loss_bbox = F.smooth_l1_loss(src_boxes, target_combined, reduction='none')
        losses['loss_grasp_box'] = loss_bbox.sum() / num_grasp_boxes

        out_angle = outputs['pred_angles_grasp'][idx] 
        max_indices = torch.argmax(out_angle, dim=1)
        real_angle = (max_indices.unsqueeze(1) - 8) * 10 - 5

        tgt_angle = torch.cat([t['grasp_angles'][i].unsqueeze(-1) for t, (_, i) in zip(targets, indices_grasp)], dim=0)

        out_grasp = torch.cat((src_boxes * 10, real_angle),dim=1).unsqueeze(0)
        tgt_grasp = torch.cat((target_combined * 10, tgt_angle),dim=1).unsqueeze(0)

        eiou_loss = cal_eiou(out_grasp, tgt_grasp).squeeze(0)

        loss = eiou_loss
        losses['loss_grasp_ious'] = loss.sum() / num_grasp_boxes

        return losses

    def loss_adj_matrix(self, outputs, targets, num_boxes, indices, num_grasp_boxes, indices_grasp):
        """
        Adjacency matrix loss
        Ap -> Am
        """
        assert 'pred_adj' in outputs
        pred_adj = outputs['pred_adj'] 

        loss_adjs=[]

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):

            pred_adj_part = pred_adj[batch_idx][src_idx][:, src_idx]
            pred_adj_perm = pred_adj_part.clone()
            for i in range(len(tgt_idx)):
                for j in range(len(tgt_idx)):
                    pred_adj_perm[tgt_idx[i],tgt_idx[j]]=pred_adj_part[i,j]

            target_adj = targets[batch_idx]["adj"]

            pred_adj_perm = pred_adj_perm.float()

            target_adj = target_adj.float()

            loss_adj = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_adj_perm, target_adj, reduction='sum')/num_boxes
            loss_adjs.append(loss_adj)

        losses = {'loss_adj': sum(loss_adjs)}

        return losses

    def _get_src_permutation_idx(self, indices):

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):

        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, num_boxes, indices, num_grasp_boxes, indices_grasp, **kwargs):

        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'grasp_class':self.loss_grasp_class,
            'grasp_bbox':self.loss_grasp_bbox,
            'grasp_angle':self.loss_grasp_angle,
            'adj':self.loss_adj_matrix,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'

        return loss_map[loss](outputs, targets, num_boxes, indices, num_grasp_boxes, indices_grasp, **kwargs)


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        indices = self.matcher(outputs_without_aux, targets)
        indices_grasp = self.matcher_grasp(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        num_grasp_boxes = sum(len(t["grasp_classes"]) for t in targets)
        num_grasp_boxes = torch.as_tensor([num_grasp_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_grasp_boxes)
        num_grasp_boxes = torch.clamp(num_grasp_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, num_boxes, indices, num_grasp_boxes, indices_grasp, **kwargs))


        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                indices_grasp = self.matcher_grasp(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, num_boxes, indices, num_grasp_boxes, indices_grasp, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])

            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    continue
                kwargs = {}
                if loss == 'labels':
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_grasp_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):

    @torch.no_grad()
    def forward(self, outputs, target_sizes):

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

class AdaptivePooling(nn.Module):
    def __init__(self, hidden_dim, output_dim, output_tokens):
        super().__init__()
        self.output_tokens = output_tokens
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_tokens)
    def forward(self, x):

        batch_size, num_tokens, hidden_dim = x.shape

        x = x.permute(0, 2, 1) 
        x = self.adaptive_pool(x) 
        x = x.permute(0, 2, 1)  

        x = self.fc(x) 
        return x
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


def build_dab_deformable_detr(args):
    num_classes = 32 if args.dataset_file != 'coco' else 32

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DABDeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        two_stage=args.two_stage,
        use_dab=True,
        num_patterns=args.num_patterns,
        random_refpoints_xy=args.random_refpoints_xy
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher, matcher_grasp = build_matcher(args)

    weight_dict = {
                    'loss_ce': args.cls_loss_coef,
                    'loss_bbox': args.bbox_loss_coef,
                    'loss_giou': args.giou_loss_coef,                    
                    'loss_grasp_ce':args.grasp_ce_loss_coef, 
                    'loss_grasp_angle':args.grasp_angle_loss_coef,
                    'loss_grasp_ious':args.grasp_ious_loss_coef,
                    'loss_grasp_box':args.grasp_box_loss_coef,
                    'loss_adj':args.adj_loss_coef, 
                   }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'grasp_class', 'grasp_bbox', 'grasp_angle', 'adj']

    criterion = SetCriterion(num_classes, weight_dict, losses, focal_alpha=args.focal_alpha, matcher = matcher, matcher_grasp = matcher_grasp)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
