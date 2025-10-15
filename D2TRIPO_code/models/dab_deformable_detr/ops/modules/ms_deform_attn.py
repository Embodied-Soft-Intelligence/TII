# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction
def sample_points():
    point=torch.zeros(8,4,2)
    for m in range(8):
        for k in range(4):
            dx=math.cos((m%2+2*k)*math.pi/4)
            dy=math.sin((m%2+2*k)*math.pi/4)
            a=(m+2)//2
            b=max(abs(dx),abs(dy))
            c=a/b
            x=dx*c
            y=dy*c
            point[m][k][0]=round(x)
            point[m][k][1]=round(y)
    # point=point[None,None,:,None,:,:].repeat(4,300,1,4,1,1)
    return point

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.sampling_points = sample_points()
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    # def sample_points(self, point):
    #     for m in range(8):
    #         for k in range(4):
    #             dx=math.cos((m%2+2*k)*math.pi/4)
    #             dy=math.sin((m%2+2*k)*math.pi/4)
    #             a=(m+2)//2
    #             b=max(abs(dx),abs(dy))
    #             c=a/b
    #             x=dx*c
    #             y=dy*c
    #             point[m][k][0]=round(x)
    #             point[m][k][1]=round(y)
    #     point=point[None,None,:,None,:,:].repeat(8,300,1,4,1,1)
    #     return point
    
    # def get_rot_mat(self, theta):
    #     # 创建一个空 tensor，用于存储结果
    #     mat_list = []
        
    #     for i in theta:
    #         # 将角度从度数转换为弧度
    #         j = i * math.pi / 180
            
    #         # 计算 2x2 的旋转矩阵
    #         mat = torch.tensor([[torch.cos(j), -torch.sin(j)],
    #                             [torch.sin(j), torch.cos(j)]])
    #         mat_list.append(mat.T)
        
    #     # 将结果列表转化为 tensor，形状为 [300, 2, 2]
    #     return torch.stack(mat_list)
    def get_rot_mat(self, theta):
        # 将角度从度数转换为弧度，并进行批量处理
        theta_rad = theta * math.pi / 180

        # 使用广播来计算 cos 和 sin
        cos_vals = torch.cos(theta_rad)
        sin_vals = torch.sin(theta_rad)

        # 构造旋转矩阵，批量处理
        rot_mat = torch.stack([cos_vals, sin_vals, -sin_vals, cos_vals], dim=-1)
        
        # 将旋转矩阵的形状调整为 [300, 2, 2]
        rot_mat = rot_mat.view(-1, 2, 2)

        return rot_mat
    
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, 
                reference_angle_grasp = None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
    
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            if reference_angle_grasp is None:
                sampling_locations = reference_points[:, :, None, :, None, :2] \
                                    + sampling_offsets  / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            elif reference_angle_grasp is not None:
                bias = self.sampling_offsets.bias.detach().view(self.n_heads, self.n_levels, self.n_points, 2).repeat(N,Len_q,1,1,1,1)
                point = self.sampling_points.to(device=reference_points.device)
                point = point[None,None,:,None,:,:].repeat(N,Len_q,1,4,1,1)
                _, max_indices = torch.max(reference_angle_grasp, dim=-1)
                angle_grasp=(max_indices[0,:,0]-8)*10-5
                rot_mat=self.get_rot_mat(angle_grasp).to(reference_points.device)
                sampling_offsets = (sampling_offsets - bias + point)  / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5 
                sampling_offsets = torch.matmul(sampling_offsets.unsqueeze(-2), rot_mat.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4))
                sampling_offsets = sampling_offsets.squeeze(-2)
                sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets
                
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output
