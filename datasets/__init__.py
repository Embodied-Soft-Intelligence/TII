# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .vmrd import build as build_vmrd

def build_dataset(image_set):
    return build_vmrd(image_set)