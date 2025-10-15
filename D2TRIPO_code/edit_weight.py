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
from collections import OrderedDict
from torchstat import stat
def load_model(model_path , args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] 当前使用{}做推断".format(device))
    model, _, _ =  build_dab_deformable_detr(args)
    # model.cuda()
    model.eval()
    # stat(model,(3,800,1066))
    state_dict = torch.load(model_path) # <-----------修改加载模型的路径
    # for k, v in enumerate(state_dict["model"]):
    #     print(v)
    # model.load_state_dict(state_dict["model"])
    # missing_keys, unexpected_keys = model.load_state_dict(state_dict["model"], strict=False)
    # print(missing_keys)
    # print(unexpected_keys)
    # for k, v in model.state_dict().items():
    #     print(k)
    new_dict = OrderedDict()
    new_dict = state_dict["model"]
    # new_dict["tgt_embed_grasp.weight"]=state_dict["model"]["tgt_embed.weight"]
    # # del new_dict["tgt_embed.weight"]
    # new_dict["refpoint_embed_grasp.weight"]=state_dict["model"]["refpoint_embed.weight"]
    # del new_dict["refpoint_embed.weight"]
    # new_dict = OrderedDict()
    for key in list(state_dict["model"].keys()):
        if 'transformer.decoder' in key:
                # 生成新的键名，将 'transformer.decoder' 替换为 'transformer.decoder_grasp'
            new_key = key.replace('transformer.decoder', 'transformer.decoder_grasp')
            # 将原来的值复制到新的键中
            new_dict[new_key] = state_dict["model"][key]   
        elif 'class_embed' in key:
            # new_key = key.replace('class_embed', 'class_embed_grasp')
            # new_dict[new_key] = state_dict["model"][key]   
            del new_dict[key]
        elif 'bbox_embed' in key:
            new_key = key.replace('bbox_embed', 'bbox_embed_grasp')
            new_dict[new_key] = state_dict["model"][key]   
        elif 'tgt_embed' in key:
            new_key = key.replace('tgt_embed', 'tgt_embed_grasp')
            new_dict[new_key] = state_dict["model"][key]   
        elif 'refpoint_embed' in key:
            new_key = key.replace('refpoint_embed', 'refpoint_embed_grasp')  
            new_dict[new_key] = state_dict["model"][key]  
        # elif 'backbone' in key:
        #     del new_dict[key] 
        else:
            pass  


    print(new_dict.keys())
    test={'model':new_dict}
    torch.save(test, '/home/pmh/code/dab_deformable_resnet_5d_Rotate_Point/resnet50_doubledetr' + '.pth')  
    for k, v in new_dict.items():
        print(k)
    # print("Missing keys:", missing_keys)
    # print("Unexpected keys:", unexpected_keys)
    # model.to(device)
    print("load model sucess")
    return model
main_args = get_main_args_parser().parse_args()
model = load_model('/root/autodl-tmp/results/checkpoint0120.pth', main_args)
