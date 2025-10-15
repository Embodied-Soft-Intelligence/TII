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
import numpy as np
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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
# import onnxruntime
# import netron
def main():
    main_args = get_main_args_parser().parse_args()
    # 加载模型
    model = load_model('/root/autodl-tmp/results_liner_open/checkpoint0104.pth', main_args)
    device = torch.device('cuda')
    batch_size = 1
    height = 768
    width = 1024
    dummy_input = torch.rand(batch_size, 3, height, width).to(device)
    # dummy_input1 = torch.randn(batch_size, 3, width, height,dtype=torch.float).to(device)
    # torch.onnx.export(model,
    #                 dummy_input,
    #                 '/home/pmh/nvme1/Code/dab_deformable_swin_multi/dab_deform_swin_4d_34.onnx',)
    # torch.onnx.export(model,
    #                 dummy_input1,
    #                 '/home/pmh/nvme1/Code/dab_deformable_swin_multi/dab_deform_swin_4d_43.onnx',)
    # onnx_model_path = "/home/pmh/nvme1/Code/dab_deformable_swin_multi/dab_deform_swin_4d_34.onnx"
    # # netron.start(onnx_model_path)
    # providers= [
    # (
    #     "CUDAExecutionProvider",
    #     {
    #         "device_id": torch.cuda.current_device(),
    #         # "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,

    #     },
    # )
    # ]
    # resnet_session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
    # inputs={resnet_session.get_inputs()[0].name: to_numpy(dummy_input)}
    # providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    repetitions = 300
    timings=np.zeros((repetitions,1))
    for _ in range(10):
        #  a = model(dummy_input)
        # a=resnet_session.run(None, inputs)[0]
        _=model(dummy_input)
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            # _=resnet_session.run(None, inputs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        mean_fps = 1000. / mean_syn
        print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
        print(mean_syn)
if __name__ == "__main__":
    main()