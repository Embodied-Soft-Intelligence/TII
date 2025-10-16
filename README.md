# D2TriPO-DETR

> **Dual-Decoder Triple-Parallel-Output Detection Transformer**

This repository is an official implementation of the paper [D2TriPO-DETR: Dual-Decoder Triple-Parallel-Output Detection Transformer](https://mok1170.github.io/TII/).

<p align="center">
  <img src="https://github.com/Embodied-Soft-Intelligence/TII/blob/main/picture/3.png" alt="Fig. 3" />
</p>

---

## Table of Contents

- [Introduction](#introduction)
- [Abstract](#abstract)
- [Features](#features)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference / Demo](#inference--demo)
- [Real-World Experiments](#real-world-experiments)
- [Pretrained Models](#pretrained-models)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)
- [Authors & Contact](#authors--contact)
- [Acknowledgements](#acknowledgements)
- [Changelog](#changelog)

---

## Introduction

D2TriPO-DETR is an end-to-end vision-based grasp detection framework designed for object-stacking scenarios. It adopts a dual-decoder transformer architecture with **triple-parallel outputs**: object detection, manipulation relationship reasoning, and grasp detection. Two modules — Distributed Attention Perception (DAP) and Visual Attention Adaptation (VAA) — are integrated to improve inter-object relational reasoning and viewpoint adaptation.

---

## Abstract

Vision-based grasping is essential for industrial and household robotics but remains challenging in stacked-object scenarios. We introduce D2TriPO-DETR, a dual-decoder transformer producing three parallel outputs for object detection, manipulation relationship reasoning, and grasp detection. Producing the three outputs concurrently reduces error propagation between subtasks. On the Visual Manipulation Relationship Dataset (VMRD), D2TriPO-DETR outperforms prior methods on several key metrics and demonstrates strong real-world performance on a UR3 robot equipped with a parallel gripper.

---

## Features

- End-to-end transformer-based architecture
- Dual decoders generating three parallel outputs
- Modules for distributed attention and viewpoint adaptation
- Training and evaluation recipes compatible with Deformable-DETR-style codebases
- Real-robot validation (UR3 + RealSense + Robotiq gripper)

---

## Main Results

<div align="center">

<table>
<tr>
<td valign="top">

### Grasp detection (VMRD)

<table>
  <tr><th align="left">Model</th><th align="right">Accuracy (%)</th></tr>
  <tr><td>FCGN[25]</td><td align="right">54.5</td></tr>
  <tr><td>VMRN[3]</td><td align="right">70.5</td></tr>
  <tr><td>SE-ResUNet[26]</td><td align="right">81.2</td></tr>
  <tr><td>GR-ConvNet[27]</td><td align="right">82.7</td></tr>
  <tr><td>TFGrasp[28]</td><td align="right">83.3</td></tr>
  <tr><td>SKGNet[29]</td><td align="right">86.1</td></tr>
  <tr><td>SPANet[30]</td><td align="right">86.7</td></tr>
  <tr><td>EGNet[5]</td><td align="right">87.1</td></tr>
  <tr><td><strong>D2TriPO-DETR (Ours)</strong></td><td align="right"><strong>88.6</strong></td></tr>
</table>

</td>

<td valign="top" style="padding-left: 40px;">

### Grasp detection (Cornell)

<table>
  <tr><th align="left">Model</th><th align="center">Input</th><th align="right">Accuracy (%)</th></tr>
  <tr><td>TFGrasp[28]</td><td align="center">RGB</td><td align="right">96.78</td></tr>
  <tr><td>TFGrasp[28]</td><td align="center">RGBD</td><td align="right">97.99</td></tr>
  <tr><td>DSNet[33]</td><td align="center">RGBD</td><td align="right">98.31</td></tr>
  <tr><td>Yang[34]</td><td align="center">RGB</td><td align="right">98.74</td></tr>
  <tr><td><strong>D2TriPO-DETR (Ours)</strong></td><td align="center">RGB</td><td align="right"><strong>98.87</strong></td></tr>
</table>

</td>
</tr>
</table>

</div>

> All D2TriPO-DETR experiments reported here used a total batch size of 4 (see code for implementation details).

---

## Real-World Experiment Results

<div align="center"> <table> <tr> <th>Models</th> <th>Cluttered Scenes (%)</th> <th>Stacked Scenes (%)</th> </tr> <tr> <td>Mutli-Task CNN[8]</td> <td align="center">90.60</td> <td align="center">65.65</td> </tr> <tr> <td>SMTNet[35]</td> <td align="center">86.13</td> <td align="center">65.00</td> </tr> <tr> <td>EGNet[5]</td> <td align="center">93.60</td> <td align="center">69.60</td> </tr> <tr> <td><strong>D2TriPO-DETR (Ours)</strong></td> <td align="center"><strong>95.71</strong></td> <td align="center"><strong>74.29</strong></td> </tr> </table> </div>

---

## Requirements

- OS: Linux (recommended)
- CUDA >= 11.6
- Python >= 3.8
- PyTorch >= 1.13.1
- torchvision >= 0.14.1
- Additional: git, cmake, GCC compatible with CUDA
- Optional: 4 × NVIDIA GPUs (the authors used 4 × RTX 4090 in their experiments)

---

## Installation

```bash
conda create -n d2tripo python=3.8 -y
conda activate d2tripo
conda install pytorch=1.13.1 torchvision=0.14.1 cudatoolkit=11.6 -c pytorch -y
pip install -r requirements.txt
```

## Compiling CUDA operators

```bash
cd ./models/dab_deformable_detr/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
---

## Usage
### Dataset Preparation

Please download VMRD dataset and organize them as following:

```
code_root/
└── data/
    └── vmrd/
        ├── adj/
        	├── train
        	└── val
               └──00001.txt
        └── grasps/
        	├── train
        	└── val
               └──00001.txt
        └── images/
        	├── train
        	└── val
               └──00001.jpg
        └── labels/
        	├── train
        	└── val
               └──00001.txt
        └── xml/
        	├── train
        	└── val
               └──00001.xml
```

---

### Training

To train the baseline D2TriPO-DETR on a single node with 4 GPUs for 120 epochs, first modify the dataset path in `./datasets/vmrd.py`, then download the resnet101_doubledetr weight file and place it in the project root directory, and finally run the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py
```
120 epoch training takes around 15 hours on a single machine with 4 RTX4090 cards. To ease reproduction of our results we provide results for the best checkpoint.pth for 120 epoch, achieving 88.6 accuracy.
We train D2TriPO-DETR with the AdamW optimizer, using a learning rate of 1e-4 for the transformer and 1e-5 for the backbone, and a weight decay of 1e-4. Training uses a batch size of 4 for 120 epochs, with the learning rate dropped at epoch 80. Data augmentation includes horizontal flipping, random scaling, and random cropping. The transformer uses a dropout of 0.1, and gradient clipping with a maximum norm of 0.1 is applied to stabilize training.

---

### Evaluation

You can obtain the configuration file and pretrained model of D2TriPO-DETR, then modify the folder path of the weight files to be batch-evaluated and the batch size in the evaluation script, and run :
```bash
python evaluate.py
```

---

## Real-World Experiments

This repository includes the code used for **real-world robotic grasping experiments**.  
The code path for the real-world experiment is `./robot/predict.py`. To perform experiments using an actual robotic arm, run:

```bash
python predict.py
```

The script `predict.py` connects to a **UR robotic arm** equipped with an **Intel RealSense D435** depth camera and a **Robotiq 2F-85** parallel gripper.  It performs **real-time grasp detection and execution** using the trained D2TriPO-DETR model.  Before running, ensure that the robot, gripper, and camera are properly connected and configured.

---

## Citation

If you find D2TriPO-DETR useful in your research, please consider citing:

```bibtex
@article{zhu2020deformable,
  title={D2TriPO-DETR: Dual-Decoder Triple-Parallel-Output Detection Transformer},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
```

---

## License

This project is released under the [D2TriPO-DETR](https://mok1170.github.io/TII/). See `LICENSE` for details.

---

