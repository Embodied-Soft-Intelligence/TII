# D2TriPO-DETR

> **Dual-Decoder Triple-Parallel-Output Detection Transformer**

This repository is an official implementation of the paper [D2TriPO-DETR: Dual-Decoder Triple-Parallel-Output Detection Transformer](https://mok1170.github.io/TII/).

<p align="center">
  <img src="https://github.com/mok1170/TII/blob/main/Fig.%203.jpg" alt="Fig. 3" />
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

## Abstract

Vision-based grasping is essential for industrial and household robotics but remains challenging in stacked-object scenarios. We introduce D2TriPO-DETR, a dual-decoder transformer producing three parallel outputs for object detection, manipulation relationship reasoning, and grasp detection. Producing the three outputs concurrently reduces error propagation between subtasks. On the Visual Manipulation Relationship Dataset (VMRD), D2TriPO-DETR outperforms prior methods on several key metrics and demonstrates strong real-world performance on a UR3 robot equipped with a parallel gripper.

## Features

- End-to-end transformer-based architecture
- Dual decoders generating three parallel outputs
- Modules for distributed attention and viewpoint adaptation
- Training and evaluation recipes compatible with Deformable-DETR-style codebases
- Real-robot validation (UR3 + RealSense + Robotiq gripper)

## Results

<div>

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

---

## Dataset Preparation

Organize the VMRD dataset as follows:

```
data/vmrd/
  ├── adj/
  ├── grasps/
  ├── images/
  └── labels/
```

---

## Training

Example (4 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py
```

---

## Evaluation

```bash
python evaluate.py
```

---

## Inference / Demo

```bash
python demo.py --img_path ./demo_images/example.jpg
```

---

## Real-World Experiments

This repository includes the code used for **real-world robotic grasping experiments**.  
To perform experiments using an actual robotic arm, run:

```bash
python predict.py
```

The script `predict.py` connects to a **UR robotic arm** equipped with an **Intel RealSense D435** depth camera and a **Robotiq 2F-85** parallel gripper.  
It performs **real-time grasp detection and execution** using the trained D2TriPO-DETR model.  
Before running, ensure that the robot, gripper, and camera are properly connected and configured.

---

## Real-World Experiment Results

<div align="center"> <table> <tr> <th>Models</th> <th>Cluttered Scenes (%)</th> <th>Stacked Scenes (%)</th> </tr> <tr> <td>Mutli-Task CNN[8]</td> <td align="center">90.60</td> <td align="center">65.65</td> </tr> <tr> <td>SMTNet[35]</td> <td align="center">86.13</td> <td align="center">65.00</td> </tr> <tr> <td>EGNet[5]</td> <td align="center">93.60</td> <td align="center">69.60</td> </tr> <tr> <td><strong>D2TriPO-DETR (Ours)</strong></td> <td align="center"><strong>95.71</strong></td> <td align="center"><strong>74.29</strong></td> </tr> </table> </div>

---

## Pretrained Models

(Optional) Add pretrained weights here.

---

## Citation

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

This project is released under the [Apache License 2.0](https://mok1170.github.io/TII/). See `LICENSE` for details.

---

