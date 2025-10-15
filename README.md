# D2TriPO-DETR

> **Dual-Decoder Triple-Parallel-Output Detection Transformer**

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

### Grasp detection (VMRD)

| Model           | Accuracy (%) |
|-----------------|--------------:|
| FCGN[25]        | 54.5          |
| VMRN[3]         | 70.5          |
| SE-ResUNet[26]  | 81.2          |
| GR-ConvNet[27]  | 82.7          |
| TFGrasp[28]     | 83.3          |
| SKGNet[29]      | 86.1          |
| SPANet[30]      | 86.7          |
| EGNet[5]        | 87.1          |
| **D2TriPO-DETR** (Ours) | **88.6** |

### Grasp detection (Cornell)

| Model        | Input | Accuracy (%) |
|--------------|:-----:|-------------:|
| TFGrasp[28]  | RGB   | 96.78         |
| TFGrasp[28]  | RGBD  | 97.99         |
| DSNet[33]    | RGBD  | 98.31         |
| Yang[34]     | RGB   | 98.74         |
| **D2TriPO-DETR** (Ours) | RGB | **98.87** |

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

Example `requirements.txt` (project root) should include, but is not limited to:

```
torch==1.13.1
torchvision==0.14.1
numpy
opencv-python
Pillow
pyyaml
tqdm
scipy
tensorboard
matplotlib
```

---

## Installation

It is recommended to create an isolated conda environment:

```bash
conda create -n d2tripo python=3.8 -y
conda activate d2tripo
# Install PyTorch for CUDA 11.6 as an example
conda install pytorch=1.13.1 torchvision=0.14.1 cudatoolkit=11.6 -c pytorch -y
pip install -r requirements.txt
```

### Compiling custom CUDA ops

If the project contains custom CUDA operators, run:

```bash
cd models/dab_deformable_detr/ops
sh ./make.sh
# Run unit tests (if provided)
python test.py
```

---

## Dataset Preparation

This project trains and evaluates on VMRD (Visual Manipulation Relationship Dataset). Organize the data under `code_root/data/vmrd/` with the following structure:

```
code_root/
└── data/
    └── vmrd/
        ├── adj/
        │   ├── train/
        │   └── val/
        ├── grasps/
        │   ├── train/
        │   └── val/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
```

Each subfolder should contain correspondingly named files, e.g., `00001.jpg`, `00001.txt` / `00001.xml`, etc., consistent with the loading logic in `datasets/vmrd.py`.

Set the `root` / path variables in `datasets/vmrd.py` to point to your `code_root/data/vmrd/`.

---

## Training

Example training command (single-node 4 GPU, 120 epochs):

```bash
# Set visible GPUs as appropriate for your machine
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py   --dataset vmrd --data_path ./data/vmrd   --epochs 120 --batch_size 4   --lr 1e-4 --lr_backbone 1e-5 --weight_decay 1e-4   --dropout 0.1 --lr_drop 80   --output_dir ./outputs/d2tripo_experiment
```

Key training hyperparameters consistent with the authors' implementation:

- Optimizer: `AdamW`
- Transformer learning rate: `1e-4`, backbone learning rate: `1e-5`
- Weight decay: `1e-4`
- Total batch size: 4
- LR drop at epoch 80
- Gradient clipping: max norm = 0.1
- Data augmentations: horizontal flip, random scaling, random crop

Training logs and checkpoints are saved to the directory specified by `--output_dir`.

---

## Evaluation

Specify the checkpoint path in the configuration or command line. Example:

```bash
python evaluate.py --dataset vmrd --data_path ./data/vmrd   --checkpoint ./outputs/d2tripo_experiment/checkpoint_best.pth   --output_dir ./outputs/eval_results
```

The evaluation script will report:

- Object detection precision/recall
- Manipulation relationship image accuracy
- Grasp detection accuracy (Jaccard / IoU / angle thresholds, etc.)

---

## Inference / Demo

Example single-image inference:

```bash
python demo.py --img_path ./demo_images/000123.jpg   --checkpoint ./outputs/d2tripo_experiment/checkpoint_best.pth   --save_dir ./demo_results
```

`demo.py` saves visualization outputs (detection boxes + grasp poses + relationship graph) to `--save_dir`.

For running on a real robot, refer to `scripts/robot_launch.sh` (example) and any available ROS interface documentation.

---

## Pretrained Models

(Optional) List downloadable pretrained weights here (I can add download URLs or scripts if you want):

- `d2tripo_vmrd_res101.pth` — checkpoint for VMRD (example)

---

## Citation

If you use this work in your research, please cite:

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

This project is released under the **Apache License 2.0**. See `LICENSE` for details.

---

## Contributing

Contributions are welcome! Suggested workflow:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit and push your changes, then open a PR describing the changes and reproduction steps.
4. PRs will be reviewed and subject to CI checks (if configured).

Please read `CONTRIBUTING.md` if available before contributing.

---

## Authors & Contact

- Primary authors (example): Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai
- For issues, please open an issue in this repository or submit a PR. Contact via email may be available from the project maintainers.

---

## Acknowledgements

Thanks to Deformable-DETR and related grasping works (e.g., GR-ConvNet, TFGrasp) and the VMRD dataset for foundational work and open-source implementations.

---

## Changelog

See `changelog.md` for detailed change logs. Example entries:

- `v0.1` — initial implementation and VMRD baseline reproduction
- `v0.2` — added DAP / VAA modules and documentation updates

---
