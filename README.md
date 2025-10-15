# D2TriPO-DETR

> **Dual-Decoder Triple-Parallel-Output Detection Transformer**

<p align="center">
  <img src="https://github.com/mok1170/TII/blob/main/Fig.%203.jpg" alt="Fig. 3" />
</p>

---

## Introduction

D2TriPO-DETR is an end-to-end vision-based grasp detection framework targeting object-stacking scenarios. It addresses three common challenges of existing methods — limited inter-object relationship understanding, poor grasp adaptation across viewpoints, and error propagation — via a dual-decoder transformer architecture that produces three parallel outputs: object detection, manipulation-relationship reasoning, and grasp detection.

## Abstract

Vision-based grasping remains essential for industrial and household robotics but struggles in stacked-object scenarios. We introduce D2TriPO-DETR: a dual-decoder transformer with triple-parallel outputs for object detection, manipulation relationship reasoning, and grasp detection. Two novel modules — a Distributed Attention Perception (DAP) module and a Visual Attention Adaptation (VAA) module — are integrated into the parallel decoders to enhance inter-object relational reasoning and viewpoint adaptation. Because the three outputs are produced concurrently, the design intrinsically reduces error propagation between subtasks. On the Visual Manipulation Relationship Dataset (VMRD), D2TriPO-DETR outperforms prior methods across metrics, e.g. +5.3% precision (object detection), +6.7% image accuracy (manipulation relationship), and +1.5% accuracy (grasp detection). Real-world robotic tests validate the approach on grasping stacked objects.

## License

This project is released under the **Apache 2.0** license.

## Changelog

See `changelog.md` for detailed logs of major changes.

## Citing D2TriPO-DETR

If you use this work in research, please cite:

```bibtex
@article{zhu2020deformable,
  title={D2TriPO-DETR: Dual-Decoder Triple-Parallel-Output Detection Transformer},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
