# GHOST: Geometry-Guided Hallucination of Opaque Surface Textures

[![arXiv](https://img.shields.io/badge/arXiv-2607.11118-b31b1b.svg)](https://arxiv.org/abs/2607.11118)
[![ICML](https://img.shields.io/badge/ICML-2026-4b6bff.svg)](https://icml.cc/)

[ICML 2026] Official implementation of **GHOST**, a geometry-guided preprocessing framework that transforms transparent regions into opaque, structurally consistent RGB images to improve downstream depth estimation and 3D reconstruction — without retraining the target models.

## Pipeline

GHOST consists of four modules:

1. **TransDINO** — transparent object mask prediction  
2. **TransDecomp** — transparency decomposition (alpha / foreground)  
3. **DAF-Net** — surface normal prior estimation  
4. **GeoSemTransNet** — opaque RGB hallucination from multi-modal cues  

## Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- OpenCV, NumPy
- DINOv3 (included under `dinov3/`)

```bash
pip install torch torchvision opencv-python numpy
```

## Checkpoints

Place pretrained weights under `checkpoints/`:

| File | Module |
|------|--------|
| `transdino.pth` | TransDINO |
| `transdecomp.pth` | TransDecomp |
| `dafnet.pth` | DAF-Net |
| `geosemtransnet.pth` | GeoSemTransNet |
| `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` | DINOv3 backbone |

## Inference

Prepare an input RGB image (e.g. `images/rgb.jpg`), then run the pipeline step by step:

```bash
# 1. Predict transparent object mask
python inference_TransDINO.py

# 2. Decompose transparency (alpha / foreground)
python inference_TransDecomp.py

# 3. Estimate surface normals
python inference_DAFNet.py

# 4. Hallucinate opaque RGB
python inference_GeoSemTransNet.py
```

Edit the input/output paths at the bottom of each script as needed.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{zhao2026ghostgeometryguidedhallucinationopaque,
      title={GHOST: Geometry-Guided Hallucination of Opaque Surface Textures}, 
      author={Langxu Zhao and Zuan Gu and Tianhan Gao},
      year={2026},
      eprint={2607.11118},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.11118}, 
}
```

## Acknowledgments

This repository uses [DINOv3](https://github.com/facebookresearch/dinov3) as the visual foundation backbone.
