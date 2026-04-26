# Memory, Reconstruction, or Prototypes?
## Benchmarking DINOv2-Based Models for Multi-Class Industrial Anomaly Detection

Bachelor thesis — Wirtschaftsinformatik, FAU Erlangen-Nürnberg
Author: Jeremi Degenhardt
Supervisor: René Gröbner

---

## Overview

This repository contains the complete implementation for a benchmarking
study comparing three DINOv2-based anomaly detection models on the
Real-IAD multi-view industrial dataset.

| Model | Paradigm | Venue | Backbone |
|---|---|---|---|
| AnomalyDINO | Memory-based (training-free) | WACV 2025 | DINOv2-Register ViT-Base/14 |
| Dinomaly | Reconstruction-based | CVPR 2025 | DINOv2-Register ViT-Base/14 |
| INP-Former | Prototype-based | CVPR 2025 | DINOv2-Register ViT-Base/14 |

Dataset: Real-IAD 512px (30 categories, 5 viewpoints)
Secondary: MVTec AD (implementation validation only)

---

## Repository Structure

BachelorsThesis/
├── data/
│   ├── realiad_utils.py       — Real-IAD data loader
│   └── realiad_dataset.py     — PyTorch Dataset wrapper
├── evaluation/
│   ├── metrics.py             — I-AUROC, S-AUROC, P-AUROC, AUPRO
│   ├── wga.py                 — Worst-Group Analysis
│   └── visualisation.py      — Figures and anomaly map comparison
├── models/
│   ├── inp_former/            — INP-Former submodule (forked)
│   └── trainer.py             — Unified training and inference
├── notebooks/
│   ├── 00_setup.ipynb         — Dataset download and preparation
│   ├── 00b_mvtec_validation.ipynb — AnomalyDINO implementation check
│   ├── 01_smoke_test.ipynb    — Pipeline validation
│   ├── 02_standard_protocol.ipynb — Main benchmark (all 30 categories)
│   ├── 03_crossview_protocol.ipynb — Viewpoint robustness
│   ├── 04_ablation_study.ipynb — Systematic ablation investigations
│   └── 05_analysis.ipynb      — Figures and tables for thesis
└── results/                   — Saved scores, figures, anomaly maps

---

## Replication Steps

1. Run `00_setup.ipynb` — download Real-IAD from HuggingFace
2. Run `00b_mvtec_validation.ipynb` — validate AnomalyDINO on MVTec AD
3. Run `02_standard_protocol.ipynb` — main experiment
4. Run `03_crossview_protocol.ipynb` — cross-view robustness
5. Run `04_ablation_study.ipynb` — ablation investigations
6. Run `05_analysis.ipynb` — generate all thesis figures

Each notebook has a USER CONFIGURATION block at the top.
Set repo_path, dataset_root, and MAPS_SAVE_DIR before running.

---

## Key Implementation Notes

- All three models use frozen DINOv2-Register ViT-Base/14 backbone
- Dinomaly: dropout rate 0.4 and image score top 0.1% corrected for Real-IAD
- INP-Former: official repo with path fixes for 512px Real-IAD
- AnomalyDINO: ViT-Base/14 backbone (deviation from published ViT-Small,
  justified for backbone-controlled paradigm comparison)
- Anomaly maps saved as compressed numpy (.npz) for anomalous images only

---

## Requirements

See `requirements.txt` for the full dependency list.

Recommended environment: Google Colab with T4 or L4 GPU.
Install dependencies with:
```
pip install -r requirements.txt
```