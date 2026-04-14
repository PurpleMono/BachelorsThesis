# Bachelor's Thesis — DINOv2-Based Industrial Anomaly Detection Benchmark

**Author:** Jeremi Degenhardt  
**Institution:** FAU Erlangen-Nürnberg, Chair of IT Management (Prof. Amberg)  
**Supervisor:** René Gröbner  

---

## Research Overview

This thesis benchmarks three DINOv2-based multi-class anomaly detection models under identical conditions, with a focus on cross-viewpoint robustness in industrial inspection settings.

All three models share the same frozen DINOv2 ViT-Base/14 backbone. The only variable is how each model defines normality:

| Model | Mechanism | Venue |
|-------|-----------|-------|
| AnomalyDINO | Training-free nearest-neighbour memory bank | WACV 2025 |
| Dinomaly | Reconstruction-based with trained decoder | CVPR 2025 |
| INP-Former | Reconstruction guided by test-image prototypes | CVPR 2025 |

## Research Questions

**RQ1:** Does the detection mechanism influence cross-viewpoint robustness when the DINOv2 backbone is held constant?

**RQ2:** Which combinations of object category, viewpoint, and defect type represent systematic failure modes, and do these differ across detection mechanisms?

**RQ3:** How do the three models compare on the accuracy-latency trade-off on identical hardware (Google Colab T4)?

## Datasets

- **Real-IAD** (primary) — 30 categories, 5 viewpoints, ~150k images
- **MVTec AD** (secondary) — 15 categories, standard benchmark
- **MVTec AD 2** (supplementary) — 8 categories, harder conditions

## Evaluation Protocols

- **Standard protocol** — train all 5 views, test all 5 views
- **Cross-viewpoint protocol** — train on C1+C2, test on C3-C5

## Metrics

I-AUROC, P-AUROC, AUPRO, S-AUROC, Performance Degradation Ratio, WGA, Inference Time, Memory Footprint

## Repository Structure
├── evaluation/          # metric computation and analysis code
├── data/                # dataset loading utilities
├── notebooks/           # Colab experiment notebooks
├── results/             # generated outputs (not committed)
└── requirements.txt

## Setup
```bash
git clone https://github.com/PurpleMono/BachelorsThesis.git
cd BachelorsThesis
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Setup

This project uses the Real-IAD dataset (512px version) from Hugging Face.
The dataset is publicly available — no account required.

Dataset link: https://huggingface.co/datasets/Real-IAD/Real-IAD

When running experiment notebooks the dataset will be downloaded automatically
via the Hugging Face datasets library.


## Hardware

All experiments run on Google Colab for hardware-consistent comparisons. GPU tier subject to change based on computational requirements.