# Federated Foundation Models over Vehicular Networks

**Kasra Borazjani, Fardis Nadimi, Payam Abdisarabshali, Owen Palinski, Allan Salihovic,
Dinh Nguyen, Minghui Liwang, and Seyyedali Hosseinalipour**

---

## Overview

This repository implements **FedAdapt**, the M3T FedFM framework introduced in *Federated Foundation Models over Vehicular Networks*. FedAdapt adapts a shared CLIP ViT-B/16 foundation model across 20 geo-distributed vehicle clients for simultaneous multi-task autonomous driving perception — without centralising raw sensor data.

The central case study is **task onboarding**: given that three tasks are already pre-trained and frozen across the federation, a new (arriving) task is onboarded using modality-specific adapters, a shared task adapter, and Conflict-Averse Gradient Descent (CAGrad), all coordinated via FedAvg. FedAdapt demonstrates that the shared task adapter serves as an effective knowledge transfer anchor, enabling the arriving task to leverage cross-task representations learned from the pre-trained tasks.

The model jointly solves four perception tasks spanning camera and LiDAR modalities alongside an augmented text modality:

| Task | Modalities | Output | Classes | Metric |
|---|---|---|---|---|
| **3-D semantic segmentation** (`seg3d`) | Image + LiDAR + Text | `[B, 22, 224, 224]` | 22 | mIoU |
| **3-D object detection** (`det3d`) | Image + LiDAR + Text | cls `[B, 200, 5]` · box `[B, 200, 8]` | 4 | mAPH/L2 |
| **2-D panoptic segmentation** (`pan2d`) | Image + Text | `[B, 28, 224, 224]` | 28 | wSTQ |
| **2-D object detection** (`det2d`) | Image + Text | cls `[B, 100, 4]` · box `[B, 100, 4]` | 3 | mAP@0.5 |

---

## Architecture

```
                     ┌─────────────────────────────────────────────────────────┐
                     │                  CustomCLIP  (trainers/mmadapter.py)     │
                     │                                                           │
  Camera image ─────►│  CLIP ViT-B/16 image encoder  (frozen)                  │
  [B, 3, 224, 224]   │  + modality-specific visual adapters δv at layers 5–12  │──► patch tokens
                     │                                                           │    [B, 196, 512]
                     │                                                           │
  LiDAR range ───────►│  LiDAREncoder  (clip/lidar_encoder.py)                  │──► patch tokens
  [B, 3, 224, 224]   │  ├── LiDARInputStem  (conv3×3 + BN, near-identity init) │    [B, 196, 512]
  (null for 2-D tasks)│  └── modality-specific LiDAR adapters δℓ at layers 5–12│
                     │                                                           │
  Task text ─────────►│  CLIP text encoder  (frozen)                            │──► class embeddings
                     │  + modality-specific text adapters δt at layers 5–12    │    [57, 512]
                     │  (no feature caching while adapters are active)          │
                     └─────────────────────────────────────────────────────────┘
                                              │
                         image + LiDAR patch tokens fused [B, 196, 512]
                                              │
                                   Shared Task Adapter
                             (cross-task knowledge transfer)
                                              │
               ┌──────────────────────────────┼──────────────────────────────┐
               ▼                              ▼                              ▼                    ▼
         SegHead3D                       DetHead3D                     PanHead2D             DetHead2D
   bilinear upsample               DETR decoder (N=200)          bilinear upsample      DETR decoder (N=100)
   cosine-sim classifier            2 decoder layers              cosine-sim classifier   2 decoder layers
   [B, 22, 224, 224]                Hungarian matching            [B, 28, 224, 224]       Hungarian matching
   cross-entropy loss               focal + L1 loss               cross-entropy loss      focal + L1 loss
```

### Task onboarding protocol

For each reported result, one task is designated as the **arriving task** while the remaining three are **pre-trained tasks**. When the arriving task is onboarded:

- The task head for the arriving task is **initialised from scratch** and trained.
- The **shared task adapter** and all **modality-specific adapters** (δv, δℓ, δt) are fine-tuned.
- The **frozen CLIP backbone** and the **task heads of the three pre-trained tasks** remain unchanged throughout.

FedAvg is used for model aggregation across all methods.

### Trainable parameters

| Component | Trainable | Federated phase |
|---|---|---|
| CLIP image / LiDAR / text backbone | ✗ frozen | — |
| Modality-specific visual adapters δv (layers 5–12) | ✓ | Phase 1 & 2 |
| `LiDARInputStem` (conv3×3 + BN) | ✓ | Phase 1 only (frozen in phase 2) |
| Modality-specific LiDAR adapters δℓ (layers 5–12) | ✓ | Phase 1 & 2 |
| Modality-specific text adapters δt (layers 5–12) | ✓ | Phase 1 & 2 |
| Shared task adapter | ✓ | Phase 1 & 2 |
| Arriving task head | ✓ | Phase 1 & 2 |
| Pre-trained task heads | ✗ frozen | — |

---

## Repository structure

```
FedMMA-fork/
│
├── clip/
│   ├── model_mma.py          # Modified CLIP ViT-B/16: returns patch tokens [B,196,512];
│   │                         # encode_text accepts adapter_func for adapter injection
│   ├── lidar_encoder.py      # LiDAREncoder: LiDARInputStem + LiDAR adapters (layers 5–12)
│   └── clip.py / model.py    # Original CLIP code (minimally modified)
│
├── trainers/
│   ├── task_heads.py         # SegHead3D, DetHead3D, PanHead2D, DetHead2D + build_task_heads()
│   └── mmadapter.py          # CustomCLIP: 3-branch fusion, shared task adapter, null_lidar
│
├── datasets/
│   └── waymo.py              # WaymoMultiTaskDataset: .pt segment loader + collate
│
├── utils/
│   ├── text_templates.py     # TASK_CLASSES (57 total), TEXT_SLICES, TASK_MODALITIES
│   ├── fed_utils.py          # Metrics: mIoU, mAPH/L2, wSTQ, mAP@0.5; FedAvg aggregation
│   └── waymo_data_manager.py # 20-client segment split + per-task dataloader builder
│
├── scripts/
│   └── preprocess_waymo.py   # Waymo v2 GCS parquets → .pt segment files
│
├── configs/
│   ├── datasets/waymo.yaml               # Dataset config (20 clients, 4 tasks)
│   └── trainers/MultiModalAdapter/
│       └── waymo_multitask.yaml          # Training hyperparameters
│
├── federated_main.py         # Task onboarding federated training loop + baselines
└── cuda_waymo.def            # Apptainer container definition for HPC
```

---

## Dataset: Waymo Open Dataset v2

### Prerequisites

- Access to [Waymo Open Dataset v2](https://waymo.com/open/) via Google Cloud Storage
- Apptainer / Singularity container (see `cuda_waymo.def`) for HPC preprocessing
- ~2 TB storage for the full training split preprocessed

### Preprocessing segments into `.pt` files

The preprocessing script downloads Waymo v2 parquet components from GCS and converts each driving segment into a self-contained `.pt` file:

```bash
# Inside the Apptainer container on HPC:

# Discover actual GCS column names first (run once):
python3 scripts/preprocess_waymo.py --discover --split training

# Preprocess all segments (parallelised, 4 workers):
python3 scripts/preprocess_waymo.py \
    --split    training \
    --out_dir  /path/to/waymo_pt/training \
    --workers  4

# Smoke-test a single segment:
python3 scripts/preprocess_waymo.py \
    --split    training \
    --out_dir  /tmp/waymo_test \
    --max_segs 1
```

Each `.pt` file has the following structure:

```python
{
    "segment_id": str,
    "split":      str,          # "training" / "validation"
    "frames": [
        {
            "frame_id":    int,
            "image":       Tensor[3, 224, 224]  # float16, CLIP-normalised front camera
            "lidar":       Tensor[3, 224, 224]  # float16, [range, intensity, elongation]
            "seg3d_label": Tensor[224, 224]     # int16, 0–21 classes, 255=ignore (~15% of frames)
            "det3d_boxes": Tensor[N, 8]         # float32, (cx, cy, cz, sx, sy, sz, heading_rad, cls)
            "pan2d_label": Tensor[224, 224]     # int32, 0–27 classes, 255=ignore
            "det2d_boxes": Tensor[M, 5]         # float32, (cx_n, cy_n, w_n, h_n, cls) normalised
        },
        ...   # ~198 frames per segment at 10 Hz
    ]
}
```

> **Note on seg3d annotation cadence:** Waymo annotates LiDAR segmentation at ~1.5 Hz while capturing at 10 Hz, so only ~15% of frames (~30/198) carry `seg3d_label`. This is expected behaviour.

---

## Installation

```bash
git clone https://github.com/KasraBorazjani/vehicular-fedfm.git
cd vehicular-fedfm

# Core dependencies
pip install torch torchvision          # tested: torch 2.3+
pip install scipy ftfy regex tqdm Pillow

# Preprocessing only (not required for training on pre-built .pt files)
pip install waymo-open-dataset-tf-2-12-0 dask[dataframe] pyarrow
pip install google-cloud-storage

# Dassl (included; install in-place)
cd Dassl && pip install -e . && cd ..
```

---

## Training

### Configuration

Edit `configs/datasets/waymo.yaml` to set your data paths and `configs/trainers/MultiModalAdapter/waymo_multitask.yaml` for hyperparameters.

Key config options:

```yaml
# configs/datasets/waymo.yaml
DATASET:
  NAME: WaymoMultiTask
  ROOT: /path/to/waymo_pt
  NUM_CLIENTS: 20
  TASKS: [seg3d, det3d, pan2d, det2d]

# configs/trainers/MultiModalAdapter/waymo_multitask.yaml
TRAINER:
  PHASE1_ROUNDS: 100       # cross-modal alignment (stem + adapters aggregated)
  PHASE2_ROUNDS: 50        # task specialisation (stem frozen, adapters aggregated)
  LOCAL_EPOCHS: 5
  CLIENTS_PER_ROUND: 10
  LOSS_BALANCE: cagrad     # cagrad | gradnorm | uniform
MODEL:
  ADAPTER_DIM: 64
  ADAPTER_START: 5
  NUM_QUERIES_3D: 200
  NUM_QUERIES_2D: 100
  NUM_DECODER_LAYERS: 2
OPTIM:
  NAME: AdamW
  LR: 1.0e-4
  WEIGHT_DECAY: 0.01
```

### Run federated task onboarding

```bash
python federated_main.py \
    --config-file configs/trainers/MultiModalAdapter/waymo_multitask.yaml \
    --root        /path/to/waymo_pt \
    --arriving-task seg3d \
    --output-dir  output/fedadapt_seg3d
```

The `--arriving-task` flag designates which of the four tasks is treated as the arriving task. The remaining three task heads are loaded from a pre-trained checkpoint and frozen throughout.

### Baselines

All methods use FedAvg for model aggregation. Baselines are selectable via `--baseline`:

| Flag | Method | Shared task adapter | Gradient balancing |
|---|---|---|---|
| `nta` | **NTA** — modality-specific adapters only, no shared task adapter | ✗ | none |
| `nta_gr` | **NTA+GR** — NTA with GradNorm | ✗ | GradNorm |
| `nta_ca` | **NTA+CA** — NTA with CAGrad | ✗ | CAGrad |
| `fedadapt_no_ca` | **FedAdapt w/o CA** — shared task adapter, no gradient balancing | ✓ | none |
| `fedadapt_no_ca_gr` | **FedAdapt w/o CA + GR** — shared task adapter with GradNorm | ✓ | GradNorm |
| `fedadapt` | **FedAdapt** *(proposed)* — shared task adapter + CAGrad | ✓ | CAGrad |

```bash
python federated_main.py \
    --config-file configs/trainers/MultiModalAdapter/waymo_multitask.yaml \
    --root        /path/to/waymo_pt \
    --arriving-task det3d \
    --baseline    nta_ca \
    --output-dir  output/nta_ca_det3d
```

---

## Evaluation

Evaluation runs automatically at the end of each federated round and on a held-out validation split. Metrics per task:

| Task | Metric | Description |
|---|---|---|
| `seg3d` | **mIoU** | Mean intersection-over-union over 22 semantic classes |
| `det3d` | **mAPH/L2** | Waymo heading-aware mean average precision (L2 difficulty) |
| `pan2d` | **wSTQ** | Weighted Segmentation and Tracking Quality (Waymo panoptic metric) |
| `det2d` | **mAP@0.5** | Mean average precision at IoU threshold 0.5 |

To evaluate a saved checkpoint:

```bash
python federated_main.py \
    --config-file configs/trainers/MultiModalAdapter/waymo_multitask.yaml \
    --root        /path/to/waymo_pt \
    --arriving-task pan2d \
    --eval-only \
    --model-dir   output/fedadapt_pan2d/round_150
```

---

## Key design decisions

### Shared task adapter

The shared task adapter captures cross-task representations and acts as a knowledge anchor for arriving tasks. It contains information accumulated from pre-trained tasks, which is re-adapted with each task arrival and usable across all tasks. This design offloads task adaptation from the backbone (intended for multi-modal feature extraction) to a dedicated lightweight module.

### Why DETR-style detection heads?

Empirical analysis of the Waymo training split confirmed that a 14×14 patch grid is insufficient for direct grid-based detection (89.9% patch collision rate for 2-D boxes, ~99.9% for 3-D). Following [TransFusion (CVPR 2022)](https://arxiv.org/abs/2203.11932) and [SEED (ECCV 2024)](https://arxiv.org/abs/2407.10749), we use learned-query transformer decoders (2 layers) with Hungarian matching — a notable implementation improvement over the two-deconvolution-layer heads described in the paper.

### Why bilinear upsample for segmentation?

Patch-level classification evaluates at 14×14 resolution, where Waymo's ground-truth labels are defined at per-pixel level. Bilinear upsampling from `[B, 512, 14, 14]` to `[B, 512, 224, 224]` followed by cosine-similarity classification correctly aligns predictions with the label resolution.

### Why sin/cos heading encoding?

Raw heading regression has a ±π discontinuity. `DetHead3D` regresses `(sin θ, cos θ)` separately and reconstructs heading via `atan2(sin, cos)`. The `DetHead3D.encode_boxes()` static method converts raw `.pt` box tensors to this format before loss computation.

### LiDAR input stem

A lightweight `Conv2d(3→3, k=3, p=1) + BatchNorm` layer with near-identity initialisation maps LiDAR channel statistics (range, intensity, elongation) into a distribution compatible with CLIP's RGB-pretrained patch embedding. The stem is aggregated during phase 1 only and frozen at the phase 2 boundary via `encoder.freeze_stem()`.

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- scipy ≥ 1.10
- CLIP (included in `clip/`)
- Dassl (included as subdirectory)
- Waymo Open Dataset tools (preprocessing only)

---

## Acknowledgements

This project builds on:
- [FedPHA (ICML 2025)](https://openreview.net/forum?id=y7pDvbi9xz) — federated adapter baseline codebase
- [CLIP (OpenAI)](https://github.com/openai/CLIP) — vision-language backbone
- [Waymo Open Dataset v2](https://waymo.com/open/) — autonomous driving data
- [TransFusion (CVPR 2022)](https://arxiv.org/abs/2203.11932) · [SEED (ECCV 2024)](https://arxiv.org/abs/2407.10749) — detection head design reference
- [CAGrad](https://arxiv.org/abs/2110.14048) · [GradNorm](https://arxiv.org/abs/1711.02257) — multi-task gradient balancing

---

## Citation

> Paper under preparation. BibTeX entry will be added upon submission.

---

*Kasra Borazjani, Fardis Nadimi, Payam Abdisarabshali, Owen Palinski, Allan Salihovic,
Dinh Nguyen, Minghui Liwang, and Seyyedali Hosseinalipour — University at Buffalo*
