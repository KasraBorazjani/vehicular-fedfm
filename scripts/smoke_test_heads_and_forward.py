#!/usr/bin/env python3
"""
scripts/smoke_test_heads_and_forward.py
----------------------------------------
Synthetic end-to-end smoke test: all four task heads (forward + loss)
and CustomCLIP forward for all four tasks using a mock CLIP backbone.

No Waymo data, no CLIP download required.

Run from the repo root:
    python scripts/smoke_test_heads_and_forward.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, T   = 2, 5
N196   = 196
D      = 512

print("=" * 60)
print(f"FedMMA heads + forward smoke test  [{DEVICE}]")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Individual head forward + loss
# ─────────────────────────────────────────────────────────────────────────────

from models.task_heads import SegHead3D, DetHead3D, PanHead2D, DetHead2D

# ── SegHead3D ────────────────────────────────────────────────────────────────
print("\n[1/4] SegHead3D (fused tokens → upsample → cosine-sim, WCE+Lovász) ...")
head          = SegHead3D().to(DEVICE)
fused_patches = torch.randn(B, N196, D, device=DEVICE)   # pre-fused by ImageLidarFusion
tembs         = torch.randn(22, D, device=DEVICE)
logits        = head(fused_patches, tembs)
assert logits.shape == (B, 22, 224, 224), logits.shape
labels        = torch.randint(0, 22, (B, 224, 224), device=DEVICE)
losses        = head.compute_loss(logits, labels)
assert not losses["loss"].isnan(),     "NaN total loss"
assert not losses["loss_wce"].isnan(), "NaN WCE"
assert not losses["loss_lovasz"].isnan(), "NaN Lovász"
print(f"    logits {tuple(logits.shape)}  "
      f"loss={losses['loss'].item():.4f}  "
      f"wce={losses['loss_wce'].item():.4f}  "
      f"lov={losses['loss_lovasz'].item():.4f}  OK")

# ── DetHead3D (DETR-style) ────────────────────────────────────────────────────
print("\n[2/4] DetHead3D (DETR-style, fused tokens → decoder → cls+box) ...")
Q    = 10   # small for speed
head = DetHead3D(num_decoder_layers=1, num_queries=Q).to(DEVICE)
fused_patches = torch.randn(B, N196, D, device=DEVICE)
tembs         = torch.randn(4, D, device=DEVICE)

cls_logits, box_preds = head(fused_patches, tembs)
assert cls_logits.shape == (B, Q, 5), cls_logits.shape   # 4 cls + 1 no-obj
assert box_preds.shape  == (B, Q, 7), box_preds.shape    # cx,cy,cz,sx,sy,sz,heading

# Build det3d targets: boxes [M, 8] where col 7 = class idx
targets = []
for _ in range(B):
    raw = torch.zeros(3, 8)
    raw[:, 0] = torch.tensor([-20.0, 0.0, 30.0])   # cx
    raw[:, 1] = torch.tensor([10.0, -5.0, 50.0])    # cy
    raw[:, 2] = 0.5                                  # cz
    raw[:, 3:6] = torch.tensor([[4.0, 2.0, 1.5]])   # sx, sy, sz
    raw[:, 6] = 0.3                                  # heading
    raw[:, 7] = torch.tensor([0, 1, 3]).float()      # cls
    targets.append({
        "labels": raw[:, 7].long(),
        "boxes":  raw,
    })
losses = head.compute_loss((cls_logits, box_preds), targets)
assert not losses["loss"].isnan(),     "NaN loss"
assert not losses["loss_cls"].isnan(), "NaN loss_cls"
assert not losses["loss_box"].isnan(), "NaN loss_box"
print(f"    cls {tuple(cls_logits.shape)}  box {tuple(box_preds.shape)}  "
      f"loss={losses['loss'].item():.4f}  "
      f"cls={losses['loss_cls'].item():.4f}  "
      f"box={losses['loss_box'].item():.4f}  OK")

# Test with empty targets (no GT boxes in frame)
empty_targets = [{"labels": torch.zeros(0, dtype=torch.long),
                  "boxes":  torch.zeros(0, 8)} for _ in range(B)]
losses_empty = head.compute_loss((cls_logits, box_preds), empty_targets)
assert not losses_empty["loss"].isnan(), "NaN loss on empty targets"
print(f"    empty targets: loss={losses_empty['loss'].item():.4f}  OK")

# ── PanHead2D (FPN Panoptic DeepLab) ─────────────────────────────────────────
print("\n[3/4] PanHead2D (FPN, semantic+center+offset+contrastive) ...")
head       = PanHead2D().to(DEVICE)
seq_tokens = torch.randn(B, T, N196, D, device=DEVICE)
tembs      = torch.randn(28, D, device=DEVICE)

sem_l, ctr_l, off_p, embs = head(seq_tokens, tembs)
from models.task_heads import PanHead2D as _P
hw = _P.FPN_OUT_SIZE
assert sem_l.shape == (B, T, 28, hw, hw),        sem_l.shape
assert ctr_l.shape == (B, T, 1,  hw, hw),        ctr_l.shape
assert off_p.shape == (B, T, 2,  hw, hw),        off_p.shape
assert embs.shape  == (B, T, _P.EMB_DIM, hw, hw), embs.shape

# GT: 2 instances per frame in distinct regions
sem_gt  = torch.full((B, T, 224, 224), 255, dtype=torch.long, device=DEVICE)
inst_gt = torch.zeros((B, T, 224, 224), dtype=torch.long, device=DEVICE)
for b in range(B):
    for iid in [1, 2]:
        r0 = (iid - 1) * 80
        sem_gt[b, :, r0:r0+60, :60]  = iid % 28
        inst_gt[b, :, r0:r0+60, :60] = iid

targets = {"semantic": sem_gt, "instance": inst_gt}
losses  = head.compute_loss((sem_l, ctr_l, off_p, embs), targets)
assert not losses["loss"].isnan(), "NaN total loss"
for k in ("loss_sem", "loss_center", "loss_offset", "loss_contrast"):
    assert not losses[k].isnan(), f"NaN {k}"
print(f"    sem {tuple(sem_l.shape)}  center {tuple(ctr_l.shape)}  "
      f"loss={losses['loss'].item():.4f}  "
      f"sem={losses['loss_sem'].item():.4f}  "
      f"ctr={losses['loss_center'].item():.4f}  "
      f"off={losses['loss_offset'].item():.4f}  "
      f"con={losses['loss_contrast'].item():.4f}  OK")

# ── DetHead2D (DETR, scalar no-object, GIoU) ─────────────────────────────────
print("\n[4/4] DetHead2D (DETR, scalar no-obj bias, GIoU+focal+L1) ...")
head   = DetHead2D(num_decoder_layers=2).to(DEVICE)
tokens = torch.randn(B, N196, D, device=DEVICE)
tembs  = torch.randn(3, D, device=DEVICE)
cls_l, box_p = head(tokens, tembs)
assert cls_l.shape == (B, 100, 4)   # 3 cls + 1 no-obj scalar
assert box_p.shape == (B, 100, 4)
targets = [{"labels": torch.randint(0, 3, (4,)), "boxes": torch.rand(4, 4)}
           for _ in range(B)]
losses  = head.compute_loss(cls_l, box_p, targets)
assert not losses["loss"].isnan(), "NaN loss"
for k in ("loss_cls", "loss_box", "loss_giou"):
    assert not losses[k].isnan(), f"NaN {k}"
print(f"    cls {tuple(cls_l.shape)}  box {tuple(box_p.shape)}  "
      f"loss={losses['loss'].item():.4f}  "
      f"cls={losses['loss_cls'].item():.4f}  "
      f"giou={losses['loss_giou'].item():.4f}  OK")

# ─────────────────────────────────────────────────────────────────────────────
# 2. ImageLidarFusion standalone
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5] ImageLidarFusion (concat-project) ...")
from models.fusion import ImageLidarFusion
fusion        = ImageLidarFusion().to(DEVICE)
img_patches   = torch.randn(B, N196, D, device=DEVICE)
lidar_patches = torch.randn(B, N196, D, device=DEVICE)
fused         = fusion(img_patches, lidar_patches)
assert fused.shape == (B, N196, D), fused.shape
print(f"    fused {tuple(fused.shape)}  OK")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CustomCLIP with mock CLIP backbone
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6] CustomCLIP forward (mock backbone) ...")

from models.custom_clip import CustomCLIP


class _MockVisual(nn.Module):
    def forward(self, args):
        x, _ = args
        return torch.randn(x.shape[0], 196, 512, device=x.device, dtype=x.dtype)


class _MockTransformer(nn.Module):
    def forward(self, args):
        x = args[0] if isinstance(args, (list, tuple)) else args
        return x


class _MockClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual               = _MockVisual()
        self.dtype                = torch.float32
        self.transformer          = _MockTransformer()
        self.positional_embedding = nn.Parameter(torch.randn(77, 768))
        self.ln_final             = nn.LayerNorm(768)
        self.text_projection      = nn.Parameter(torch.randn(768, 512))
        self.token_embedding      = nn.Embedding(49408, 768)

    def requires_grad_(self, flag=True):
        for p in self.parameters():
            p.requires_grad_(flag)
        return self


model = CustomCLIP(
    clip_model         = _MockClip(),
    arriving_task      = "pan2d",
    adapter_dim        = 32,
    shared_adapter_dim = 32,
    adapter_start      = 5,
    num_decoder_layers = 1,
    num_queries_3d     = 10,
    num_queries_2d     = 10,
    num_queries_pan    = 10,
).to(DEVICE)

for task, image, lidar in [
    ("seg3d", torch.randn(B, 3, 224, 224, device=DEVICE),
              torch.randn(B, 3, 224, 224, device=DEVICE)),
    ("det3d", torch.randn(B, 3, 224, 224, device=DEVICE),
              torch.randn(B, 3, 224, 224, device=DEVICE)),
    ("pan2d", torch.randn(B, T, 3, 224, 224, device=DEVICE), None),
    ("det2d", torch.randn(B, 3, 224, 224, device=DEVICE),    None),
]:
    out = model(task, image, lidar)
    if isinstance(out, tuple):
        shapes = " | ".join(str(tuple(o.shape)) for o in out)
    else:
        shapes = str(tuple(out.shape))
    print(f"    {task}: {shapes}")

# Verify det3d output shapes (DETR-style)
det3d_out = model("det3d",
                  torch.randn(B, 3, 224, 224, device=DEVICE),
                  torch.randn(B, 3, 224, 224, device=DEVICE))
cls_l, box_p = det3d_out
assert cls_l.shape == (B, 10, 5), cls_l.shape   # 10 queries, 4 cls + no-obj
assert box_p.shape == (B, 10, 7), box_p.shape   # 7D box
print(f"    det3d shapes verified: cls {tuple(cls_l.shape)}  box {tuple(box_p.shape)}  OK")

# ─────────────────────────────────────────────────────────────────────────────
# 4. waymo_collate_fn
# ─────────────────────────────────────────────────────────────────────────────

print("\n[7] waymo_collate_fn ...")
from datasets.waymo import waymo_collate_fn

# pan2d
pan_batch = [
    {
        "task": "pan2d", "image": torch.randn(T, 3, 224, 224), "lidar": None,
        "targets": {
            "semantic": torch.randint(0, 28, (T, 224, 224)),
            "instance": torch.randint(0, 10, (T, 224, 224)),
        },
    }
    for _ in range(B)
]
out = waymo_collate_fn(pan_batch)
assert out["image"].shape               == (B, T, 3, 224, 224)
assert out["targets"]["semantic"].shape == (B, T, 224, 224)
assert out["lidar"] is None
print(f"    pan2d: image {tuple(out['image'].shape)}  OK")

# seg3d
seg_batch = [
    {
        "task": "seg3d", "image": torch.randn(3, 224, 224),
        "lidar": torch.randn(3, 224, 224),
        "targets": torch.randint(0, 22, (224, 224)),
    }
    for _ in range(B)
]
out = waymo_collate_fn(seg_batch)
assert out["image"].shape   == (B, 3, 224, 224)
assert out["targets"].shape == (B, 224, 224)
print(f"    seg3d: image {tuple(out['image'].shape)}  OK")

# det3d
det3d_batch = [
    {
        "task": "det3d", "image": torch.randn(3, 224, 224),
        "lidar": torch.randn(3, 224, 224),
        "targets": {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 8)},
    }
    for _ in range(B)
]
out = waymo_collate_fn(det3d_batch)
assert out["image"].shape == (B, 3, 224, 224)
print(f"    det3d: image {tuple(out['image'].shape)}  OK")

print("\n" + "=" * 60)
print("All smoke tests passed.")
print("=" * 60)
