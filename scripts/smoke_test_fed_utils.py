#!/usr/bin/env python3
"""
scripts/smoke_test_fed_utils.py
--------------------------------
Smoke tests for utils/fed_utils.py additions:
  - fedavg_aggregate / fedmma_aggregate
  - compute_miou
  - compute_map50
  - cagrad
  - GradNorm

No Waymo data or CLIP download required.

Run from repo root:
    python scripts/smoke_test_fed_utils.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from utils.fed_utils import (
    fedavg_aggregate,
    fedmma_aggregate,
    compute_miou,
    compute_map50,
    cagrad,
    GradNorm,
)

print("=" * 60)
print("fed_utils smoke test")
print("=" * 60)


# ── 1. fedavg_aggregate ───────────────────────────────────────────────────────
print("\n[1] fedavg_aggregate ...")

sd_a = {"w": torch.ones(4) * 2.0, "b": torch.zeros(4)}
sd_b = {"w": torch.ones(4) * 4.0, "b": torch.zeros(4)}
# Uniform weights → average should be 3.0
agg  = fedavg_aggregate([sd_a, sd_b])
assert torch.allclose(agg["w"], torch.ones(4) * 3.0), agg["w"]

# Data-weighted: weight_a=1, weight_b=3 → weighted avg = 3.5
agg2 = fedavg_aggregate([sd_a, sd_b], weights=[1.0, 3.0])
assert torch.allclose(agg2["w"], torch.ones(4) * 3.5), agg2["w"]

# Subset of keys
agg3 = fedavg_aggregate([sd_a, sd_b], keys=["w"])
assert "b" not in agg3
print("    fedavg_aggregate: OK")


# ── 2. fedmma_aggregate ───────────────────────────────────────────────────────
print("\n[2] fedmma_aggregate ...")

sd_c = {"vis_adapters.0": torch.ones(3), "task_heads.seg3d.fc": torch.zeros(3)}
sd_d = {"vis_adapters.0": torch.ones(3) * 3.0, "task_heads.seg3d.fc": torch.ones(3) * 99.0}
g_keys = ["vis_adapters.0"]
agg4 = fedmma_aggregate([sd_c, sd_d], global_keys=g_keys)
assert "vis_adapters.0" in agg4
assert "task_heads.seg3d.fc" not in agg4   # local key not aggregated
assert torch.allclose(agg4["vis_adapters.0"], torch.ones(3) * 2.0)
print("    fedmma_aggregate: OK")


# ── 3. compute_miou ───────────────────────────────────────────────────────────
print("\n[3] compute_miou ...")

# Perfect prediction → mIoU = 1.0
N, H, W, C = 2, 8, 8, 4
labels = torch.randint(0, C, (N, H, W))
preds  = labels.clone()       # perfect match
out    = compute_miou(preds, labels, num_classes=C)
assert abs(out["miou"] - 1.0) < 1e-6, out["miou"]

# Test with ignore_index
labels_ign = labels.clone()
labels_ign[:, 0, 0] = 255     # mark one pixel as ignore
out_ign = compute_miou(preds, labels_ign, num_classes=C)
assert out_ign["miou"] <= 1.0

# Test with logits input [N, C, H, W]
logits = torch.zeros(N, C, H, W)
for c in range(C):
    logits[:, c, :, :] = (labels == c).float() * 100.0
out_logits = compute_miou(logits, labels, num_classes=C)
assert abs(out_logits["miou"] - 1.0) < 1e-6
print(f"    compute_miou: perfect={out['miou']:.4f}  with_ignore={out_ign['miou']:.4f}  OK")


# ── 4. compute_map50 ─────────────────────────────────────────────────────────
print("\n[4] compute_map50 ...")

# Perfect detection: pred boxes == gt boxes, score = 1.0
num_cls = 3
gt_box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])   # cxcywh
gt_lbl = torch.tensor([0])
pred_boxes_list  = [gt_box]
pred_scores_list = [torch.tensor([1.0])]
pred_labels_list = [gt_lbl]
gt_boxes_list    = [gt_box]
gt_labels_list   = [gt_lbl]

out = compute_map50(
    pred_boxes_list, pred_scores_list, pred_labels_list,
    gt_boxes_list,   gt_labels_list, num_classes=num_cls,
)
# Class 0 should have AP = 1.0 (or close); classes 1,2 → 0.0
assert out["map50"] >= 0.0, out
assert len(out["per_class_ap"]) == num_cls
assert out["per_class_ap"][0] > 0.8, out["per_class_ap"]
print(f"    compute_map50: map50={out['map50']:.4f}  per_class={[f'{x:.2f}' for x in out['per_class_ap']]}  OK")


# ── 5. cagrad ────────────────────────────────────────────────────────────────
print("\n[5] cagrad ...")

D = 64
T = 3
grads = torch.randn(T, D)
g_out = cagrad(grads, alpha=0.5)
assert g_out.shape == (D,), g_out.shape
assert not g_out.isnan().any(), "NaN in cagrad output"

# Single task → should return g_mean unchanged
g_single = cagrad(grads[:1], alpha=0.5)
assert torch.allclose(g_single, grads[0]), "cagrad T=1 should return g_mean"

# alpha=0 → pure g_mean
g_alpha0 = cagrad(grads, alpha=0.0)
assert torch.allclose(g_alpha0, grads.mean(0))
print(f"    cagrad: shape {tuple(g_out.shape)}  no NaN  OK")


# ── 6. GradNorm ──────────────────────────────────────────────────────────────
print("\n[6] GradNorm ...")

gn = GradNorm(num_tasks=4, alpha=1.5, lr=1e-3)
task_losses = torch.tensor([1.0, 0.5, 2.0, 0.8])
total, w = gn.weighted_loss(task_losses)
assert total.shape == (), f"expected scalar, got {total.shape}"
assert w.shape == (4,), w.shape
assert not w.isnan().any()

# Simulate a shared parameter
shared_p = torch.randn(16, 16, requires_grad=True)
(shared_p.sum() * task_losses.sum()).backward()
gn.step(shared_p, task_losses.detach())
# Weights should still be valid after update
total2, w2 = gn.weighted_loss(task_losses)
assert not w2.isnan().any()
print(f"    GradNorm: total={total.item():.4f}  weights={w.detach().tolist()[:2]}...  OK")


print("\n" + "=" * 60)
print("All fed_utils smoke tests passed.")
print("=" * 60)
