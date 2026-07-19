#!/usr/bin/env python3
"""
scripts/smoke_test_data_manager.py
------------------------------------
Smoke test for utils/waymo_data_manager.py.

Creates minimal mock .pt files (two frames each, all four task annotations)
then exercises WaymoDataManager without any real Waymo data.

Run from repo root:
    python scripts/smoke_test_data_manager.py
"""

import sys, os, tempfile, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path

print("=" * 60)
print("WaymoDataManager smoke test")
print("=" * 60)


# ── Build mock .pt segments ───────────────────────────────────────────────────

def _make_frame():
    """Synthetic single-frame dict matching .pt schema."""
    N = 5   # LiDAR boxes
    M = 3   # 2D boxes
    return {
        "image":         torch.zeros(3, 224, 224),
        "lidar":         torch.zeros(3, 224, 224),
        "seg3d_label":   torch.randint(0, 22, (224, 224), dtype=torch.int16),
        "det3d_boxes":   torch.rand(N, 8),
        "pan2d_label":   torch.randint(0, 28, (224, 224), dtype=torch.int32),
        "pan2d_instance":torch.randint(0, 5,  (224, 224), dtype=torch.int32),
        "det2d_boxes":   torch.rand(M, 5),
    }


def make_mock_pt_dir(n_files: int = 6, frames_per_file: int = 10) -> str:
    tmp = tempfile.mkdtemp(prefix="fedmma_smoke_")
    for i in range(n_files):
        frames = [_make_frame() for _ in range(frames_per_file)]
        torch.save({"frames": frames}, os.path.join(tmp, f"seg_{i:04d}.pt"))
    return tmp


print("\n[1] Building 6 mock .pt files (10 frames each) ...")
pt_dir = make_mock_pt_dir(n_files=6, frames_per_file=10)
print(f"    Written to {pt_dir}")


# ── WaymoDataManager construction ─────────────────────────────────────────────
print("\n[2] Constructing WaymoDataManager (4 clients, seg3d task) ...")
from utils.waymo_data_manager import WaymoDataManager, TASKS

mgr = WaymoDataManager(
    pt_dir        = pt_dir,
    num_clients   = 4,
    arriving_task = "seg3d",
    batch_size    = 2,
    pan_seq_len   = 5,
    num_workers   = 0,   # no multi-process in smoke test
    seed          = 42,
    eval_frac     = 0.2,
)
assert mgr.num_clients == 4
print("    OK")


# ── Check segment assignment ───────────────────────────────────────────────────
print("\n[3] Segment assignment (non-overlapping) ...")
all_assigned = []
for c in range(mgr.num_clients):
    segs = mgr.client_segments[c]
    all_assigned.extend(segs)
    print(f"    client {c}: {len(segs)} segments")
# All assigned segments are unique
assert len(all_assigned) == len(set(all_assigned)), "Segment overlap detected!"
# Eval segments not in any client
for s in mgr._eval_segments:
    assert s not in all_assigned, f"Eval segment leaked into client: {s}"
print("    non-overlapping & eval excluded  OK")


# ── dataset_sizes ─────────────────────────────────────────────────────────────
print("\n[4] dataset_sizes() ...")
for c in range(mgr.num_clients):
    sizes = mgr.dataset_sizes(c)
    for t in TASKS:
        assert t in sizes
        assert sizes[t] >= 0
print(f"    client 0 sizes: { {t: mgr.dataset_sizes(0)[t] for t in TASKS} }  OK")


# ── get_loader: seg3d ─────────────────────────────────────────────────────────
print("\n[5] get_loader (seg3d) ...")
loader = mgr.get_loader(0, "seg3d")
batch  = next(iter(loader))
assert batch["task"] == "seg3d"
assert batch["image"].shape == (2, 3, 224, 224)
assert batch["lidar"].shape == (2, 3, 224, 224)
assert batch["targets"].shape == (2, 224, 224)
print(f"    seg3d batch image {tuple(batch['image'].shape)}  targets {tuple(batch['targets'].shape)}  OK")


# ── get_loader: det3d ─────────────────────────────────────────────────────────
print("\n[6] get_loader (det3d) ...")
loader3d = mgr.get_loader(0, "det3d")
b3d      = next(iter(loader3d))
assert b3d["task"] == "det3d"
assert isinstance(b3d["targets"], list)
print(f"    det3d batch image {tuple(b3d['image'].shape)}  targets list[{len(b3d['targets'])}]  OK")


# ── get_loader: pan2d ─────────────────────────────────────────────────────────
print("\n[7] get_loader (pan2d) ...")
# pan2d batch_size = batch_size // pan_seq_len = 2 // 5 = 1 (minimum 1)
loader_pan = mgr.get_loader(0, "pan2d")
T = mgr.pan_seq_len
b_pan = next(iter(loader_pan))
assert b_pan["task"] == "pan2d"
assert b_pan["image"].shape[1] == T, f"expected T={T} got {b_pan['image'].shape}"
assert b_pan["targets"]["semantic"].shape[1] == T
assert b_pan["lidar"] is None
print(f"    pan2d batch image {tuple(b_pan['image'].shape)}  sem {tuple(b_pan['targets']['semantic'].shape)}  OK")


# ── get_loader: det2d ─────────────────────────────────────────────────────────
print("\n[8] get_loader (det2d) ...")
loader2d = mgr.get_loader(0, "det2d")
b2d      = next(iter(loader2d))
assert b2d["task"] == "det2d"
assert b2d["lidar"] is None
print(f"    det2d batch image {tuple(b2d['image'].shape)}  OK")


# ── get_eval_loader ───────────────────────────────────────────────────────────
print("\n[9] get_eval_loader (seg3d) ...")
eval_loader = mgr.get_eval_loader("seg3d")
be = next(iter(eval_loader))
assert be["task"] == "seg3d"
print(f"    eval seg3d batch image {tuple(be['image'].shape)}  OK")


# ── get_cycle_loader ──────────────────────────────────────────────────────────
print("\n[10] get_cycle_loader (infinite iterator) ...")
cycle_it = mgr.get_cycle_loader(0, "seg3d")
batches  = [next(cycle_it) for _ in range(5)]
assert all(b["task"] == "seg3d" for b in batches)
print(f"    5 consecutive batches from cycle_loader  OK")


# ── arriving_task_sizes ───────────────────────────────────────────────────────
print("\n[11] arriving_task_sizes() ...")
sizes = mgr.arriving_task_sizes()
assert len(sizes) == mgr.num_clients
assert all(s >= 0 for s in sizes)
print(f"    arriving_task_sizes = {sizes}  OK")


# ── Cleanup ───────────────────────────────────────────────────────────────────
import shutil
shutil.rmtree(pt_dir, ignore_errors=True)

print("\n" + "=" * 60)
print("All WaymoDataManager smoke tests passed.")
print("=" * 60)
