"""
datasets/waymo.py
-----------------
WaymoMultiTaskDataset: scans preprocessed .pt segment files once at init to
build per-task indices, then loads frame data lazily on each __getitem__.

Segment assignment (mutual exclusion across clients) is handled upstream by
waymo_data_manager.py, which calls WaymoMultiTaskDataset with each client's
non-overlapping list of segment paths.

Usage
-----
  from datasets.waymo import WaymoMultiTaskDataset, waymo_collate_fn

  ds     = WaymoMultiTaskDataset(segment_paths)
  loader = DataLoader(
      ds.task_dataset("seg3d"),
      batch_size=8, shuffle=True, collate_fn=waymo_collate_fn,
  )

.pt format per frame (after preprocess_waymo.py)
-------------------------------------------------
One .pt file per (segment, camera) — see preprocess_waymo.py output layout.
  seg3d_label    : Tensor[224, 224]  int16  | None  (0-21, 255=ignore)
  det3d_boxes    : Tensor[N, 8]      float32        (always present, N may be 0;
                                                      already filtered to this camera's FOV)
  pan2d_label    : Tensor[224, 224]  int32  | None  (0-27, 255=ignore)
  pan2d_instance : Tensor[224, 224]  int32  | None  (0=stuff, >0=trackable thing)
  det2d_boxes    : Tensor[M, 5]      float32 | None (cx_n,cy_n,w_n,h_n,cls)
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


# ── Dataset ───────────────────────────────────────────────────────────────────

class WaymoMultiTaskDataset:
    """
    Builds per-task frame indices from a list of segment .pt paths.
    Use .task_dataset(task) to obtain a task-specific Dataset for DataLoader.

    Args:
        segment_paths : absolute paths to preprocessed .pt segment files.
        pan_seq_len   : number of consecutive pan2d-labeled frames per sequence.
    """

    def __init__(self, segment_paths: List[str], pan_seq_len: int = 5):
        self.pan_seq_len = pan_seq_len
        self._build_indices(segment_paths)

    def _build_indices(self, segment_paths: List[str]):
        self.seg3d_index: List[Tuple[str, int]]       = []
        self.det3d_index: List[Tuple[str, int]]       = []
        self.det2d_index: List[Tuple[str, int]]       = []
        self.pan2d_index: List[Tuple[str, List[int]]] = []

        T = self.pan_seq_len

        for pt_path in segment_paths:
            data   = torch.load(pt_path, map_location="cpu", weights_only=False)
            frames = data["frames"]

            pan2d_labeled: List[int] = []

            for i, frame in enumerate(frames):
                if frame.get("seg3d_label") is not None:
                    self.seg3d_index.append((pt_path, i))

                # det3d_boxes is always a tensor (possibly [0, 8]); include all frames
                self.det3d_index.append((pt_path, i))

                if frame.get("pan2d_label") is not None:
                    pan2d_labeled.append(i)

                if frame.get("det2d_boxes") is not None:
                    self.det2d_index.append((pt_path, i))

            # Non-overlapping windows of T consecutive pan2d-labeled frames
            for start in range(0, len(pan2d_labeled) - T + 1, T):
                self.pan2d_index.append((pt_path, pan2d_labeled[start : start + T]))

            del data

    def task_dataset(self, task: str) -> "WaymoTaskDataset":
        return WaymoTaskDataset(self, task)


class WaymoTaskDataset(Dataset):
    """
    Single-task Dataset view of WaymoMultiTaskDataset.
    All frame data is loaded on demand in __getitem__; nothing is cached.
    """

    def __init__(self, multi_ds: WaymoMultiTaskDataset, task: str):
        self.task  = task
        self.index = getattr(multi_ds, f"{task}_index")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        if self.task == "pan2d":
            return self._getitem_pan2d(idx)
        return self._getitem_single(idx)

    # ── Single-frame tasks (seg3d, det3d, det2d) ──────────────────────────────

    def _getitem_single(self, idx: int) -> Dict:
        task             = self.task
        pt_path, frame_i = self.index[idx]
        data             = torch.load(pt_path, map_location="cpu", weights_only=False)
        frame            = data["frames"][frame_i]

        image = frame["image"].float()
        lidar = frame["lidar"].float() if task in ("seg3d", "det3d") else None

        if task == "seg3d":
            targets = frame["seg3d_label"].long()   # [224, 224]

        elif task == "det3d":
            raw = frame["det3d_boxes"]                # [N, 8] float32
            if raw.shape[0] == 0:
                targets = {
                    "labels": torch.zeros(0, dtype=torch.long),
                    "boxes":  torch.zeros(0, 8),
                }
            else:
                targets = {
                    "labels": raw[:, 7].long(),
                    "boxes":  raw,   # raw (cx,cy,cz,sx,sy,sz,heading,cls) for GT heatmap
                }

        else:   # det2d
            raw = frame.get("det2d_boxes")
            if raw is None or raw.shape[0] == 0:
                targets = {
                    "labels": torch.zeros(0, dtype=torch.long),
                    "boxes":  torch.zeros(0, 4),
                }
            else:
                targets = {
                    "labels": raw[:, 4].long(),
                    "boxes":  raw[:, :4],
                }

        return {
            "task":    task,
            "image":   image,
            "lidar":   lidar,
            "targets": targets,
        }

    # ── Pan2d: T-frame sequence ───────────────────────────────────────────────

    def _getitem_pan2d(self, idx: int) -> Dict:
        pt_path, frame_indices = self.index[idx]
        data   = torch.load(pt_path, map_location="cpu", weights_only=False)
        frames = data["frames"]

        images   = torch.stack([frames[i]["image"].float() for i in frame_indices])
        semantic = torch.stack([frames[i]["pan2d_label"].long() for i in frame_indices])

        # pan2d_instance may be absent in .pt files pre-dating the preprocessing fix
        inst_parts = []
        for i in frame_indices:
            inst = frames[i].get("pan2d_instance")
            if inst is None:
                inst = torch.zeros_like(frames[i]["pan2d_label"])
            inst_parts.append(inst.long())
        instance = torch.stack(inst_parts)

        return {
            "task":    "pan2d",
            "image":   images,
            "lidar":   None,
            "targets": {"semantic": semantic, "instance": instance},
        }


# ── Collate function ──────────────────────────────────────────────────────────

def waymo_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate a list of per-item dicts into a batched dict.

    Target formats:
      seg3d  → targets stacked to [B, H, W]
      pan2d  → targets dict with "semantic"/"instance" each [B, T, H, W]
      det*   → targets kept as list of per-image dicts (variable M)
    """
    task   = batch[0]["task"]
    images = torch.stack([b["image"] for b in batch])

    lidars: Optional[torch.Tensor] = None
    if batch[0]["lidar"] is not None:
        lidars = torch.stack([b["lidar"] for b in batch])

    if task == "seg3d":
        targets = torch.stack([b["targets"] for b in batch])
    elif task == "pan2d":
        targets = {
            "semantic": torch.stack([b["targets"]["semantic"] for b in batch]),
            "instance": torch.stack([b["targets"]["instance"] for b in batch]),
        }
    else:   # det3d, det2d
        targets = [b["targets"] for b in batch]

    return {
        "task":    task,
        "image":   images,
        "lidar":   lidars,
        "targets": targets,
    }
