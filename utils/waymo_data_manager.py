"""
utils/waymo_data_manager.py
----------------------------
Splits preprocessed .pt segment files across N clients and builds per-task
DataLoaders for each client.

Design
------
  - All .pt files in pt_dir are sorted → deterministically shuffled with seed
    → round-robin partitioned into num_clients non-overlapping slices.
  - DataLoaders are built lazily (on first call to get_loader / get_loaders).
  - pan2d: effective batch size = batch_size // pan_seq_len (minimum 1) so that
    each physical batch contains `batch_size` total frames.
  - An optional eval split (eval_frac of total segments) is held out before
    client assignment; get_eval_loader() returns a single loader over those.

Usage
-----
  from utils.waymo_data_manager import WaymoDataManager

  mgr = WaymoDataManager(
      pt_dir        = "/path/to/waymo_pt",
      num_clients   = 20,
      arriving_task = "seg3d",
      batch_size    = 8,
      pan_seq_len   = 5,
      seed          = 1,
  )

  # Per-client loaders (arriving_task only, by default)
  loader = mgr.get_loader(client_id=0, task="seg3d")

  # Or all tasks at once
  loaders = mgr.get_loaders(client_id=0)

  # Held-out eval loader
  eval_loader = mgr.get_eval_loader(task="seg3d")

  # Dataset sizes for FedAvg weighting
  sizes = mgr.dataset_sizes(client_id=0)
"""

import itertools
import os
import random
from typing import Dict, Iterator, List, Optional

from torch.utils.data import DataLoader

from datasets.waymo import WaymoMultiTaskDataset, waymo_collate_fn

TASKS = ["seg3d", "det3d", "pan2d", "det2d"]


class WaymoDataManager:
    """
    Manages per-client segment splits and on-demand DataLoader construction.

    Args
    ----
    pt_dir        : directory containing *.pt segment files (preprocess_waymo.py output).
    num_clients   : number of federated clients.
    arriving_task : task being onboarded; used as default when task is unspecified.
    batch_size    : physical batch size for single-frame tasks.
    pan_seq_len   : T for pan2d; actual sequences per batch = batch_size // pan_seq_len.
    num_workers   : DataLoader worker count.
    seed          : RNG seed for client shuffle.
    eval_frac     : fraction of segments reserved as held-out eval (not client-assigned).
    """

    def __init__(
        self,
        pt_dir:        str,
        num_clients:   int   = 20,
        arriving_task: str   = "seg3d",
        batch_size:    int   = 8,
        pan_seq_len:   int   = 5,
        num_workers:   int   = 4,
        seed:          int   = 1,
        eval_frac:     float = 0.1,
    ):
        self.arriving_task = arriving_task
        self.batch_size    = batch_size
        self.pan_seq_len   = pan_seq_len
        self.num_workers   = num_workers
        self.num_clients   = num_clients

        # ── Gather + shuffle .pt files ─────────────────────────────────────────
        pt_files = sorted(
            os.path.join(pt_dir, f)
            for f in os.listdir(pt_dir)
            if f.endswith(".pt")
        )
        if not pt_files:
            raise ValueError(f"No .pt files found in {pt_dir!r}")

        rng = random.Random(seed)
        rng.shuffle(pt_files)

        # ── Eval split ─────────────────────────────────────────────────────────
        n_eval = max(1, int(len(pt_files) * eval_frac))
        self._eval_segments: List[str]        = pt_files[:n_eval]
        train_segs:          List[str]        = pt_files[n_eval:]

        # ── Round-robin partition into client slices ───────────────────────────
        self.client_segments: List[List[str]] = [[] for _ in range(num_clients)]
        for i, path in enumerate(train_segs):
            self.client_segments[i % num_clients].append(path)

        # ── Lazy-built caches ──────────────────────────────────────────────────
        self._client_datasets: Dict[int, WaymoMultiTaskDataset] = {}
        self._eval_dataset: Optional[WaymoMultiTaskDataset]     = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _client_dataset(self, client_id: int) -> WaymoMultiTaskDataset:
        if client_id not in self._client_datasets:
            segs = self.client_segments[client_id]
            if not segs:
                raise ValueError(f"Client {client_id} has no segments assigned")
            self._client_datasets[client_id] = WaymoMultiTaskDataset(
                segs, pan_seq_len=self.pan_seq_len
            )
        return self._client_datasets[client_id]

    def _eval_ds(self) -> WaymoMultiTaskDataset:
        if self._eval_dataset is None:
            self._eval_dataset = WaymoMultiTaskDataset(
                self._eval_segments, pan_seq_len=self.pan_seq_len
            )
        return self._eval_dataset

    def _make_loader(
        self,
        ds: WaymoMultiTaskDataset,
        task: str,
        shuffle: bool = True,
    ) -> DataLoader:
        task_ds = ds.task_dataset(task)
        bs      = (
            max(1, self.batch_size // self.pan_seq_len)
            if task == "pan2d"
            else self.batch_size
        )
        return DataLoader(
            task_ds,
            batch_size  = bs,
            shuffle     = shuffle,
            num_workers = self.num_workers,
            collate_fn  = waymo_collate_fn,
            drop_last   = True,
            pin_memory  = True,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_loader(self, client_id: int, task: Optional[str] = None) -> DataLoader:
        """DataLoader for a single (client, task) pair."""
        return self._make_loader(
            self._client_dataset(client_id),
            task or self.arriving_task,
        )

    def get_loaders(self, client_id: int) -> Dict[str, DataLoader]:
        """DataLoaders for all four tasks for a given client."""
        ds = self._client_dataset(client_id)
        return {t: self._make_loader(ds, t) for t in TASKS}

    def get_cycle_loader(
        self, client_id: int, task: Optional[str] = None
    ) -> Iterator:
        """Infinite iterator over a task loader (for round-based training)."""
        return itertools.cycle(self.get_loader(client_id, task))

    def get_eval_loader(self, task: Optional[str] = None) -> DataLoader:
        """DataLoader over the held-out eval segments (no shuffle)."""
        return self._make_loader(self._eval_ds(), task or self.arriving_task, shuffle=False)

    def dataset_sizes(self, client_id: int) -> Dict[str, int]:
        """Number of samples per task for a given client (for FedAvg weighting)."""
        ds = self._client_dataset(client_id)
        return {
            "seg3d": len(ds.seg3d_index),
            "det3d": len(ds.det3d_index),
            "pan2d": len(ds.pan2d_index),
            "det2d": len(ds.det2d_index),
        }

    def arriving_task_sizes(self) -> List[int]:
        """Arriving-task dataset size for each client (for weighted FedAvg)."""
        return [
            self.dataset_sizes(c)[self.arriving_task]
            for c in range(self.num_clients)
        ]
