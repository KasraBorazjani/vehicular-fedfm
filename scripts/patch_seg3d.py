#!/usr/bin/env python3
"""
scripts/patch_seg3d.py
-----------------------
Re-extracts seg3d_label for .pt segment files where it was stored as None,
without re-downloading the large lidar / camera / box parquets.

For each .pt file:
  1. Skip if any frame already has a non-None seg3d_label (already patched).
  2. Download only the lidar_segmentation parquet for that segment from GCS.
  3. Re-run the fixed parse_seg3d_label (channel 1 = semantic class).
  4. Inject labels into the frame dicts and atomically re-save.

Prerequisite: scripts/preprocess_waymo.py must have the channel-1 fix applied.

Usage (run inside the Apptainer container on the HPC):
    python3 scripts/patch_seg3d.py \\
        --pt_dir /projects/academic/alipour/kasrabor/FedMMA-fork/waymo_pt/training

    # limit to first N segments for smoke-testing:
    python3 scripts/patch_seg3d.py --pt_dir ... --n_segs 5
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Import helpers from preprocess_waymo ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.preprocess_waymo import (
    _gcs_path,
    read_parquet_from_gcs,
    resolve_columns,
    parse_seg3d_label,
    NEEDED_COLS,
    FRAME_KEY,
    LIDAR_KEY,
    TOP_LIDAR,
)


# ── Per-segment patch logic ───────────────────────────────────────────────────

def patch_segment(pt_path: Path) -> str:
    """
    Attempt to inject seg3d_label into one .pt file.

    Returns one of: "patched", "skipped", "failed".
    """
    # ── load ──────────────────────────────────────────────────────────────────
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        log.warning(f"FAIL  {pt_path.name}: could not load — {e}")
        return "failed"

    frames = data.get("frames", [])
    if not frames:
        log.info(f"SKIP  {pt_path.name}: no frames")
        return "skipped"

    # Skip only if labels already contain valid (non-ignore) pixels.
    # A non-None but all-255 tensor means the old code stored instance IDs
    # instead of semantic classes — those files still need patching.
    _IGNORE = 255
    def _has_valid_labels(frames_):
        for f in frames_:
            lbl = f.get("seg3d_label")
            if lbl is not None and (lbl != _IGNORE).any():
                return True
        return False

    if _has_valid_labels(frames):
        log.info(f"SKIP  {pt_path.name}: already has valid seg3d labels")
        return "skipped"

    seg_id = data["segment_id"]
    split  = data["split"]
    log.info(f"PATCH {seg_id[:40]}…")

    # ── fetch lidar_seg parquet (small — only this component) ─────────────────
    try:
        gcs_path = _gcs_path(split, "lidar_segmentation", seg_id)
        df = read_parquet_from_gcs(gcs_path, columns=NEEDED_COLS["lidar_seg"])
    except Exception as e:
        log.warning(f"FAIL  {seg_id[:24]}: GCS fetch failed — {e}")
        return "failed"

    col = resolve_columns(df.columns.tolist())
    if not col.get("seg_values") or not col.get("seg_shape"):
        log.warning(f"FAIL  {seg_id[:24]}: seg columns unresolved")
        return "failed"

    # ── build frame_id → row lookup (TOP lidar only) ──────────────────────────
    top_df = df[df[LIDAR_KEY] == TOP_LIDAR]
    if top_df.empty:
        log.warning(f"FAIL  {seg_id[:24]}: no rows with laser_name={TOP_LIDAR}")
        return "failed"

    seg_by_fid = {int(row[FRAME_KEY]): row for _, row in top_df.iterrows()}

    del df, top_df
    gc.collect()

    # ── inject labels ─────────────────────────────────────────────────────────
    n_injected = 0
    for frame in frames:
        fid = frame["frame_id"]
        if fid not in seg_by_fid:
            continue
        seg_arr = parse_seg3d_label(
            seg_by_fid[fid][col["seg_values"]],
            seg_by_fid[fid][col["seg_shape"]],
        )
        if seg_arr is not None:
            frame["seg3d_label"] = torch.from_numpy(seg_arr)   # int16 [224, 224]
            n_injected += 1

    del seg_by_fid
    gc.collect()

    if n_injected == 0:
        log.warning(f"FAIL  {seg_id[:24]}: parse_seg3d_label returned None for all frames")
        return "failed"

    # ── atomic re-save ────────────────────────────────────────────────────────
    tmp = pt_path.with_suffix(".pt.tmp")
    torch.save(data, tmp)
    tmp.rename(pt_path)

    log.info(f"  → {n_injected}/{len(frames)} frames patched → {pt_path.name}")
    return "patched"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject seg3d_label into .pt files that stored it as None."
    )
    parser.add_argument(
        "--pt_dir", required=True,
        help="Directory containing .pt segment files (e.g. waymo_pt/training)"
    )
    parser.add_argument(
        "--n_segs", type=int, default=None,
        help="Limit to first N .pt files (smoke-test)"
    )
    args = parser.parse_args()

    pt_files = sorted(Path(args.pt_dir).glob("*.pt"))
    if args.n_segs:
        pt_files = pt_files[: args.n_segs]

    if not pt_files:
        log.error(f"No .pt files found in {args.pt_dir}")
        sys.exit(1)

    log.info(f"Processing {len(pt_files)} segment(s) in {args.pt_dir}")

    n_patched = n_skipped = n_failed = 0
    for pt_file in pt_files:
        result = patch_segment(pt_file)
        if result == "patched":
            n_patched += 1
        elif result == "skipped":
            n_skipped += 1
        else:
            n_failed += 1

    log.info("─" * 60)
    log.info(f"Done: {n_patched} patched, {n_skipped} skipped, {n_failed} failed"
             f" (out of {len(pt_files)} total)")
    if n_failed:
        log.info("Re-run to retry failed segments — already-patched files are skipped.")


if __name__ == "__main__":
    main()
