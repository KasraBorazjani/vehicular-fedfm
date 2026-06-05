#!/usr/bin/env python3
"""
scripts/check_detection_granularity.py
---------------------------------------
Checks whether the 14×14 patch grid (PATCH_SIZE=16 px in 224×224 space) is
sufficient granularity for the detection and segmentation labels produced by
scripts/preprocess_waymo.py.

Metrics reported
----------------
det2d   collision rate   — fraction of frames where ≥2 box centres share a patch
det2d   sub-patch rate   — fraction of boxes whose width OR height < 16 px (224-space)
det3d   collision rate   — same, using approximate spherical projection (see note below)
seg3d   patch purity     — fraction of patches containing only one class label
pan2d   patch purity     — same

.pt file structure (produced by scripts/preprocess_waymo.py)
------------------------------------------------------------
Each file is a segment dict:
    {
        "segment_id": str,
        "split":      str,
        "frames": [
            {
                "frame_id":    int,
                "image":       Tensor[3, 224, 224]  float16
                "lidar":       Tensor[3, 224, 224]  float16
                "seg3d_label": Tensor[224, 224]     int16   (0–21, 255=ignore) | None
                "det3d_boxes": Tensor[N, 8]         float32 (cx,cy,cz,sx,sy,sz,heading,cls) metres | None
                "pan2d_label": Tensor[224, 224]     int32   (0–27, 255=ignore) | None
                "det2d_boxes": Tensor[M, 5]         float32 (cx_n,cy_n,w_n,h_n,cls) normalised | None
            },
            ...
        ]
    }

det3d note
----------
The exact mapping from 3-D ego-frame coordinates to range-image pixels requires
per-beam azimuth/elevation calibration data, which is not stored in the .pt files.
This script uses a spherical approximation (TOP LiDAR angular bounds from the
Waymo sensor spec) to estimate patch assignments.  The collision rate reported
for det3d is therefore approximate; use it for order-of-magnitude guidance only.

Usage
-----
    python scripts/check_detection_granularity.py \\
        --pt_dir /path/to/waymo_pt/training \\
        --n_segs 50
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import torch

# ── Constants (must match preprocess_waymo.py) ────────────────────────────────

IMG_SIZE   = 224
PATCH_SIZE = 16          # pixels per patch side in 224×224 space
GRID       = IMG_SIZE // PATCH_SIZE   # 14

# Approximate Waymo TOP LiDAR angular geometry
# Source: Waymo Open Dataset sensor spec / community measurements.
# Exact mapping requires the lidar_calibration parquet component.
LIDAR_ROWS   = 64
LIDAR_COLS   = 2650
LIDAR_EL_MIN = math.radians(-17.6)   # lowest beam elevation
LIDAR_EL_MAX = math.radians(2.4)     # highest beam elevation

DET2D_CLASS_NAMES = ["vehicle", "pedestrian", "cyclist"]
DET3D_CLASS_NAMES = ["vehicle", "pedestrian", "sign", "cyclist"]

SEP = "─" * 66


# ── Spatial helpers ───────────────────────────────────────────────────────────

def to_patch_idx(cx_224: float, cy_224: float) -> int:
    """(cx, cy) in 224×224 pixels → flat patch index [0, 195]."""
    col = max(0, min(GRID - 1, int(cx_224 / PATCH_SIZE)))
    row = max(0, min(GRID - 1, int(cy_224 / PATCH_SIZE)))
    return row * GRID + col


def project_3d_approx(cx: float, cy: float, cz: float) -> int:
    """
    Approximate: 3-D ego-frame point (metres) → flat patch index.

    Maps azimuth angle → range-image column and elevation → row using rough
    TOP LiDAR bounds, then scales to the 14×14 grid.
    """
    az    = math.atan2(cy, cx)                              # [-π, π]
    dxy   = math.sqrt(cx * cx + cy * cy)
    el    = math.atan2(cz, dxy) if dxy > 0.0 else 0.0

    ri_col = (az + math.pi) / (2.0 * math.pi) * LIDAR_COLS
    el_rng = LIDAR_EL_MAX - LIDAR_EL_MIN
    # higher elevation → lower row index (top of image)
    ri_row = (LIDAR_EL_MAX - el) / el_rng * LIDAR_ROWS

    cx_224 = ri_col / LIDAR_COLS * IMG_SIZE
    cy_224 = ri_row / LIDAR_ROWS * IMG_SIZE
    return to_patch_idx(cx_224, cy_224)


# ── Segmentation helper ───────────────────────────────────────────────────────

def patch_purity(label_224: torch.Tensor, ignore: int = 255) -> float:
    """
    Fraction of 14×14 patches that are 'pure': every non-ignore pixel in the
    patch carries the same class label.

    label_224 : [224, 224] integer tensor (any dtype).
    Returns NaN if no patch has any valid (non-ignore) pixel.

    Reshape derivation
    ------------------
    reshape(GRID, PATCH_SIZE, GRID, PATCH_SIZE) yields dims
        [patch_row, row_in_patch, patch_col, col_in_patch].
    permute(0, 2, 1, 3) → [patch_row, patch_col, row_in_patch, col_in_patch].
    reshape(GRID*GRID, PATCH_SIZE*PATCH_SIZE) → [196, 256]: one row per patch.
    """
    label = label_224.long()
    label = label.reshape(GRID, PATCH_SIZE, GRID, PATCH_SIZE)
    label = label.permute(0, 2, 1, 3)                 # [14, 14, 16, 16]
    label = label.reshape(GRID * GRID, PATCH_SIZE * PATCH_SIZE)  # [196, 256]

    n_pure  = 0
    n_valid = 0
    for patch in label:
        valid = patch[patch != ignore]
        if valid.numel() == 0:
            continue
        n_valid += 1
        if valid.unique().numel() == 1:
            n_pure += 1

    return n_pure / n_valid if n_valid > 0 else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check patch-level detection/segmentation granularity."
    )
    parser.add_argument(
        "--pt_dir", required=True,
        help="Directory containing .pt segment files (e.g. waymo_pt/training)"
    )
    parser.add_argument(
        "--n_segs", type=int, default=None,
        help="Max segment files to inspect (default: all)"
    )
    args = parser.parse_args()

    pt_files = sorted(Path(args.pt_dir).glob("*.pt"))
    if args.n_segs:
        pt_files = pt_files[: args.n_segs]

    if not pt_files:
        print(f"No .pt files found in {args.pt_dir}")
        return

    print(f"Inspecting {len(pt_files)} segment(s) in {args.pt_dir}\n")

    # ── Accumulators ─────────────────────────────────────────────────────────

    # det2d
    d2_frames      = 0          # frames containing at least one box
    d2_boxes       = 0          # total boxes
    d2_collisions  = 0          # frames with ≥1 patch collision
    d2_small       = 0          # boxes smaller than one patch in w or h
    d2_cls_total   = Counter()
    d2_cls_small   = Counter()
    d2_cls_collide = Counter()  # boxes that share their patch with another box

    # det3d (approximate)
    d3_frames      = 0
    d3_boxes       = 0
    d3_collisions  = 0
    d3_bpf_list    = []         # boxes-per-frame, for avg / max
    d3_cls_total   = Counter()

    # segmentation purity
    seg3d_purity = []
    pan2d_purity = []

    # ── Iterate ───────────────────────────────────────────────────────────────

    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [warn] {pt_file.name}: {e}")
            continue

        for frame in data.get("frames", []):

            # ── det2d ─────────────────────────────────────────────────────────
            boxes2d = frame.get("det2d_boxes")
            if boxes2d is not None and boxes2d.numel() > 0:
                d2_frames += 1
                n = len(boxes2d)

                # Coordinates: (cx_norm, cy_norm, w_norm, h_norm, cls)
                # Normalised by original image dims → multiply by IMG_SIZE
                cx_224 = boxes2d[:, 0] * IMG_SIZE   # [M]
                cy_224 = boxes2d[:, 1] * IMG_SIZE
                w_224  = boxes2d[:, 2] * IMG_SIZE
                h_224  = boxes2d[:, 3] * IMG_SIZE
                cls    = boxes2d[:, 4].long()

                pidxs  = [to_patch_idx(cx_224[i].item(), cy_224[i].item())
                          for i in range(n)]
                counts = Counter(pidxs)

                d2_boxes += n
                if any(v > 1 for v in counts.values()):
                    d2_collisions += 1

                for i in range(n):
                    c = cls[i].item()
                    d2_cls_total[c] += 1
                    if w_224[i].item() < PATCH_SIZE or h_224[i].item() < PATCH_SIZE:
                        d2_small += 1
                        d2_cls_small[c] += 1
                    if counts[pidxs[i]] > 1:
                        d2_cls_collide[c] += 1

            # ── det3d (approximate) ───────────────────────────────────────────
            boxes3d = frame.get("det3d_boxes")
            if boxes3d is not None and boxes3d.numel() > 0:
                d3_frames += 1
                n = len(boxes3d)
                d3_bpf_list.append(n)

                # Columns: (cx, cy, cz, sx, sy, sz, heading, cls)
                pidxs3 = [
                    project_3d_approx(
                        boxes3d[i, 0].item(),
                        boxes3d[i, 1].item(),
                        boxes3d[i, 2].item(),
                    )
                    for i in range(n)
                ]
                counts3 = Counter(pidxs3)

                d3_boxes += n
                if any(v > 1 for v in counts3.values()):
                    d3_collisions += 1

                for i in range(n):
                    d3_cls_total[int(boxes3d[i, 7].item())] += 1

            # ── seg3d purity ──────────────────────────────────────────────────
            seg = frame.get("seg3d_label")
            if seg is not None:
                seg3d_purity.append(patch_purity(seg, ignore=255))

            # ── pan2d purity ──────────────────────────────────────────────────
            pan = frame.get("pan2d_label")
            if pan is not None:
                pan2d_purity.append(patch_purity(pan, ignore=255))

    # ── Report ────────────────────────────────────────────────────────────────

    print(SEP)
    print("DET2D  (2D camera boxes, 14×14 patch grid, 16×16 px per patch)")
    print(SEP)
    if d2_frames == 0:
        print("  No det2d frames found.")
    else:
        print(f"  Frames with boxes            : {d2_frames}")
        print(f"  Total boxes                  : {d2_boxes}")
        print(f"  Avg boxes / frame            : {d2_boxes / d2_frames:.1f}")
        print(f"  Frames with ≥1 collision     : {d2_collisions}"
              f"  ({100 * d2_collisions / d2_frames:.1f}%)")
        print(f"  Boxes smaller than one patch : {d2_small}/{d2_boxes}"
              f"  ({100 * d2_small / d2_boxes:.1f}%)"
              f"  [width OR height < 16 px in 224-space]")
        print()
        print(f"  {'Class':<15} {'Total':>8} {'Small (%)':>12} {'Collide (%)':>14}")
        print(f"  {'─'*15} {'─'*8} {'─'*12} {'─'*14}")
        for i, name in enumerate(DET2D_CLASS_NAMES):
            n   = d2_cls_total[i]
            s   = d2_cls_small[i]
            col = d2_cls_collide[i]
            if n == 0:
                continue
            print(f"  {name:<15} {n:>8}"
                  f"  {s:>5} ({100*s/n:4.1f}%)"
                  f"  {col:>5} ({100*col/n:4.1f}%)")

    print()
    print(SEP)
    print("DET3D  (3D LiDAR boxes, APPROXIMATE — spherical projection)")
    print("  Exact mapping needs lidar_calibration component (not in .pt files).")
    print(SEP)
    if d3_frames == 0:
        print("  No det3d frames found.")
    else:
        avg_bpf = sum(d3_bpf_list) / len(d3_bpf_list)
        max_bpf = max(d3_bpf_list)
        print(f"  Frames with boxes                    : {d3_frames}")
        print(f"  Total boxes                          : {d3_boxes}")
        print(f"  Avg / max boxes per frame            : {avg_bpf:.1f} / {max_bpf}")
        print(f"  Frames with ≥1 collision (approx)    : {d3_collisions}"
              f"  ({100 * d3_collisions / d3_frames:.1f}%)")
        print()
        print("  Per-class box counts:")
        for i, name in enumerate(DET3D_CLASS_NAMES):
            print(f"    {name:<15}: {d3_cls_total[i]}")

    print()
    print(SEP)
    print("SEG3D  (fraction of 14×14 patches that are single-class)")
    print(SEP)
    valid_s3 = [v for v in seg3d_purity if not math.isnan(v)]
    if not valid_s3:
        print("  No seg3d labels found.")
    else:
        avg_p = sum(valid_s3) / len(valid_s3)
        frac_90 = sum(1 for v in valid_s3 if v > 0.9) / len(valid_s3)
        print(f"  Frames with seg3d labels     : {len(valid_s3)}")
        print(f"  Avg patch purity             : {100 * avg_p:.1f}%")
        print(f"  Frames with purity > 90%     : {100 * frac_90:.1f}%")

    print()
    print(SEP)
    print("PAN2D  (fraction of 14×14 patches that are single-class)")
    print(SEP)
    valid_p2 = [v for v in pan2d_purity if not math.isnan(v)]
    if not valid_p2:
        print("  No pan2d labels found.")
    else:
        avg_p = sum(valid_p2) / len(valid_p2)
        frac_90 = sum(1 for v in valid_p2 if v > 0.9) / len(valid_p2)
        print(f"  Frames with pan2d labels     : {len(valid_p2)}")
        print(f"  Avg patch purity             : {100 * avg_p:.1f}%")
        print(f"  Frames with purity > 90%     : {100 * frac_90:.1f}%")

    print()
    print(SEP)
    print("SUMMARY")
    print(SEP)
    if d2_frames > 0:
        print(f"  det2d  collision rate  : {100 * d2_collisions / d2_frames:.1f}% of frames")
        print(f"  det2d  sub-patch boxes : {100 * d2_small / d2_boxes:.1f}% of all boxes")
    if d3_frames > 0:
        print(f"  det3d  collision rate  : {100 * d3_collisions / d3_frames:.1f}% (APPROXIMATE)")
    if valid_s3:
        print(f"  seg3d  patch purity    : {100 * sum(valid_s3) / len(valid_s3):.1f}%")
    if valid_p2:
        print(f"  pan2d  patch purity    : {100 * sum(valid_p2) / len(valid_p2):.1f}%")
    print()


if __name__ == "__main__":
    main()
