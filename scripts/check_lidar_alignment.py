"""
scripts/check_lidar_alignment.py
---------------------------------
Visual sanity-check that the LiDAR scatter and seg3d labels are aligned
with the corresponding camera image inside a preprocessed .pt file.

Renders the first N frames (default 10) as a grid:
  col 0 — camera image (CLIP denorm)
  col 1 — LiDAR channel (black background, plasma hits)
  col 2 — camera + seg3d label overlay  [only when seg3d_label is present]

Usage
-----
  python scripts/check_lidar_alignment.py --pt_file data/waymo/seg_xxx_cam1.pt
  python scripts/check_lidar_alignment.py --pt_file data/waymo/seg_xxx_cam1.pt --n_frames 5
  python scripts/check_lidar_alignment.py --pt_file data/waymo/seg_xxx_cam1.pt --save out.png
  python scripts/check_lidar_alignment.py --pt_file data/waymo/seg_xxx_cam1.pt --lidar_ch 1
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

# ── Constants matching preprocess_waymo.py ────────────────────────────────────

CLIP_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_SEG3D_COLORS = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
    "#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4",
    "#469990","#dcbeff","#9A6324","#fffac8","#800000",
    "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9",
    "#ffffff","#000000",
]
SEG3D_CMAP = mcolors.ListedColormap(_SEG3D_COLORS, name="seg3d")
SEG3D_NORM = mcolors.BoundaryNorm(list(range(23)), SEG3D_CMAP.N)

LIDAR_CH_LABEL = ["range (norm)", "intensity (norm)", "elongation (norm)"]


def denorm_image(t: torch.Tensor) -> np.ndarray:
    """[3,H,W] float → [H,W,3] uint8, undoing CLIP normalisation."""
    img = t.numpy().transpose(1, 2, 0).astype(np.float32)
    img = img * CLIP_STD + CLIP_MEAN
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_file",   required=True)
    ap.add_argument("--n_frames",  type=int, default=10,
                    help="How many frames to show (default 10)")
    ap.add_argument("--lidar_ch",  type=int, default=0,
                    help="LiDAR channel: 0=range, 1=intensity, 2=elongation")
    ap.add_argument("--save",      default=None,
                    help="Save figure here instead of displaying")
    args = ap.parse_args()

    data   = torch.load(args.pt_file, map_location="cpu", weights_only=False)
    frames = data["frames"]
    total  = len(frames)
    n      = min(args.n_frames, total)
    ch     = args.lidar_ch

    print(f"Loaded {total} frames from {args.pt_file}  — rendering first {n}")

    # Determine if any of the selected frames has seg3d so we know ncols
    has_any_seg = any(frames[i].get("seg3d_label") is not None for i in range(n))
    ncols = 3 if has_any_seg else 2

    fig, axes = plt.subplots(
        n, ncols,
        figsize=(5 * ncols, 3.5 * n),
        squeeze=False,
    )
    fig.suptitle(
        f"{Path(args.pt_file).name}  —  LiDAR ch{ch} ({LIDAR_CH_LABEL[ch]})",
        fontsize=13, y=1.002,
    )

    for row, frame in enumerate(frames[:n]):
        img_uint8 = denorm_image(frame["image"].float())

        lidar    = frame["lidar"].float()          # [3,224,224]
        lidar_ch = lidar[ch].numpy()               # [224,224]
        hit_mask = lidar_ch > 0
        hit_pct  = hit_mask.sum() / hit_mask.size * 100

        seg     = frame.get("seg3d_label")
        has_seg = seg is not None

        # ── col 0: camera ──────────────────────────────────────────────────────
        axes[row, 0].imshow(img_uint8)
        axes[row, 0].set_ylabel(f"frame {row}", fontsize=9)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("Camera (CLIP denorm)", fontsize=10)

        # ── col 1: LiDAR standalone ────────────────────────────────────────────
        lidar_display = np.zeros_like(lidar_ch)
        lidar_display[hit_mask] = lidar_ch[hit_mask]
        vmax = float(lidar_ch[hit_mask].max()) if hit_mask.any() else 1.0
        im = axes[row, 1].imshow(lidar_display, cmap="plasma", vmin=0, vmax=vmax)
        axes[row, 1].axis("off")
        if row == 0:
            axes[row, 1].set_title(f"LiDAR ch{ch}", fontsize=10)
        axes[row, 1].text(
            0.02, 0.02, f"{hit_pct:.1f}% hits",
            transform=axes[row, 1].transAxes,
            fontsize=7, color="white", va="bottom",
        )

        # ── col 2: camera + seg3d overlay (when available) ────────────────────
        if has_any_seg:
            ax = axes[row, 2]
            if has_seg:
                seg_np    = seg.numpy().astype(np.int32)
                seg_valid = seg_np != 255
                seg_disp  = np.where(seg_valid, seg_np, np.nan).astype(np.float32)
                ax.imshow(img_uint8)
                ax.imshow(seg_disp, cmap=SEG3D_CMAP, norm=SEG3D_NORM, alpha=0.55)
                n_cls = len(np.unique(seg_np[seg_valid]))
                ax.text(
                    0.02, 0.02, f"{n_cls} classes",
                    transform=ax.transAxes,
                    fontsize=7, color="white", va="bottom",
                )
            else:
                ax.text(0.5, 0.5, "no seg3d", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title("Camera + seg3d", fontsize=10)

    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=120, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        try:
            matplotlib.use("TkAgg")
            plt.show()
        except Exception:
            out = "alignment_check.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            print(f"No display — saved to {out}")


if __name__ == "__main__":
    main()
