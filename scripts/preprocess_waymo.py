#!/usr/bin/env python3
"""
scripts/preprocess_waymo.py
-----------------------------
Preprocesses Waymo Open Dataset v2 segments from GCS into .pt files
for FedMMA federated multi-task learning.

STEP 1 — discover actual column names (do this before anything else):
  python scripts/preprocess_waymo.py --discover --split training

STEP 2 — preprocess all qualifying segments:
  python scripts/preprocess_waymo.py \\
      --split training \\
      --out_dir /scratch/waymo_pt \\
      --workers 4

STEP 3 — smoke-test a single segment:
  python scripts/preprocess_waymo.py \\
      --split training \\
      --out_dir /tmp/waymo_test \\
      --max_segs 1

Output layout (one .pt per segment PER CAMERA):
  {out_dir}/{split}/{segment_id}_cam{camera_id}.pt
  camera_id ∈ {1:FRONT, 2:FRONT_LEFT, 3:FRONT_RIGHT, 4:SIDE_LEFT, 5:SIDE_RIGHT}

Each .pt is a dict:
  {
    "segment_id": str,
    "camera_id":  int,
    "split":      str,
    "frames": [
      {
        "frame_id":       int,
        "image":          Tensor[3,224,224],   # float16, CLIP-normalized
        "lidar":          Tensor[3,224,224],   # float16, [range, intensity, elongation],
                                                #   scattered into this camera's pixel grid
                                                #   via the exact LiDAR→camera projection
                                                #   (see scatter_lidar_to_camera()).
        "seg3d_label":    Tensor[224,224] | None,  # int16, 0-21, 255=ignore; same projection
        "det3d_boxes":    Tensor[N,8]            ,  # float32 (cx,cy,cz,sx,sy,sz,heading,cls);
                                                     #   filtered to this camera's azimuth FOV
        "pan2d_label":    Tensor[224,224] | None,  # int32, 0-27, 255=ignore
        "pan2d_instance": Tensor[224,224] | None,  # int32, instance IDs
        "det2d_boxes":    Tensor[M,5]     | None,  # float32 (cx_n,cy_n,w_n,h_n,cls)
      },
      ...
    ]
  }

Prerequisites:
  pip install waymo-open-dataset-tf-2-12-0 dask[dataframe] pyarrow Pillow torch
"""

import warnings
warnings.simplefilter("ignore", FutureWarning)

import argparse
import io
import logging
import sys
from pathlib import Path
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    import tensorflow as tf
    import pyarrow.parquet as pq
    import pandas as pd
    from PIL import Image
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BUCKET = "gs://waymo_open_dataset_v_2_0_1"

# Components we read; values are the GCS subdirectory names
COMPONENTS = {
    "lidar":            "lidar",
    "lidar_camera_proj":"lidar_camera_projection",
    "camera_image":     "camera_image",
    "lidar_seg":        "lidar_segmentation",
    "lidar_box":        "lidar_box",
    "camera_seg":       "camera_segmentation",   # optional — absent for 102 segments
    "camera_box":       "camera_box",            # optional
}

# Sensor filter constants (Waymo enum values)
# CameraName: 1=FRONT 2=FRONT_LEFT 3=FRONT_RIGHT 4=SIDE_LEFT 5=SIDE_RIGHT
CAMERAS   = [1, 2, 3, 4, 5]
TOP_LIDAR = 1   # LaserName.TOP

# Target spatial resolution for CLIP ViT-B/16
IMG_SIZE = 224

# CLIP ViT-B/16 image normalization (mean / std per channel)
CLIP_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# LiDAR normalization constants (per-channel clip values)
LIDAR_RANGE_MAX  = 75.0   # meters; Waymo TOP LiDAR max reliable range
LIDAR_INTENS_MAX = 1.0    # intensity is already in [0, 1]
LIDAR_ELONG_MAX  = 1.5    # elongation typical upper bound

# Shared key column names (same across all components)
SEG_KEY   = "key.segment_context_name"
FRAME_KEY = "key.frame_timestamp_micros"
CAM_KEY   = "key.camera_name"
LIDAR_KEY = "key.laser_name"


# ── Label remapping tables ────────────────────────────────────────────────────

# seg3d: Waymo range-image label IDs 1–22 → our 0-based class indices
#   Waymo 0 = TYPE_UNDEFINED → ignored (255)
SEG3D_IGNORE = 255
SEG3D_WAYMO_TO_IDX = {w: w - 1 for w in range(1, 23)}
# Our class order (matches utils/text_templates.py TASK_CLASSES["seg3d"]):
#   0:car  1:truck  2:bus  3:other vehicle  4:motorcyclist  5:bicyclist
#   6:pedestrian  7:sign  8:traffic light  9:pole  10:construction cone
#   11:bicycle  12:motorcycle  13:building  14:vegetation  15:tree trunk
#   16:curb  17:road  18:lane marker  19:other ground  20:walkable  21:sidewalk

# det3d: Waymo lidar box type enum → 0-based index
#   1=VEHICLE  2=PEDESTRIAN  3=SIGN  4=CYCLIST
DET3D_TYPE_MAP = {1: 0, 2: 1, 3: 2, 4: 3}

# pan2d: Waymo camera segmentation semantic IDs → our 0-based class indices
#   (from SemanticLabel proto enum; 0 = UNDEFINED → ignored)
# Our class order (matches TASK_CLASSES["pan2d"]):
#   0:car  1:bus  2:truck  3:other large vehicle  4:trailer  5:ego vehicle
#   6:motorcycle  7:bicycle  8:pedestrian  9:cyclist  10:motorcyclist
#   11:ground animal  12:bird  13:pole  14:sign  15:traffic light
#   16:construction cone  17:pedestrian object  18:building  19:road
#   20:sidewalk  21:road marker  22:lane marker  23:vegetation  24:sky
#   25:ground  26:static  27:dynamic
PAN2D_IGNORE = 255
PAN2D_WAYMO_TO_IDX = {
    0:  PAN2D_IGNORE,  # UNDEFINED
    1:  5,   # EGO_VEHICLE
    2:  0,   # CAR
    3:  2,   # TRUCK
    4:  1,   # BUS
    5:  3,   # OTHER_LARGE_VEHICLE
    6:  7,   # BICYCLE
    7:  6,   # MOTORCYCLE
    8:  4,   # TRAILER
    9:  8,   # PEDESTRIAN
    10: 9,   # CYCLIST
    11: 10,  # MOTORCYCLIST
    12: 12,  # BIRD
    13: 11,  # GROUND_ANIMAL
    14: 16,  # CONSTRUCTION_CONE_POLE
    15: 13,  # POLE
    16: 17,  # PEDESTRIAN_OBJECT
    17: 14,  # SIGN
    18: 15,  # TRAFFIC_LIGHT
    19: 18,  # BUILDING
    20: 19,  # ROAD
    21: 22,  # LANE_MARKER
    22: 21,  # ROAD_MARKER
    23: 20,  # SIDEWALK
    24: 23,  # VEGETATION
    25: 24,  # SKY
    26: 25,  # GROUND
    27: 27,  # DYNAMIC
    28: 26,  # STATIC
}

# det2d: Waymo camera box type enum → 0-based index
#   1=VEHICLE  2=PEDESTRIAN  4=CYCLIST  (3=SIGN excluded from det2d)
DET2D_TYPE_MAP = {1: 0, 2: 1, 4: 2}

# Build a vectorised lookup table for fast numpy remapping (index = Waymo ID)
_SEG3D_LUT = np.full(256, SEG3D_IGNORE, dtype=np.int16)
for _w, _i in SEG3D_WAYMO_TO_IDX.items():
    _SEG3D_LUT[_w] = _i

_PAN2D_LUT = np.full(256, PAN2D_IGNORE, dtype=np.int32)
for _w, _i in PAN2D_WAYMO_TO_IDX.items():
    if _i != PAN2D_IGNORE:
        _PAN2D_LUT[_w] = _i


# Loading only the columns we actually use cuts peak RAM by ~60 %:
#   • lidar:      skips return2 and all metadata columns
#   • camera_image: skips 12 velocity/pose/shutter columns
#   • lidar_seg:  skips return2
#   • camera_seg: skips instance-mapping and metadata columns
NEEDED_COLS = {
    "lidar": [
        SEG_KEY, FRAME_KEY, LIDAR_KEY,
        "[LiDARComponent].range_image_return1.values",
        "[LiDARComponent].range_image_return1.shape",
    ],
    "lidar_camera_proj": [
        SEG_KEY, FRAME_KEY, LIDAR_KEY,
        "[LiDARCameraProjectionComponent].range_image_return1.values",
        "[LiDARCameraProjectionComponent].range_image_return1.shape",
    ],
    "camera_image": [
        SEG_KEY, FRAME_KEY, CAM_KEY,
        "[CameraImageComponent].image",
    ],
    "lidar_seg": [
        SEG_KEY, FRAME_KEY, LIDAR_KEY,
        "[LiDARSegmentationLabelComponent].range_image_return1.values",
        "[LiDARSegmentationLabelComponent].range_image_return1.shape",
    ],
    "lidar_box": [
        SEG_KEY, FRAME_KEY,
        "[LiDARBoxComponent].box.center.x", "[LiDARBoxComponent].box.center.y",
        "[LiDARBoxComponent].box.center.z", "[LiDARBoxComponent].box.size.x",
        "[LiDARBoxComponent].box.size.y",   "[LiDARBoxComponent].box.size.z",
        "[LiDARBoxComponent].box.heading",  "[LiDARBoxComponent].type",
    ],
    "camera_seg": [
        SEG_KEY, FRAME_KEY, CAM_KEY,
        "[CameraSegmentationLabelComponent].panoptic_label",
        "[CameraSegmentationLabelComponent].panoptic_label_divisor",
    ],
    "camera_box": [
        SEG_KEY, FRAME_KEY, CAM_KEY,
        "[CameraBoxComponent].box.center.x", "[CameraBoxComponent].box.center.y",
        "[CameraBoxComponent].box.size.x",   "[CameraBoxComponent].box.size.y",
        "[CameraBoxComponent].type",
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# GCS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _gcs_path(split: str, component: str, seg_id: str) -> str:
    return f"{BUCKET}/{split}/{component}/{seg_id}.parquet"


def segment_ids_for_component(split: str, component: str) -> List[str]:
    """Return all segment IDs (parquet stems) for a given split + component."""
    pattern = f"{BUCKET}/{split}/{component}/*.parquet"
    files = tf.io.gfile.glob(pattern)
    return [Path(f).stem for f in files]


def read_parquet_from_gcs(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a single GCS parquet file into a pandas DataFrame.

    The raw bytes blob can be 500 MB–2 GB for a lidar parquet.
    We delete it immediately after parsing so the worker's RSS drops
    before the next component file is loaded.
    """
    raw = tf.io.gfile.GFile(path, "rb").read()
    buf = io.BytesIO(raw)
    del raw          # free the blob — don't wait for GC
    gc.collect()
    return pd.read_parquet(buf, columns=columns)


def discover_schema(split: str, component: str) -> List[str]:
    """Return column names from the first parquet of a component."""
    pattern = f"{BUCKET}/{split}/{component}/*.parquet"
    files = tf.io.gfile.glob(pattern)
    if not files:
        return []
    raw = tf.io.gfile.GFile(files[0], "rb").read()
    return pq.read_schema(io.BytesIO(raw)).names


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN NAME RESOLUTION
# (tries multiple plausible names; populated once per process on first use)
# ══════════════════════════════════════════════════════════════════════════════

def _first_present(candidates: List[str], available: List[str]) -> Optional[str]:
    """Return the first candidate that appears in available, or None."""
    for c in candidates:
        if c in available:
            return c
    return None


def resolve_columns(df_columns: List[str]) -> dict:
    """
    Given the actual column list of a DataFrame, return a dict mapping
    logical names to physical column names.

    Column names confirmed by --discover on 2026-05-25.
    All component-specific columns use the [ComponentName]. prefix.
    """
    c = list(df_columns)
    resolved = {}

    mapping = {
        # ── camera_image ──────────────────────────────────────────────────────
        # No width/height stored — dimensions are read from the JPEG itself.
        "cam_image":   ["[CameraImageComponent].image"],

        # ── lidar (range image) ───────────────────────────────────────────────
        # Shape column is a struct; use _parse_shape() to extract dims.
        "ri_values":   ["[LiDARComponent].range_image_return1.values"],
        "ri_shape":    ["[LiDARComponent].range_image_return1.shape"],

        # ── lidar_camera_projection ───────────────────────────────────────────
        # [H, W, 6] per range-image pixel: (cam1_name, cam1_u, cam1_v,
        #                                    cam2_name, cam2_u, cam2_v)
        # -1 in a *_name slot means no projection for that slot.
        "proj_values": ["[LiDARCameraProjectionComponent].range_image_return1.values"],
        "proj_shape":  ["[LiDARCameraProjectionComponent].range_image_return1.shape"],

        # ── lidar_segmentation ────────────────────────────────────────────────
        "seg_values":  ["[LiDARSegmentationLabelComponent].range_image_return1.values"],
        "seg_shape":   ["[LiDARSegmentationLabelComponent].range_image_return1.shape"],

        # ── lidar_box ─────────────────────────────────────────────────────────
        "box_cx":      ["[LiDARBoxComponent].box.center.x"],
        "box_cy":      ["[LiDARBoxComponent].box.center.y"],
        "box_cz":      ["[LiDARBoxComponent].box.center.z"],
        "box_lx":      ["[LiDARBoxComponent].box.size.x"],
        "box_ly":      ["[LiDARBoxComponent].box.size.y"],
        "box_lz":      ["[LiDARBoxComponent].box.size.z"],
        "box_heading": ["[LiDARBoxComponent].box.heading"],
        "box_type":    ["[LiDARBoxComponent].type"],

        # ── camera_segmentation ───────────────────────────────────────────────
        "pan_label":   ["[CameraSegmentationLabelComponent].panoptic_label"],
        "pan_divisor": ["[CameraSegmentationLabelComponent].panoptic_label_divisor"],

        # ── camera_box ────────────────────────────────────────────────────────
        # box.size.x = width (horizontal), box.size.y = height (vertical)
        "cbox_cx":     ["[CameraBoxComponent].box.center.x"],
        "cbox_cy":     ["[CameraBoxComponent].box.center.y"],
        "cbox_w":      ["[CameraBoxComponent].box.size.x"],
        "cbox_h":      ["[CameraBoxComponent].box.size.y"],
        "cbox_type":   ["[CameraBoxComponent].type"],
    }

    for key, candidates in mapping.items():
        col = _first_present(candidates, c)
        if col is None:
            # DEBUG only: each DataFrame only contains its own component's columns,
            # so most "not resolved" messages are expected cross-component noise.
            log.debug(f"  Column '{key}' not resolved in this DataFrame. Tried: {candidates}.")
        resolved[key] = col

    return resolved


def _parse_shape(shape_val) -> Optional[List[int]]:
    """
    Unpack a Waymo range-image shape value into a plain list of ints.

    The '.shape' column is a Parquet struct.  Depending on the pyarrow
    version it deserialises as:
      - a list/tuple  : [64, 2650, 4]
      - a numpy array : array([64, 2650, 4])
      - a dict        : {"dims": [64, 2650, 4]}

    Falls back to inferring [64, 2650, 4] (TOP LiDAR) from value count.
    """
    try:
        if isinstance(shape_val, (list, tuple)):
            return [int(x) for x in shape_val]
        if hasattr(shape_val, "tolist"):          # numpy array
            return [int(x) for x in shape_val.tolist()]
        if isinstance(shape_val, dict):
            for key in ("dims", "values", "shape"):
                if key in shape_val:
                    return [int(x) for x in shape_val[key]]
        # last resort: iterate
        return [int(x) for x in shape_val]
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def decode_and_resize_jpeg(jpeg_bytes: bytes,
                           size: int = IMG_SIZE) -> tuple:
    """
    JPEG bytes → (float32 [3, size, size], orig_w, orig_h).

    Returns the original image dimensions alongside the normalised array
    so callers can normalise 2-D bounding box coordinates.
    CLIP normalisation: mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276].
    """
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    orig_w, orig_h = img.size          # PIL: (width, height)
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0   # [H,W,3] ∈ [0,1]
    arr = (arr - CLIP_MEAN) / CLIP_STD              # CLIP normalisation
    return arr.transpose(2, 0, 1), orig_w, orig_h   # [3,H,W], int, int


# ══════════════════════════════════════════════════════════════════════════════
# LIDAR PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
#
# Instead of resizing the full 360° range image (most of which the camera
# never sees), every LiDAR-derived tensor is scattered directly into the
# target camera's pixel grid using the exact lidar_camera_projection
# component — no hardcoded per-camera FOV angles needed.

def _parse_projection(proj_values, proj_shape, H: int, W: int) -> Optional[np.ndarray]:
    """Parse the lidar_camera_projection range image → float32 [H, W, 6]."""
    try:
        dims = _parse_shape(proj_shape)
        if dims is None:
            dims = [H, W, 6]
        return np.array(proj_values, dtype=np.float32).reshape(dims)
    except Exception as e:
        log.debug(f"_parse_projection error: {e}")
        return None


def _camera_hit_slots(proj: np.ndarray, cam_id: int) -> List[Tuple[np.ndarray, int, int]]:
    """
    Channels of `proj` are (cam1_name, cam1_u, cam1_v, cam2_name, cam2_u, cam2_v).
    Returns a list of (boolean mask [H,W], u_channel, v_channel) for whichever
    of the two projection slots matches cam_id (a range-image pixel can project
    onto two cameras in their overlap region, so both slots are checked).
    """
    hits = []
    for cam_slot, u_slot, v_slot in ((0, 1, 2), (3, 4, 5)):
        mask = proj[..., cam_slot].astype(np.int32) == cam_id
        if mask.any():
            hits.append((mask, u_slot, v_slot))
    return hits


def scatter_lidar_to_camera(
    ri_values, ri_shape, proj_values, proj_shape,
    cam_id: int, orig_w: int, orig_h: int, size: int = IMG_SIZE,
) -> Optional[np.ndarray]:
    """
    Scatter TOP LiDAR returns into this camera's pixel grid via the exact
    LiDAR→camera projection. Each projected return lands at its (u, v) pixel
    in the *original* camera image, rescaled by size/orig_w, size/orig_h —
    the same scale factor applied when the camera image itself was resized.
    Pixels with no projected return are zero. Where two returns land on the
    same output pixel, the nearer one (smaller range) wins.

    Output : float32 [3, size, size], channels = [range, intensity, elongation],
             each normalised to [0, 1]. Returns None on error.
    """
    try:
        dims = _parse_shape(ri_shape)
        if dims is None:
            n = len(ri_values)
            dims = [64, n // (64 * 4), 4]
        ri = np.array(ri_values, dtype=np.float32).reshape(dims)   # [H, W, C]
        H, W = ri.shape[:2]

        proj = _parse_projection(proj_values, proj_shape, H, W)
        if proj is None:
            return None

        valid   = ri[..., 0] > 0.0
        ri_norm = ri[..., :3].copy()
        ri_norm[..., 0] = np.clip(ri_norm[..., 0], 0.0, LIDAR_RANGE_MAX)  / LIDAR_RANGE_MAX
        ri_norm[..., 1] = np.clip(ri_norm[..., 1], 0.0, LIDAR_INTENS_MAX) / LIDAR_INTENS_MAX
        ri_norm[..., 2] = np.clip(ri_norm[..., 2], 0.0, LIDAR_ELONG_MAX)  / LIDAR_ELONG_MAX

        out = np.zeros((3, size, size), dtype=np.float32)

        us_all, vs_all, r_all, val_all = [], [], [], []
        for mask, u_slot, v_slot in _camera_hit_slots(proj, cam_id):
            m = mask & valid
            if not m.any():
                continue
            us_all.append(proj[..., u_slot][m])
            vs_all.append(proj[..., v_slot][m])
            r_all.append(ri[..., 0][m])
            val_all.append(ri_norm[m])

        if not us_all:
            return out   # no LiDAR returns project onto this camera this frame

        u    = np.concatenate(us_all)
        v    = np.concatenate(vs_all)
        r    = np.concatenate(r_all)
        vals = np.concatenate(val_all)   # [N, 3]

        us = np.clip((u * size / orig_w).astype(np.int64), 0, size - 1)
        vs = np.clip((v * size / orig_h).astype(np.int64), 0, size - 1)

        # Farthest-first so the nearest return overwrites at shared pixels
        order = np.argsort(r)[::-1]
        out[:, vs[order], us[order]] = vals[order].T
        return out
    except Exception as e:
        log.debug(f"scatter_lidar_to_camera error: {e}")
        return None


def parse_seg3d_label_for_camera(
    seg_values, seg_shape, proj_values, proj_shape,
    cam_id: int, orig_w: int, orig_h: int, size: int = IMG_SIZE,
) -> Optional[np.ndarray]:
    """
    Scatter Waymo lidar_segmentation semantic labels into this camera's pixel
    grid via the same LiDAR→camera projection as scatter_lidar_to_camera(),
    so the label and the LiDAR/image tensors stay pixel-aligned.

    Waymo semantic label IDs 1–22 → our 0-based indices 0–21.
    Pixels with no projected LiDAR return → SEG3D_IGNORE (255).
    Output: int16 [size, size]. Returns None on error.
    """
    try:
        dims = _parse_shape(seg_shape)
        if dims is None:
            n = len(seg_values)
            dims = [64, n // 64]   # lidar_seg shape is [H, W] (no channel dim)
            log.debug(f"parse_seg3d_label_for_camera: shape not parsed, inferred {dims}")
        # Waymo v2 lidar_segmentation stores [H, W, 2]:
        #   channel 0 = instance ID, channel 1 = semantic class (1–22; 0 = unlabelled)
        arr = np.array(seg_values, dtype=np.int32).reshape(dims)
        sem = arr[..., 1] if arr.ndim == 3 else arr   # [H, W] semantic channel
        H, W = sem.shape

        proj = _parse_projection(proj_values, proj_shape, H, W)
        if proj is None:
            return None

        out = np.full((size, size), SEG3D_IGNORE, dtype=np.int16)

        for mask, u_slot, v_slot in _camera_hit_slots(proj, cam_id):
            u        = proj[..., u_slot][mask]
            v        = proj[..., v_slot][mask]
            labels   = np.clip(sem[mask], 0, 255).astype(np.uint8)
            remapped = _SEG3D_LUT[labels]

            us = np.clip((u * size / orig_w).astype(np.int64), 0, size - 1)
            vs = np.clip((v * size / orig_h).astype(np.int64), 0, size - 1)
            out[vs, us] = remapped

        return out
    except Exception as e:
        log.debug(f"parse_seg3d_label_for_camera error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# BOUNDING BOX PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_det3d_boxes(df: pd.DataFrame, col: dict) -> torch.Tensor:
    """
    Build a [N, 8] float32 tensor from a lidar_box DataFrame slice.
    Columns: (cx, cy, cz, size_x, size_y, size_z, heading, cls_idx).
    Returns Tensor[0, 8] when no valid boxes.
    """
    required = ["box_cx", "box_cy", "box_cz", "box_lx", "box_ly", "box_lz",
                "box_heading", "box_type"]
    if df.empty or any(col.get(k) is None for k in required):
        return torch.zeros(0, 8, dtype=torch.float32)

    boxes = []
    for _, row in df.iterrows():
        cls_idx = DET3D_TYPE_MAP.get(int(row[col["box_type"]]), -1)
        if cls_idx < 0:
            continue
        boxes.append([
            float(row[col["box_cx"]]),
            float(row[col["box_cy"]]),
            float(row[col["box_cz"]]),
            float(row[col["box_lx"]]),
            float(row[col["box_ly"]]),
            float(row[col["box_lz"]]),
            float(row[col["box_heading"]]),
            float(cls_idx),
        ])
    if not boxes:
        return torch.zeros(0, 8, dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def filter_det3d_boxes_for_camera(
    boxes_t: torch.Tensor, proj_values, proj_shape, cam_id: int,
) -> torch.Tensor:
    """
    Keep only det3d boxes whose ego-frame center azimuth falls within the
    azimuth span of range-image columns that project onto cam_id — derived
    directly from the projection map, no hardcoded camera FOV.

    boxes_t: [N, 8] float32 (cx,cy,cz,sx,sy,sz,heading,cls). Returns [K, 8].
    """
    if boxes_t.shape[0] == 0:
        return boxes_t
    try:
        dims = _parse_shape(proj_shape)
        proj = np.array(proj_values, dtype=np.float32).reshape(dims)   # [H, W, 6]
        W    = proj.shape[1]

        col_hit = ((proj[..., 0].astype(np.int32) == cam_id) |
                   (proj[..., 3].astype(np.int32) == cam_id)).any(axis=0)   # [W]
        cam_cols = np.where(col_hit)[0]
        if cam_cols.size == 0:
            return torch.zeros(0, 8, dtype=torch.float32)

        azimuths = np.linspace(np.pi, -np.pi, W, endpoint=False, dtype=np.float32)
        az_lo, az_hi = float(azimuths[cam_cols].min()), float(azimuths[cam_cols].max())

        boxes_np = boxes_t.numpy()
        box_az   = np.arctan2(boxes_np[:, 1], boxes_np[:, 0])
        keep     = (box_az >= az_lo) & (box_az <= az_hi)
        return torch.from_numpy(boxes_np[keep].copy())
    except Exception as e:
        log.debug(f"filter_det3d_boxes_for_camera error: {e}")
        return boxes_t


def parse_det2d_boxes(df: pd.DataFrame, col: dict,
                      img_w: int = 1920, img_h: int = 1280) -> torch.Tensor:
    """
    Build a [M, 5] float32 tensor from a camera_box DataFrame slice.
    Columns: (cx_norm, cy_norm, w_norm, h_norm, cls_idx).
    Coordinates normalised by the original (pre-resize) image dimensions.
    Returns Tensor[0, 5] when no valid boxes.

    NOTE: Waymo camera_box stores centre-format boxes. The column names for
    width/height differ between dataset versions — run --discover to verify.
    """
    required = ["cbox_cx", "cbox_cy", "cbox_w", "cbox_h", "cbox_type"]
    if df.empty or any(col.get(k) is None for k in required):
        return torch.zeros(0, 5, dtype=torch.float32)

    boxes = []
    for _, row in df.iterrows():
        cls_idx = DET2D_TYPE_MAP.get(int(row[col["cbox_type"]]), -1)
        if cls_idx < 0:
            continue
        boxes.append([
            float(row[col["cbox_cx"]]) / img_w,
            float(row[col["cbox_cy"]]) / img_h,
            float(row[col["cbox_w"]])  / img_w,
            float(row[col["cbox_h"]])  / img_h,
            float(cls_idx),
        ])
    if not boxes:
        return torch.zeros(0, 5, dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# PANOPTIC LABEL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_pan2d_label(
    png_bytes: bytes,
    divisor:   int,
    size:      int = IMG_SIZE,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Decode a Waymo camera_segmentation panoptic PNG.

    Waymo stores a uint32 panoptic label where:
        semantic_id = panoptic_value // divisor
        instance_id = panoptic_value  % divisor   (0 = stuff / no instance)

    Returns (semantic_arr, instance_arr) both int32 [size, size], or None.
    semantic_arr : 0-27 remapped classes, 255 = ignore
    instance_arr : raw Waymo instance IDs (0 = stuff; >0 = trackable things)
    """
    try:
        pan_img  = Image.open(io.BytesIO(png_bytes))
        pan_arr  = np.array(pan_img, dtype=np.int32)   # uint32 stored as int32
        sem_arr  = pan_arr // divisor                   # semantic label per pixel
        inst_arr = pan_arr % divisor                    # instance ID per pixel

        # Remap semantic labels via LUT
        sem_clipped = np.clip(sem_arr, 0, 255).astype(np.uint8)
        sem_out     = _PAN2D_LUT[sem_clipped]           # int32 [H, W]

        # Nearest-neighbour resize — PIL mode 'I' preserves int32 exactly
        sem_out  = np.array(
            Image.fromarray(sem_out,  mode="I").resize((size, size), Image.Resampling.NEAREST),
            dtype=np.int32,
        )
        inst_out = np.array(
            Image.fromarray(inst_arr, mode="I").resize((size, size), Image.Resampling.NEAREST),
            dtype=np.int32,
        )
        return sem_out, inst_out
    except Exception as e:
        log.debug(f"parse_pan2d_label error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PER-SEGMENT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_segment(seg_id: str, split: str, out_dir: Path) -> bool:
    """
    Download, decode, and save one segment as one .pt file PER CAMERA.

    Returns True on success, False on failure.
    Output files are atomic (written to a .tmp path, then renamed).
    """
    out_paths = {cam_id: out_dir / f"{seg_id}_cam{cam_id}.pt" for cam_id in CAMERAS}
    if all(p.exists() for p in out_paths.values()):
        log.info(f"SKIP  {seg_id[:36]}… (all cameras already exist)")
        return True

    log.info(f"START {seg_id[:36]}…")

    # ── 1. Load all component DataFrames from GCS (only needed columns) ──────────
    dfs = {}
    for key, comp in COMPONENTS.items():
        path = _gcs_path(split, comp, seg_id)
        optional = key in ("camera_seg", "camera_box")
        try:
            dfs[key] = read_parquet_from_gcs(path, columns=NEEDED_COLS[key])
        except Exception as e:
            if optional:
                log.debug(f"  Optional {comp} not found — {e}")
                dfs[key] = None
            else:
                log.warning(f"  Required {comp} missing for {seg_id[:24]}: {e}")
                return False

    # ── 2. Resolve column names from actual schemas ───────────────────────────
    col_cam    = resolve_columns(dfs["camera_image"].columns.tolist())
    col_lidar  = resolve_columns(dfs["lidar"].columns.tolist())
    col_proj   = resolve_columns(dfs["lidar_camera_proj"].columns.tolist())
    col_seg    = resolve_columns(dfs["lidar_seg"].columns.tolist())
    col_box3d  = resolve_columns(dfs["lidar_box"].columns.tolist())
    col_camseg = resolve_columns(dfs["camera_seg"].columns.tolist()) if dfs["camera_seg"] is not None else {}
    col_box2d  = resolve_columns(dfs["camera_box"].columns.tolist()) if dfs["camera_box"] is not None else {}

    # ── 3. Enumerate frames (by unique timestamps in the lidar component) ──────
    frame_ids = sorted(dfs["lidar"][FRAME_KEY].unique().tolist())

    frames_by_cam: dict = {cam_id: [] for cam_id in CAMERAS}

    for fid in frame_ids:

        # ── TOP LiDAR range image (shared across all cameras) ─────────────────
        lidar_rows = dfs["lidar"][
            (dfs["lidar"][FRAME_KEY] == fid) &
            (dfs["lidar"][LIDAR_KEY] == TOP_LIDAR)
        ]
        if lidar_rows.empty:
            continue   # no TOP LiDAR — skip frame
        if col_lidar.get("ri_values") is None or col_lidar.get("ri_shape") is None:
            log.warning(f"  Frame {fid}: lidar range-image columns unresolved — skipping frame.")
            continue
        lidar_row = lidar_rows.iloc[0]
        ri_values = lidar_row[col_lidar["ri_values"]]
        ri_shape  = lidar_row[col_lidar["ri_shape"]]

        # ── LiDAR→camera projection map (shared across all cameras) ───────────
        proj_rows = dfs["lidar_camera_proj"][
            (dfs["lidar_camera_proj"][FRAME_KEY] == fid) &
            (dfs["lidar_camera_proj"][LIDAR_KEY] == TOP_LIDAR)
        ]
        if proj_rows.empty or col_proj.get("proj_values") is None or col_proj.get("proj_shape") is None:
            log.debug(f"  Frame {fid}: no camera projection data — skipping frame.")
            continue
        proj_row    = proj_rows.iloc[0]
        proj_values = proj_row[col_proj["proj_values"]]
        proj_shape  = proj_row[col_proj["proj_shape"]]

        # ── seg3d range-image label (shared; scattered per camera below) ──────
        seg_rows = dfs["lidar_seg"][
            (dfs["lidar_seg"][FRAME_KEY] == fid) &
            (dfs["lidar_seg"][LIDAR_KEY] == TOP_LIDAR)
        ]
        has_seg3d = (not seg_rows.empty
                     and col_seg.get("seg_values") is not None
                     and col_seg.get("seg_shape") is not None)
        seg_row = seg_rows.iloc[0] if has_seg3d else None

        # ── det3d boxes (shared; FOV-filtered per camera below) ───────────────
        box3d_rows  = dfs["lidar_box"][dfs["lidar_box"][FRAME_KEY] == fid]
        det3d_all_t = parse_det3d_boxes(box3d_rows, col_box3d)   # float32 [N,8]

        # ── per-camera processing ──────────────────────────────────────────────
        for cam_id in CAMERAS:
            cam_rows = dfs["camera_image"][
                (dfs["camera_image"][FRAME_KEY] == fid) &
                (dfs["camera_image"][CAM_KEY]   == cam_id)
            ]
            if cam_rows.empty or col_cam.get("cam_image") is None:
                continue   # this camera has no image for this frame — skip
            cam_row = cam_rows.iloc[0]

            # decode_and_resize_jpeg returns (array, orig_w, orig_h)
            img_arr, img_w, img_h = decode_and_resize_jpeg(bytes(cam_row[col_cam["cam_image"]]))
            image_t = torch.from_numpy(img_arr).to(torch.float16)   # [3,224,224]

            lidar_arr = scatter_lidar_to_camera(
                ri_values, ri_shape, proj_values, proj_shape, cam_id, img_w, img_h)
            if lidar_arr is None:
                continue
            lidar_t = torch.from_numpy(lidar_arr).to(torch.float16)  # [3,224,224]

            seg3d_t = None
            if has_seg3d:
                seg_arr = parse_seg3d_label_for_camera(
                    seg_row[col_seg["seg_values"]], seg_row[col_seg["seg_shape"]],
                    proj_values, proj_shape, cam_id, img_w, img_h)
                if seg_arr is not None:
                    seg3d_t = torch.from_numpy(seg_arr)  # int16 [224,224]

            det3d_t = filter_det3d_boxes_for_camera(
                det3d_all_t, proj_values, proj_shape, cam_id)   # float32 [K,8]

            # ── pan2d semantic label + instance IDs ───────────────────────────
            pan2d_t          = None
            pan2d_instance_t = None
            if dfs["camera_seg"] is not None and col_camseg.get("pan_label") and col_camseg.get("pan_divisor"):
                camseg_rows = dfs["camera_seg"][
                    (dfs["camera_seg"][FRAME_KEY] == fid) &
                    (dfs["camera_seg"][CAM_KEY]   == cam_id)
                ]
                if not camseg_rows.empty:
                    cs_row = camseg_rows.iloc[0]
                    result = parse_pan2d_label(bytes(cs_row[col_camseg["pan_label"]]),
                                               int(cs_row[col_camseg["pan_divisor"]]))
                    if result is not None:
                        sem_arr, inst_arr = result
                        pan2d_t          = torch.from_numpy(sem_arr)   # int32 [224,224]
                        pan2d_instance_t = torch.from_numpy(inst_arr)  # int32 [224,224]

            # ── det2d boxes ────────────────────────────────────────────────────
            det2d_t = None
            if dfs["camera_box"] is not None:
                cambox_rows = dfs["camera_box"][
                    (dfs["camera_box"][FRAME_KEY] == fid) &
                    (dfs["camera_box"][CAM_KEY]   == cam_id)
                ]
                det2d_t = parse_det2d_boxes(cambox_rows, col_box2d, img_w, img_h)

            frames_by_cam[cam_id].append({
                "frame_id":       int(fid),
                "image":          image_t,
                "lidar":          lidar_t,
                "seg3d_label":    seg3d_t,
                "det3d_boxes":    det3d_t,
                "pan2d_label":    pan2d_t,
                "pan2d_instance": pan2d_instance_t,
                "det2d_boxes":    det2d_t,
            })

    # ── free DataFrames — they can be 1–2 GB total; don't wait for GC ───────────
    del dfs
    gc.collect()

    # ── 4. Atomic save, one file per camera ────────────────────────────────────
    n_saved = 0
    for cam_id, frames in frames_by_cam.items():
        if not frames:
            continue
        out_path = out_paths[cam_id]
        payload  = {"segment_id": seg_id, "camera_id": cam_id, "split": split, "frames": frames}
        tmp_path = out_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp_path)
        tmp_path.rename(out_path)
        log.info(f"SAVED {seg_id[:36]} cam{cam_id} — {len(frames)} frames → {out_path.name}")
        n_saved += 1

    del frames_by_cam
    gc.collect()

    if n_saved == 0:
        log.warning(f"  {seg_id[:36]}: 0 valid frames for any camera — not saved")
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Waymo v2 GCS parquets → .pt files for FedMMA"
    )
    parser.add_argument("--split",    default="training",
                        choices=["training", "validation"],
                        help="Dataset split to process (default: training)")
    parser.add_argument("--out_dir",  default=None,
                        help="Root output directory (required unless --discover)")
    parser.add_argument("--workers",  type=int, default=1,
                        help="Parallel worker processes (default: 1)")
    parser.add_argument("--max_segs", type=int, default=None,
                        help="Stop after N segments (smoke-test)")
    parser.add_argument("--seg_list", default=None,
                        help="Path to a file with one segment ID per line")
    parser.add_argument("--discover", action="store_true",
                        help="Print column schemas for all components and exit")
    args = parser.parse_args()

    # ── Schema discovery mode ─────────────────────────────────────────────────
    if args.discover:
        print(f"\nSchema discovery — split='{args.split}'\n{'─'*60}")
        for key, comp in COMPONENTS.items():
            cols = discover_schema(args.split, comp)
            print(f"\n[{key}]  {comp}  ({len(cols)} columns)")
            for c in cols:
                print(f"    {c}")
        print("\nDone.  Update resolve_columns() if any column names differ.")
        return

    if args.out_dir is None:
        parser.error("--out_dir is required (or use --discover)")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine segment list ────────────────────────────────────────────────
    if args.seg_list:
        with open(args.seg_list) as fh:
            seg_ids = [l.strip() for l in fh if l.strip()]
        log.info(f"Using {len(seg_ids)} segments from {args.seg_list}")
    else:
        # Only process segments that have BOTH lidar_seg AND camera_seg
        # (the ~696 qualifying segments that support all four tasks)
        log.info("Discovering qualifying segments (lidar_segmentation ∩ camera_segmentation)…")
        lidar_seg_set  = set(segment_ids_for_component(args.split, "lidar_segmentation"))
        camera_seg_set = set(segment_ids_for_component(args.split, "camera_segmentation"))
        seg_ids = sorted(lidar_seg_set & camera_seg_set)
        log.info(
            f"  lidar_seg={len(lidar_seg_set)}  "
            f"camera_seg={len(camera_seg_set)}  "
            f"intersection={len(seg_ids)}"
        )

    if args.max_segs:
        seg_ids = seg_ids[: args.max_segs]
        log.info(f"Limited to {len(seg_ids)} segment(s) via --max_segs")

    log.info(f"Processing {len(seg_ids)} segment(s) → {out_dir}")

    # ── Process ───────────────────────────────────────────────────────────────
    if args.workers == 1:
        # Single-process — simpler debugging
        results = [preprocess_segment(sid, args.split, out_dir) for sid in seg_ids]
    else:
        from functools import partial
        fn = partial(preprocess_segment, split=args.split, out_dir=out_dir)
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fn, sid): sid for sid in seg_ids}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    log.error(f"Worker exception on {sid[:24]}: {exc}")
                    results.append(False)

    n_ok   = sum(results)
    n_fail = len(results) - n_ok
    log.info(f"\n{'─'*60}")
    log.info(f"Finished: {n_ok}/{len(seg_ids)} succeeded, {n_fail} failed.")
    if n_fail:
        log.info("Re-run to retry failed segments — existing .pt files are skipped.")


if __name__ == "__main__":
    main()
