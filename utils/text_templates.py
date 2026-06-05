"""
Class vocabularies, text prompt templates, embedding slice indices,
and modality assignments for all four FedMMA tasks.
"""

# ── Task list ─────────────────────────────────────────────────────────────────

TASKS = ["seg3d", "det3d", "pan2d", "det2d"]

# ── Class vocabularies ────────────────────────────────────────────────────────
# List index == integer label used by task heads and loss functions.
# Order must match the remapping tables in scripts/preprocess_waymo.py.

TASK_CLASSES = {
    # 22 classes — matches Waymo lidar_segmentation TYPE_CAR(1)…TYPE_SIDEWALK(22)
    "seg3d": [
        "car", "truck", "bus", "other vehicle", "motorcyclist", "bicyclist",
        "pedestrian", "sign", "traffic light", "pole", "construction cone",
        "bicycle", "motorcycle", "building", "vegetation", "tree trunk",
        "curb", "road", "lane marker", "other ground", "walkable", "sidewalk",
    ],
    # 4 classes — matches Waymo lidar_box type enum (1→0 … 4→3)
    "det3d": [
        "vehicle", "pedestrian", "sign", "cyclist",
    ],
    # 28 classes — matches Waymo camera_segmentation SemanticLabel (remapped in preprocessing)
    "pan2d": [
        "car", "bus", "truck", "other large vehicle", "trailer", "ego vehicle",
        "motorcycle", "bicycle", "pedestrian", "cyclist", "motorcyclist",
        "ground animal", "bird", "pole", "sign", "traffic light",
        "construction cone", "pedestrian object", "building", "road",
        "sidewalk", "road marker", "lane marker", "vegetation", "sky",
        "ground", "static", "dynamic",
    ],
    # 3 classes — matches Waymo camera_box type enum (1→0, 2→1, 4→2)
    "det2d": [
        "vehicle", "pedestrian", "cyclist",
    ],
}

# ── Text prompt templates ─────────────────────────────────────────────────────
# {} is replaced with each class name before encoding with CLIP's text encoder.

TASK_TEMPLATES = {
    "seg3d": "a 3d semantic segmentation of class {}.",
    "det3d": "a 3d object detection of class {}.",
    "pan2d": "a panoptic segmentation of class {}.",
    "det2d": "a 2d object detection of class {}.",
}

# ── Text embedding slice indices ──────────────────────────────────────────────
# The text encoder produces a flat [57, 512] matrix (22+4+28+3 = 57 classes).
# Each task head retrieves its classifier weights via text_embs[TEXT_SLICES[task]].

TEXT_SLICES = {
    "seg3d": slice(0,  22),   # indices  0–21
    "det3d": slice(22, 26),   # indices 22–25
    "pan2d": slice(26, 54),   # indices 26–53
    "det2d": slice(54, 57),   # indices 54–56
}

# ── Task modality assignments ─────────────────────────────────────────────────
# Determines whether real LiDAR or null_lidar is passed to the fusion step.

TASK_MODALITIES = {
    "seg3d": ("image", "lidar", "text"),
    "det3d": ("image", "lidar", "text"),
    "pan2d": ("image", "text"),           # camera-only — null_lidar injected by trainer
    "det2d": ("image", "text"),           # camera-only — null_lidar injected by trainer
}


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total = sum(len(v) for v in TASK_CLASSES.values())
    print(f"Total classes: {total}")      # must be 57
    for task, sl in TEXT_SLICES.items():
        n = len(TASK_CLASSES[task])
        assert sl.stop - sl.start == n, (
            f"{task}: slice width {sl.stop - sl.start} != {n} classes"
        )
        print(f"  {task}: {n} classes → slice({sl.start}, {sl.stop}) ✓")
