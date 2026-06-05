"""
trainers/task_heads.py
-----------------------
Four task-specific prediction heads for FedMMA multi-task learning on Waymo.

Architecture
------------
SegHead3D  — bilinear-upsample seg head for LiDAR semantic segmentation (22 cls)
PanHead2D  — bilinear-upsample seg head for camera panoptic segmentation (28 cls)
DetHead3D  — DETR-style head for 3-D LiDAR object detection   (4 cls, N=200)
DetHead2D  — DETR-style head for 2-D camera object detection  (3 cls, N=100)

Input convention (all heads)
----------------------------
patch_tokens : Tensor[B, 196, 512]
    Fused patch-token output from CustomCLIP (14×14 spatial grid,
    CLIP ViT-B/16 output_dim=512).

Segmentation heads (SegHead3D, PanHead2D)
-----------------------------------------
1. Reshape patch tokens → [B, 512, 14, 14]
2. Bilinear upsample    → [B, 512, 224, 224]
3. L2-normalise features along channel dim
4. Cosine similarity with frozen text class embeddings → [B, C, 224, 224]
5. Scale by learnable temperature (log_scale, init=0 → scale=1)

Detection heads (DetHead3D, DetHead2D)
--------------------------------------
1. N learned object queries cross-attend to 196 patch tokens (2 decoder layers)
2. Cosine similarity with frozen text embeddings + learnable background token
   → [B, N, num_classes+1]  class logits
3. Linear projection → box regression
   DetHead3D : (cx, cy, cz, sx, sy, sz, sin θ, cos θ)   — absolute ego-frame metres
   DetHead2D : sigmoid((cx_n, cy_n, w_n, h_n))           — normalised [0, 1]

Loss functions
--------------
Segmentation : F.cross_entropy  (ignore_index = 255)
Detection    : Hungarian matching (scipy) + sigmoid focal loss (cls) + L1 (box)

Usage
-----
    from trainers.task_heads import build_task_heads

    heads = build_task_heads(text_embs)          # text_embs: [57, 512]
    logits = heads["seg3d"](patch_tokens)        # [B, 22, 224, 224]
    loss   = heads["seg3d"].compute_loss(logits, seg_labels)

    cls_logits, box_preds = heads["det3d"](patch_tokens)
    targets = [{"labels": gt_cls, "boxes": DetHead3D.encode_boxes(gt_raw)} ...]
    losses  = heads["det3d"].compute_loss(cls_logits, box_preds, targets)
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ── Architecture constants ─────────────────────────────────────────────────────

IMG_SIZE   = 224          # spatial resolution of labels / upsample target
PATCH_SIZE = 16           # ViT-B/16 patch size
GRID       = IMG_SIZE // PATCH_SIZE   # 14 — patch grid side
CLIP_DIM   = 512          # ViT-B/16 output_dim / token width
SEG_IGNORE = 255          # ignore index in segmentation label tensors


# ── Shared cosine-similarity classifier ───────────────────────────────────────

def _cosine_logits(
    features:  torch.Tensor,   # [B, D, H, W]  or  [B, N, D]
    text_embs: torch.Tensor,   # [C, D]
    log_scale: nn.Parameter,   # scalar — temperature is exp(log_scale)
    spatial:   bool = True,
) -> torch.Tensor:
    """
    Cosine similarity between normalised features and class text embeddings,
    scaled by exp(log_scale).  Always runs in fp32.

    spatial=True  : input [B, D, H, W] → output [B, C, H, W]
    spatial=False : input [B, N, D]    → output [B, N, C]
    """
    scale = log_scale.exp()
    if spatial:
        feat = F.normalize(features.float(), dim=1)    # [B, D, H, W]
        cls  = F.normalize(text_embs.float(), dim=-1)  # [C, D]
        return torch.einsum("bdhw,cd->bchw", feat, cls) * scale
    else:
        feat = F.normalize(features.float(), dim=-1)   # [B, N, D]
        cls  = F.normalize(text_embs.float(), dim=-1)  # [C, D]
        return torch.einsum("bnd,cd->bnc", feat, cls) * scale


# ══════════════════════════════════════════════════════════════════════════════
# Segmentation heads
# ══════════════════════════════════════════════════════════════════════════════

class _SegHead(nn.Module):
    """
    Base bilinear-upsample segmentation head.

    Shared implementation; SegHead3D and PanHead2D differ only in the number
    of classes and the text_embs slice passed at construction time.
    """

    def __init__(self, text_embs: torch.Tensor, num_classes: int):
        """
        Args:
            text_embs   : [num_classes, 512] float32 — frozen class embeddings
                          from the appropriate TEXT_SLICES slice.
            num_classes : semantic class count (22 for seg3d, 28 for pan2d).
        """
        super().__init__()
        if text_embs.shape != (num_classes, CLIP_DIM):
            raise ValueError(
                f"Expected text_embs [{num_classes}, {CLIP_DIM}], "
                f"got {tuple(text_embs.shape)}"
            )
        self.register_buffer("text_embs", text_embs)   # frozen, [C, 512]
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens : [B, 196, 512]
        Returns:
            logits : [B, num_classes, 224, 224]
        """
        B = patch_tokens.shape[0]
        # Reshape flat token sequence to spatial grid
        x = patch_tokens.permute(0, 2, 1).reshape(B, CLIP_DIM, GRID, GRID)
        # Upsample to label resolution
        x = F.interpolate(
            x.float(), size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False,
        )
        return _cosine_logits(x, self.text_embs, self.log_scale, spatial=True)

    def compute_loss(
        self,
        logits: torch.Tensor,   # [B, C, 224, 224]
        labels: torch.Tensor,   # [B, 224, 224]  int  (0…C-1; 255 = ignore)
    ) -> torch.Tensor:
        """Cross-entropy loss; pixels labelled 255 are masked out."""
        return F.cross_entropy(logits, labels.long(), ignore_index=SEG_IGNORE)


class SegHead3D(_SegHead):
    """
    LiDAR semantic segmentation head — 22 classes.

    Matches Waymo lidar_segmentation TYPE_CAR(1) … TYPE_SIDEWALK(22),
    remapped to 0-based indices by preprocess_waymo.py.
    text_embs slice : TEXT_SLICES["seg3d"] = slice(0, 22)
    """

    NUM_CLASSES = 22

    def __init__(self, text_embs: torch.Tensor):
        super().__init__(text_embs, self.NUM_CLASSES)


class PanHead2D(_SegHead):
    """
    Camera panoptic segmentation head — 28 classes.

    Matches Waymo camera_segmentation SemanticLabel remapped by preprocess_waymo.py.
    text_embs slice : TEXT_SLICES["pan2d"] = slice(26, 54)
    """

    NUM_CLASSES = 28

    def __init__(self, text_embs: torch.Tensor):
        super().__init__(text_embs, self.NUM_CLASSES)


# ══════════════════════════════════════════════════════════════════════════════
# Detection heads
# ══════════════════════════════════════════════════════════════════════════════

class _DetHead(nn.Module):
    """
    Base DETR-style detection head.

    N learned object queries attend cross-attention to 196 CLIP patch tokens
    through a standard transformer decoder (num_decoder_layers=2 by default).
    Predictions are made per query:
      - Class logits via cosine similarity (text classes + background token)
      - Box regression via a linear head

    Loss
    ----
    Training uses Hungarian matching (scipy.optimize.linear_sum_assignment)
    followed by:
      - Sigmoid focal loss for classification (alpha=0.25, gamma=2.0)
      - L1 loss for box regression (weight _LAMBDA_BOX=5.0)
    Unmatched queries are assigned the background class.
    """

    _FOCAL_ALPHA = 0.25
    _FOCAL_GAMMA = 2.0
    _LAMBDA_BOX  = 5.0   # box loss coefficient relative to cls loss

    def __init__(
        self,
        text_embs:          torch.Tensor,
        num_classes:        int,
        num_queries:        int,
        box_dim:            int,
        num_decoder_layers: int   = 2,
        nhead:              int   = 8,
        dim_feedforward:    int   = 1024,
        dropout:            float = 0.1,
    ):
        """
        Args:
            text_embs          : [num_classes, 512] frozen class embeddings.
            num_classes        : foreground class count (background added internally).
            num_queries        : number of learned object queries (N).
            box_dim            : regression output size (8 for 3-D, 4 for 2-D).
            num_decoder_layers : transformer decoder depth (default 2).
            nhead              : attention heads in each decoder layer.
            dim_feedforward    : FFN hidden dim in each decoder layer.
            dropout            : dropout rate in transformer decoder.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.box_dim     = box_dim
        self._is_2d      = (box_dim == 4)

        # ── Frozen text embeddings + learnable background token ────────────
        self.register_buffer("text_embs", text_embs)          # [C, 512]
        self.bg_embed  = nn.Parameter(torch.randn(1, CLIP_DIM) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(0.0))

        # ── Learned object queries ─────────────────────────────────────────
        self.query_embed = nn.Embedding(num_queries, CLIP_DIM)

        # ── Transformer decoder ────────────────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model         = CLIP_DIM,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=num_decoder_layers)

        # ── Box regression head ────────────────────────────────────────────
        self.box_head = nn.Linear(CLIP_DIM, box_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.query_embed.weight, -0.1, 0.1)
        nn.init.zeros_(self.box_head.bias)
        nn.init.normal_(self.box_head.weight, std=0.01)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens : [B, 196, 512]
        Returns:
            cls_logits : [B, N, num_classes + 1]
                         Class dim: indices 0…num_classes-1 = foreground classes,
                                    index num_classes = background.
            box_preds  : [B, N, box_dim]
                         DetHead3D → (cx, cy, cz, sx, sy, sz, sinθ, cosθ) in metres
                         DetHead2D → (cx_n, cy_n, w_n, h_n) ∈ (0, 1)
        """
        B = patch_tokens.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)   # [B,N,512]
        out     = self.decoder(tgt=queries, memory=patch_tokens)            # [B,N,512]

        all_embs   = torch.cat([self.text_embs, self.bg_embed], dim=0)     # [C+1,512]
        cls_logits = _cosine_logits(out, all_embs, self.log_scale,
                                    spatial=False)                          # [B,N,C+1]
        box_preds  = self.box_head(out)                                     # [B,N,box_dim]
        if self._is_2d:
            box_preds = box_preds.sigmoid()

        return cls_logits, box_preds

    # ── Loss ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _match(
        self,
        cls_logits: torch.Tensor,   # [N, C+1]
        box_preds:  torch.Tensor,   # [N, box_dim]
        gt_cls:     torch.Tensor,   # [M]
        gt_box:     torch.Tensor,   # [M, box_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hungarian matching for one image.

        Cost = -softmax_prob[gt_class]  +  λ * L1(box).

        Returns:
            pred_indices : LongTensor [K]  matched query indices
            gt_indices   : LongTensor [K]  corresponding gt indices  (K ≤ M)
        """
        M = gt_cls.shape[0]
        if M == 0:
            empty = torch.zeros(0, dtype=torch.long, device=cls_logits.device)
            return empty, empty

        prob   = cls_logits.softmax(-1)                            # [N, C+1]
        cost_c = -prob[:, gt_cls]                                  # [N, M]
        cost_b = torch.cdist(box_preds.float(), gt_box.float(), p=1)  # [N, M]
        cost   = cost_c + self._LAMBDA_BOX * cost_b               # [N, M]

        rows, cols = linear_sum_assignment(cost.cpu().numpy())
        return (
            torch.as_tensor(rows, dtype=torch.long, device=cls_logits.device),
            torch.as_tensor(cols, dtype=torch.long, device=gt_cls.device),
        )

    def compute_loss(
        self,
        cls_logits: torch.Tensor,
        box_preds:  torch.Tensor,
        targets:    List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss for one batch.

        Args:
            cls_logits : [B, N, num_classes+1]
            box_preds  : [B, N, box_dim]
            targets    : list of B dicts, each containing:
                "labels" : LongTensor [M]           — class indices 0…num_classes-1
                "boxes"  : FloatTensor [M, box_dim] — ground-truth boxes
                    DetHead3D → DetHead3D.encode_boxes() output  (sin/cos heading)
                    DetHead2D → normalised (cx_n, cy_n, w_n, h_n) ∈ [0,1]

        Returns:
            {
              "loss_cls" : mean focal classification loss,
              "loss_box" : mean L1 box regression loss  (matched pairs only),
              "loss"     : loss_cls + _LAMBDA_BOX * loss_box,
            }
        """
        bg_idx = self.num_classes
        cls_losses, box_losses = [], []

        for b in range(cls_logits.shape[0]):
            log_b    = cls_logits[b]            # [N, C+1]
            box_b    = box_preds[b]             # [N, box_dim]
            gt_cls_b = targets[b]["labels"]     # [M]
            gt_box_b = targets[b]["boxes"]      # [M, box_dim]
            M        = gt_cls_b.shape[0]

            pred_i, gt_i = self._match(log_b, box_b, gt_cls_b, gt_box_b)

            # ── Classification loss ────────────────────────────────────────
            # Assign background to all queries; override matched ones.
            tgt_cls = torch.full((self.num_queries,), bg_idx,
                                 dtype=torch.long, device=log_b.device)
            if M > 0:
                tgt_cls[pred_i] = gt_cls_b[gt_i]

            # Sigmoid focal loss (binary per class, following Deformable DETR)
            one_hot = F.one_hot(tgt_cls, self.num_classes + 1).float()  # [N,C+1]
            p       = log_b.sigmoid()
            bce     = F.binary_cross_entropy_with_logits(
                          log_b, one_hot, reduction="none")
            p_t     = p * one_hot + (1 - p) * (1 - one_hot)
            alpha_t = (  self._FOCAL_ALPHA * one_hot
                       + (1 - self._FOCAL_ALPHA) * (1 - one_hot))
            focal   = alpha_t * (1 - p_t) ** self._FOCAL_GAMMA * bce
            cls_losses.append(focal.sum() / max(M, 1))

            # ── Box regression loss ────────────────────────────────────────
            if M > 0:
                box_losses.append(
                    F.l1_loss(box_b[pred_i], gt_box_b[gt_i].to(box_b))
                )
            else:
                box_losses.append(box_b.sum() * 0.0)   # zero grad, keeps graph

        loss_cls = torch.stack(cls_losses).mean()
        loss_box = torch.stack(box_losses).mean()
        return {
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss":     loss_cls + self._LAMBDA_BOX * loss_box,
        }


class DetHead3D(_DetHead):
    """
    LiDAR 3-D detection head — 4 classes (vehicle / pedestrian / sign / cyclist).

    Box regression: (cx, cy, cz, sx, sy, sz, sinθ, cosθ)  ego-frame metres.
    text_embs slice : TEXT_SLICES["det3d"] = slice(22, 26)
    """

    NUM_CLASSES = 4
    NUM_QUERIES = 200
    BOX_DIM     = 8   # cx cy cz sx sy sz sinθ cosθ

    def __init__(self, text_embs: torch.Tensor, num_decoder_layers: int = 2):
        super().__init__(
            text_embs          = text_embs,
            num_classes        = self.NUM_CLASSES,
            num_queries        = self.NUM_QUERIES,
            box_dim            = self.BOX_DIM,
            num_decoder_layers = num_decoder_layers,
        )

    @staticmethod
    def encode_boxes(det3d_boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert raw .pt det3d_boxes to regression targets.

        Input  : FloatTensor [M, 8]  — (cx, cy, cz, sx, sy, sz, heading_rad, cls)
        Output : FloatTensor [M, 8]  — (cx, cy, cz, sx, sy, sz, sinθ, cosθ)

        The class column (index 7) is dropped; extract labels separately via
            labels = det3d_boxes[:, 7].long()
        before calling this method.
        """
        h = det3d_boxes[:, 6]
        return torch.stack([
            det3d_boxes[:, 0], det3d_boxes[:, 1], det3d_boxes[:, 2],
            det3d_boxes[:, 3], det3d_boxes[:, 4], det3d_boxes[:, 5],
            h.sin(), h.cos(),
        ], dim=-1)   # [M, 8]


class DetHead2D(_DetHead):
    """
    Camera 2-D detection head — 3 classes (vehicle / pedestrian / cyclist).

    Box regression: (cx_norm, cy_norm, w_norm, h_norm) ∈ (0, 1).
    Coordinates are normalised by image dimensions, matching the .pt det2d_boxes
    format (cols 0-3); the class column (index 4) is extracted separately.
    text_embs slice : TEXT_SLICES["det2d"] = slice(54, 57)
    """

    NUM_CLASSES = 3
    NUM_QUERIES = 100
    BOX_DIM     = 4   # cx_norm cy_norm w_norm h_norm

    def __init__(self, text_embs: torch.Tensor, num_decoder_layers: int = 2):
        super().__init__(
            text_embs          = text_embs,
            num_classes        = self.NUM_CLASSES,
            num_queries        = self.NUM_QUERIES,
            box_dim            = self.BOX_DIM,
            num_decoder_layers = num_decoder_layers,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_task_heads(
    text_embs:          torch.Tensor,
    num_decoder_layers: int = 2,
) -> Dict[str, nn.Module]:
    """
    Construct all four task heads from the full [57, 512] text embedding matrix.

    Slices match TEXT_SLICES in utils/text_templates.py:
        seg3d  →  text_embs[ 0:22]   (22 classes)
        det3d  →  text_embs[22:26]   ( 4 classes)
        pan2d  →  text_embs[26:54]   (28 classes)
        det2d  →  text_embs[54:57]   ( 3 classes)

    Args:
        text_embs          : [57, 512] float32 — full task class embeddings.
                             Embeddings are detached before storing as buffers.
        num_decoder_layers : decoder depth shared by DetHead3D and DetHead2D.

    Returns:
        dict {"seg3d": SegHead3D, "det3d": DetHead3D,
              "pan2d": PanHead2D, "det2d": DetHead2D}

    Example::

        clip_model = load_clip_to_cpu(cfg)
        text_embs  = encode_task_text(clip_model)        # [57, 512]
        heads      = build_task_heads(text_embs)
        heads      = nn.ModuleDict(heads)                # register for state_dict
    """
    if text_embs.shape != (57, CLIP_DIM):
        raise ValueError(
            f"Expected text_embs [57, {CLIP_DIM}], got {tuple(text_embs.shape)}"
        )
    embs = text_embs.detach().float()
    return {
        "seg3d": SegHead3D(embs[ 0:22]),
        "det3d": DetHead3D(embs[22:26], num_decoder_layers),
        "pan2d": PanHead2D(embs[26:54]),
        "det2d": DetHead2D(embs[54:57], num_decoder_layers),
    }
