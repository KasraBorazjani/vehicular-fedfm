"""
models/task_heads.py
--------------------
Four task-specific prediction heads for FedMMA multi-task learning on Waymo.

Architecture
-----------------------
SegHead3D   — fused patch tokens → bilinear-upsample seg head (22 cls)
              Input: fused_patches [B, 196, 512] from ImageLidarFusion
              Loss: weighted CE + Lovász-Softmax

DetHead3D   — DETR-style 3D object detection head (4 cls)
              Input: fused_patches [B, 196, 512] from ImageLidarFusion
              TransformerDecoder + cosine-sim cls + raw box regression
              Loss: Hungarian + focal classification + L1 box regression
              Box parameterisation: (cx, cy, cz, sx, sy, sz, heading)

PanHead2D   — FPN-style dense panoptic head (28 cls)
              3-stage FPN decoder → semantic + center + offset + embedding branches
              Loss: CE (semantic) + focal (center) + masked L1 (offset) + contrastive

DetHead2D   — DETR-style 2-D camera detection head (3 cls, N=100)
              Per-query scalar no-object bias; GIoU + focal + L1 loss

Input convention
----------------
SegHead3D / DetHead3D :
    fused_patches  [B, 196, 512]  — from ImageLidarFusion (models/fusion.py)
    text_embs      [C, 512]

PanHead2D / DetHead2D :
    patch_tokens   [B, 196, 512]  or  [B, T, 196, 512] for pan2d
    text_embs      [C, 512]

Note: image/LiDAR fusion for 3D tasks is performed upstream in CustomCLIP via
ImageLidarFusion; the heads receive already-fused tokens.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou


# ── Architecture constants ─────────────────────────────────────────────────────

IMG_SIZE   = 224
PATCH_SIZE = 16
GRID       = IMG_SIZE // PATCH_SIZE   # 14
CLIP_DIM   = 512
SEG_IGNORE = 255


# ── Shared cosine-similarity classifier ───────────────────────────────────────

def _cosine_logits(
    features:  torch.Tensor,
    text_embs: torch.Tensor,
    log_scale: nn.Parameter,
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
        feat = F.normalize(features.float(), dim=1)
        cls  = F.normalize(text_embs.float(), dim=-1)
        return torch.einsum("bdhw,cd->bchw", feat, cls) * scale
    else:
        feat = F.normalize(features.float(), dim=-1)
        cls  = F.normalize(text_embs.float(), dim=-1)
        return torch.einsum("bnd,cd->bnc", feat, cls) * scale


# ── Lovász-Softmax loss ───────────────────────────────────────────────────────

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Lovász extension gradient for a sorted binary GT vector."""
    p   = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union        = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard      = 1.0 - intersection / union.clamp(min=1.0)
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_softmax(
    probas:       torch.Tensor,   # [B, C, H, W] softmax output
    labels:       torch.Tensor,   # [B, H, W] int
    ignore_index: int = 255,
) -> torch.Tensor:
    """Multi-class Lovász-Softmax loss (inlined; no external dependency)."""
    B, C, H, W = probas.shape
    probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, C)  # [BHW, C]
    labels_flat = labels.reshape(-1)                           # [BHW]

    valid = labels_flat != ignore_index
    if valid.sum() == 0:
        return probas.sum() * 0.0

    pv = probas_flat[valid]   # [P, C]
    lv = labels_flat[valid]   # [P]

    loss  = 0.0
    count = 0
    for c in range(C):
        fg = (lv == c).float()
        if fg.sum() == 0 and fg.numel() > 0:
            continue
        errors, perm = torch.sort(1.0 - pv[:, c], descending=True)
        grad = _lovasz_grad(fg[perm])
        loss  += (errors * grad).sum()
        count += 1

    return loss / max(count, 1)


# ── CenterPoint focal loss (Gaussian heatmap) — used by PanHead2D ─────────────

def _centerness_focal_loss(
    pred: torch.Tensor,   # [B, C, H, W] sigmoid probabilities
    gt:   torch.Tensor,   # [B, C, H, W] Gaussian heatmaps ∈ [0, 1]
    alpha: int = 2,
    beta:  int = 4,
) -> torch.Tensor:
    """CornerNet-style focal loss for heatmap supervision."""
    pos_mask = (gt == 1.0).float()
    neg_mask = 1.0 - pos_mask
    neg_weights = (1.0 - gt).pow(beta)

    loss_pos = pos_mask * (1.0 - pred).pow(alpha) * pred.clamp(min=1e-6).log()
    loss_neg = neg_mask * neg_weights * pred.pow(alpha) * (1.0 - pred).clamp(min=1e-6).log()

    num_pos = pos_mask.sum().clamp(min=1.0)
    return -(loss_pos.sum() + loss_neg.sum()) / num_pos


# ── Gaussian heatmap drawing — used by PanHead2D ─────────────────────────────

def _draw_gaussian(
    heatmap: torch.Tensor,   # [H, W] float32
    center:  Tuple[int, int],
    radius:  int,
) -> None:
    """Draw a 2-D Gaussian at `center` with `radius` onto `heatmap` in-place."""
    H, W = heatmap.shape
    r, c = center
    diameter = 2 * radius + 1
    sigma    = diameter / 6.0

    x  = torch.arange(-radius, radius + 1, dtype=torch.float32, device=heatmap.device)
    g  = torch.exp(-x.pow(2) / (2.0 * sigma ** 2))
    g2 = torch.outer(g, g)  # [d, d]

    r0 = max(0, r - radius); r1 = min(H, r + radius + 1)
    c0 = max(0, c - radius); c1 = min(W, c + radius + 1)
    gr0 = max(0, radius - r); gr1 = gr0 + (r1 - r0)
    gc0 = max(0, radius - c); gc1 = gc0 + (c1 - c0)

    if r1 > r0 and c1 > c0:
        heatmap[r0:r1, c0:c1] = torch.maximum(
            heatmap[r0:r1, c0:c1], g2[gr0:gr1, gc0:gc1]
        )


# ══════════════════════════════════════════════════════════════════════════════
# Task head 1 — 3D Semantic Segmentation
# ══════════════════════════════════════════════════════════════════════════════

# Approximate inverse-frequency class weights for Waymo 3D seg (22 classes).
# Order matches TASK_CLASSES["seg3d"].  Update after computing exact frequencies.
_SEG3D_CLASS_WEIGHTS = [
    0.50,   # 0  car
    1.00,   # 1  truck
    2.00,   # 2  bus
    1.50,   # 3  other_vehicle
    4.00,   # 4  motorcyclist
    3.00,   # 5  bicyclist
    2.00,   # 6  pedestrian
    2.50,   # 7  sign
    3.00,   # 8  traffic_light
    2.00,   # 9  pole
    3.50,   # 10 construction_cone
    3.00,   # 11 bicycle
    3.00,   # 12 motorcycle
    0.50,   # 13 building
    0.30,   # 14 vegetation
    1.50,   # 15 tree_trunk
    1.50,   # 16 curb
    0.20,   # 17 road
    1.50,   # 18 lane_marker
    1.00,   # 19 other_ground
    1.00,   # 20 walkable
    0.50,   # 21 sidewalk
]


class SegHead3D(nn.Module):
    """
    LiDAR 3D semantic segmentation head — 22 classes.

    Receives fused image+LiDAR patch tokens from ImageLidarFusion upstream.
    Reshapes [B, 196, 512] → [B, 512, 14, 14], bilinearly upsamples to
    224×224, then classifies via cosine similarity with text embeddings.

    Loss: weighted cross-entropy + Lovász-Softmax (λ=1.0).
    """

    NUM_CLASSES = 22
    _LAMBDA_LOV = 1.0

    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(0.0))

        w = torch.tensor(_SEG3D_CLASS_WEIGHTS, dtype=torch.float32)
        self.register_buffer("cls_weights", w / w.mean())   # normalise to mean=1

    def forward(
        self,
        fused_patches: torch.Tensor,   # [B, 196, 512]  already fused by ImageLidarFusion
        text_embs:     torch.Tensor,   # [22, 512]
    ) -> torch.Tensor:
        """Returns logits [B, 22, 224, 224]."""
        B = fused_patches.shape[0]
        x = fused_patches.permute(0, 2, 1).reshape(B, CLIP_DIM, GRID, GRID)  # [B, 512, 14, 14]
        x = F.interpolate(
            x.float(), size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False,
        )                                                                       # [B, 512, 224, 224]
        return _cosine_logits(x, text_embs, self.log_scale, spatial=True)      # [B, 22, 224, 224]

    def compute_loss(
        self,
        logits: torch.Tensor,   # [B, 22, 224, 224]
        labels: torch.Tensor,   # [B, 224, 224] int  (0-21; 255=ignore)
    ) -> Dict[str, torch.Tensor]:
        labels = labels.long()
        loss_wce = F.cross_entropy(
            logits, labels,
            weight=self.cls_weights.to(logits.device),
            ignore_index=SEG_IGNORE,
        )
        loss_lov = _lovasz_softmax(
            F.softmax(logits, dim=1), labels, ignore_index=SEG_IGNORE,
        )
        total = loss_wce + self._LAMBDA_LOV * loss_lov
        return {"loss": total, "loss_wce": loss_wce, "loss_lovasz": loss_lov}


# ══════════════════════════════════════════════════════════════════════════════
# Task head 2 — 3D Object Detection (DETR-style)
# ══════════════════════════════════════════════════════════════════════════════

class DetHead3D(nn.Module):
    """
    DETR-style 3D object detection head — 4 classes (vehicle/pedestrian/sign/cyclist).

    Replaces the former CenterPoint-BEV approach. The transformer decoder
    attends over fused image+LiDAR patch tokens (provided by ImageLidarFusion
    upstream) using Q learnable object queries.

    Forward:
        fused_patches [B, 196, D] → TransformerDecoder → Q query vectors
        cls_logits    [B, Q, 5]   (4 classes + per-query no-object scalar)
        box_preds     [B, Q, 7]   (cx, cy, cz, sx, sy, sz, heading) raw

    Loss: Hungarian matching on (cls_cost + λ * L1_position) then
          focal classification loss + L1 box regression over all 7 attributes.

    GT boxes: targets[b]["boxes"] is [M, 8] where col 7 = class index;
              cols 0:7 = (cx, cy, cz, sx, sy, sz, heading) in ego-frame metres/radians.
    """

    NUM_CLASSES = 4
    NUM_QUERIES = 300
    BOX_DIM     = 7   # cx, cy, cz, sx, sy, sz, heading

    _FOCAL_ALPHA = 0.25
    _FOCAL_GAMMA = 2.0
    _LAMBDA_BOX  = 0.25

    def __init__(
        self,
        num_decoder_layers: int   = 2,
        num_queries:        int   = 0,   # 0 → use NUM_QUERIES class default
        nhead:              int   = 8,
        dim_feedforward:    int   = 1024,
        dropout:            float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries or self.NUM_QUERIES

        self.log_scale   = nn.Parameter(torch.tensor(0.0))
        self.no_obj_bias = nn.Parameter(torch.zeros(self.num_queries, 1))
        self.query_embed = nn.Embedding(self.num_queries, CLIP_DIM)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=CLIP_DIM, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder  = nn.TransformerDecoder(decoder_layer,
                                              num_layers=num_decoder_layers)
        self.box_head = nn.Linear(CLIP_DIM, self.BOX_DIM)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.query_embed.weight, -0.1, 0.1)
        nn.init.zeros_(self.box_head.bias)
        nn.init.normal_(self.box_head.weight, std=0.01)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        fused_patches: torch.Tensor,   # [B, 196, 512]
        text_embs:     torch.Tensor,   # [4, 512]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_logits : [B, Q, NUM_CLASSES + 1]  (last = no-object)
            box_preds  : [B, Q, BOX_DIM]          raw (cx,cy,cz,sx,sy,sz,heading)
        """
        B       = fused_patches.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        out     = self.decoder(tgt=queries, memory=fused_patches)      # [B, Q, 512]

        cos_logits = _cosine_logits(out, text_embs, self.log_scale,
                                    spatial=False)                      # [B, Q, 4]
        no_obj     = self.no_obj_bias.unsqueeze(0).expand(B, -1, -1)   # [B, Q, 1]
        cls_logits = torch.cat([cos_logits, no_obj], dim=-1)            # [B, Q, 5]

        box_preds  = self.box_head(out)                                 # [B, Q, 7] raw
        return cls_logits, box_preds

    # ── Hungarian matching ────────────────────────────────────────────────────

    @torch.no_grad()
    def _match(
        self,
        cls_logits: torch.Tensor,   # [Q, 5]
        box_preds:  torch.Tensor,   # [Q, 7]
        gt_cls:     torch.Tensor,   # [M]
        gt_box:     torch.Tensor,   # [M, 7]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        M = gt_cls.shape[0]
        if M == 0:
            empty = torch.zeros(0, dtype=torch.long, device=cls_logits.device)
            return empty, empty

        prob     = cls_logits.softmax(-1)
        cost_cls = -prob[:, gt_cls]                                          # [Q, M]
        # Match on (cx, cy, cz) — position is the most discriminative cue
        cost_box = torch.cdist(box_preds[:, :3].float(),
                               gt_box[:, :3].float(), p=1)                   # [Q, M]
        cost = cost_cls + self._LAMBDA_BOX * cost_box

        rows, cols = linear_sum_assignment(cost.float().cpu().numpy())
        dev = cls_logits.device
        return (torch.as_tensor(rows, dtype=torch.long, device=dev),
                torch.as_tensor(cols, dtype=torch.long, device=dev))

    # ── Loss ──────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        output:  Tuple[torch.Tensor, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            output  : (cls_logits [B, Q, 5], box_preds [B, Q, 7])
            targets : list of B dicts, each {"labels": [M], "boxes": [M, 8]}
                      boxes[:, :7] = (cx,cy,cz,sx,sy,sz,heading); col 7 = class idx
        """
        cls_logits, box_preds = output
        B      = cls_logits.shape[0]
        dev    = cls_logits.device
        bg_idx = self.NUM_CLASSES

        cls_losses, box_losses = [], []

        for b in range(B):
            log_b    = cls_logits[b]                          # [Q, 5]
            box_b    = box_preds[b]                           # [Q, 7]
            gt_cls_b = targets[b]["labels"].to(dev)           # [M]
            gt_box_b = targets[b]["boxes"][:, :7].to(dev)    # [M, 7] (drop cls col)
            M        = gt_cls_b.shape[0]

            pred_i, gt_i = self._match(log_b, box_b, gt_cls_b, gt_box_b)

            tgt_cls = torch.full((self.num_queries,), bg_idx,
                                 dtype=torch.long, device=dev)
            if M > 0:
                tgt_cls[pred_i] = gt_cls_b[gt_i]

            # Focal classification loss
            one_hot = F.one_hot(tgt_cls, bg_idx + 1).float()
            p       = log_b.sigmoid()
            bce     = F.binary_cross_entropy_with_logits(log_b, one_hot, reduction="none")
            p_t     = p * one_hot + (1 - p) * (1 - one_hot)
            a_t     = (self._FOCAL_ALPHA * one_hot
                       + (1 - self._FOCAL_ALPHA) * (1 - one_hot))
            focal   = a_t * (1 - p_t) ** self._FOCAL_GAMMA * bce
            cls_losses.append(focal.sum() / max(M, 1))

            if M > 0:
                box_losses.append(
                    F.l1_loss(box_b[pred_i].float(), gt_box_b[gt_i].float())
                )
            else:
                box_losses.append(box_b.sum() * 0.0)

        loss_cls = torch.stack(cls_losses).mean()
        loss_box = torch.stack(box_losses).mean()
        return {
            "loss":     loss_cls + self._LAMBDA_BOX * loss_box,
            "loss_cls": loss_cls,
            "loss_box": loss_box,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Task head 3 — 2D Video Panoptic Segmentation (FPN Panoptic DeepLab style)
# ══════════════════════════════════════════════════════════════════════════════

class PanHead2D(nn.Module):
    """
    FPN-style dense video panoptic head — 28 classes.

    Architecture:
      3-stage FPN decoder (each: bilinear ×2 + Conv3×3 + GroupNorm + ReLU)
      14×14 → 28×28 → 56×56 → 112×112 dense feature map.

    Four output branches:
      semantic   : 1×1 conv → cosine-sim with text → [B, T, 28, 112, 112]
      center     : Conv3×3 → [B, T, 1, 112, 112]  heatmap logits
      offset     : Conv3×3 → [B, T, 2, 112, 112]  pixel-to-center vectors
      embedding  : Conv3×3 → [B, T, D_emb, 112, 112]  for contrastive loss

    Loss: CE (semantic) + focal (center) + masked L1 (offset) + contrastive (embedding)
    """

    NUM_CLASSES  = 28
    FPN_CHANNELS = (256, 128, 64)   # channels after each of the 3 FPN stages
    FPN_OUT_SIZE = 112              # 14 → 28 → 56 → 112
    EMB_DIM      = 64

    _LAMBDA_CENTER   = 200.0
    _LAMBDA_OFFSET   = 0.01
    _LAMBDA_CONTRAST = 0.1
    _CONTRAST_MARGIN = 1.5

    def __init__(self, num_queries: int = 0):
        # num_queries unused; kept for build_task_heads API compatibility
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(0.0))

        # FPN decoder
        in_ch = CLIP_DIM
        self.fpn_stages = nn.ModuleList()
        for out_ch in self.FPN_CHANNELS:
            self.fpn_stages.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch

        fpn_dim = self.FPN_CHANNELS[-1]   # 64

        # Semantic branch: project to CLIP_DIM then cosine-sim
        self.sem_proj = nn.Conv2d(fpn_dim, CLIP_DIM, 1)

        # Center heatmap branch
        self.center_head = nn.Sequential(
            nn.Conv2d(fpn_dim, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
        )

        # Offset branch
        self.offset_head = nn.Sequential(
            nn.Conv2d(fpn_dim, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),
        )

        # Instance embedding branch (for inter-frame contrastive loss)
        self.embed_head = nn.Sequential(
            nn.Conv2d(fpn_dim, self.EMB_DIM, 3, padding=1),
            nn.GroupNorm(8, self.EMB_DIM),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.EMB_DIM, self.EMB_DIM, 3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        patch_tokens: torch.Tensor,   # [B, T, 196, 512]
        text_embs:    torch.Tensor,   # [28, 512]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = patch_tokens.shape[:2]
        BT   = B * T
        hw   = self.FPN_OUT_SIZE

        # Flatten temporal dim, reshape to spatial grid
        x = patch_tokens.flatten(0, 1)                                 # [BT, 196, 512]
        x = x.permute(0, 2, 1).reshape(BT, CLIP_DIM, GRID, GRID)      # [BT, 512, 14, 14]

        # FPN decoder: 3 stages, each doubles spatial resolution
        for stage in self.fpn_stages:
            x = F.interpolate(x.float(), scale_factor=2.0,
                              mode="bilinear", align_corners=False)
            x = stage(x)
        # x: [BT, 64, 112, 112]

        # Semantic branch
        sem_feat   = self.sem_proj(x)                                  # [BT, 512, 112, 112]
        sem_logits = _cosine_logits(sem_feat, text_embs,
                                    self.log_scale, spatial=True)       # [BT, 28, 112, 112]

        # Other branches
        center_logits = self.center_head(x)                            # [BT, 1, 112, 112]
        offset_preds  = self.offset_head(x)                            # [BT, 2, 112, 112]
        embeddings    = self.embed_head(x)                             # [BT, D, 112, 112]

        # Reshape to [B, T, ...]
        def _reshape(t, nc):
            return t.view(B, T, nc, hw, hw)

        return (
            _reshape(sem_logits,    self.NUM_CLASSES),
            _reshape(center_logits, 1),
            _reshape(offset_preds,  2),
            _reshape(embeddings,    self.EMB_DIM),
        )

    # ── GT helpers ────────────────────────────────────────────────────────────

    def _make_center_offset_gt(
        self,
        sem_gt:  torch.Tensor,   # [hw, hw] int
        inst_gt: torch.Tensor,   # [hw, hw] int  (0=stuff, >0=thing)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            center_gt  [1, hw, hw]  Gaussian heatmap per instance centroid
            offset_gt  [2, hw, hw]  pixel-to-center offsets
            thing_mask [hw, hw]     bool foreground mask
        """
        hw  = self.FPN_OUT_SIZE
        dev = sem_gt.device

        center_gt  = torch.zeros(1, hw, hw, device=dev, dtype=torch.float32)
        offset_gt  = torch.zeros(2, hw, hw, device=dev, dtype=torch.float32)
        thing_mask = torch.zeros(hw, hw,    device=dev, dtype=torch.bool)

        rows = torch.arange(hw, device=dev, dtype=torch.float32)
        cols = torch.arange(hw, device=dev, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")

        inst_ids = inst_gt[inst_gt > 0].unique()
        for iid in inst_ids:
            mask = inst_gt == iid
            if mask.sum() == 0:
                continue

            r_center = row_grid[mask].mean()
            c_center = col_grid[mask].mean()
            radius   = max(1, int(mask.sum().float().sqrt().item() / 4.0))

            _draw_gaussian(center_gt[0],
                           (int(r_center.item()), int(c_center.item())), radius)

            offset_gt[0, mask] = r_center - row_grid[mask]
            offset_gt[1, mask] = c_center - col_grid[mask]
            thing_mask         |= mask

        return center_gt, offset_gt, thing_mask

    # ── Contrastive loss ──────────────────────────────────────────────────────

    def _contrastive_loss(
        self,
        embeddings: torch.Tensor,   # [B, T, EMB_DIM, hw, hw]
        inst_gt:    torch.Tensor,   # [B, T, hw, hw]  int
        max_pts:    int = 512,
    ) -> torch.Tensor:
        """Inter-frame contrastive loss across consecutive frame pairs."""
        B, T = embeddings.shape[:2]
        if T < 2:
            return embeddings.sum() * 0.0

        margin = self._CONTRAST_MARGIN
        losses = []

        for b in range(B):
            for t in range(T - 1):
                e1 = embeddings[b, t]    # [D, hw, hw]
                e2 = embeddings[b, t+1]
                i1 = inst_gt[b, t]       # [hw, hw]
                i2 = inst_gt[b, t+1]

                fg1 = (i1 > 0).nonzero(as_tuple=False)   # [K1, 2]
                fg2 = (i2 > 0).nonzero(as_tuple=False)   # [K2, 2]
                if fg1.shape[0] == 0 or fg2.shape[0] == 0:
                    continue

                # Subsample foreground pixels to keep memory tractable
                idx1 = fg1[torch.randperm(fg1.shape[0], device=fg1.device)[:max_pts]]
                idx2 = fg2[torch.randperm(fg2.shape[0], device=fg2.device)[:max_pts]]

                e1s  = e1[:, idx1[:, 0], idx1[:, 1]].t()  # [n1, D]
                e2s  = e2[:, idx2[:, 0], idx2[:, 1]].t()  # [n2, D]
                lbl1 = i1[idx1[:, 0], idx1[:, 1]]          # [n1]
                lbl2 = i2[idx2[:, 0], idx2[:, 1]]          # [n2]

                # Pairwise distances
                diff = e1s.unsqueeze(1) - e2s.unsqueeze(0)   # [n1, n2, D]
                dist = diff.norm(dim=-1)                       # [n1, n2]
                same = (lbl1.unsqueeze(1) == lbl2.unsqueeze(0)).float()

                loss_pos = same * dist.pow(2)
                loss_neg = (1 - same) * F.relu(margin - dist).pow(2)
                losses.append((loss_pos + loss_neg).mean())

        if not losses:
            return embeddings.sum() * 0.0
        return torch.stack(losses).mean()

    # ── Loss ──────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        output:  Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            output  : (sem_logits [B,T,28,112,112], center_logits [B,T,1,112,112],
                       offset_preds [B,T,2,112,112], embeddings [B,T,D,112,112])
            targets : {"semantic": [B,T,H,W], "instance": [B,T,H,W]}
        """
        sem_logits, center_logits, offset_preds, embeddings = output
        B, T = sem_logits.shape[:2]
        hw   = self.FPN_OUT_SIZE
        dev  = sem_logits.device

        # Downsample GT to FPN output resolution
        def _ds(t):
            return F.interpolate(
                t.float().flatten(0, 1).unsqueeze(1),
                size=(hw, hw), mode="nearest",
            ).squeeze(1).long().view(B, T, hw, hw)

        sem_gt  = _ds(targets["semantic"].to(dev))
        inst_gt = _ds(targets["instance"].to(dev))

        # ── Semantic CE loss ──
        loss_sem = F.cross_entropy(
            sem_logits.flatten(0, 1),   # [BT, 28, hw, hw]
            sem_gt.flatten(0, 1),       # [BT, hw, hw]
            ignore_index=SEG_IGNORE,
        )

        # ── Center + offset GT ──
        center_gt_list, offset_gt_list, thing_mask_list = [], [], []
        for b in range(B):
            for t in range(T):
                cgt, ogt, tm = self._make_center_offset_gt(sem_gt[b, t], inst_gt[b, t])
                center_gt_list.append(cgt)
                offset_gt_list.append(ogt)
                thing_mask_list.append(tm)

        center_gt  = torch.stack(center_gt_list).to(dev)    # [BT, 1, hw, hw]
        offset_gt  = torch.stack(offset_gt_list).to(dev)    # [BT, 2, hw, hw]
        thing_mask = torch.stack(thing_mask_list).to(dev)   # [BT, hw, hw] bool

        # ── Center focal loss ──
        center_prob = center_logits.flatten(0, 1).sigmoid()  # [BT, 1, hw, hw]
        loss_center = _centerness_focal_loss(center_prob, center_gt)

        # ── Offset L1 loss (foreground only) ──
        offset_flat     = offset_preds.flatten(0, 1)                    # [BT, 2, hw, hw]
        thing_mask_exp  = thing_mask.unsqueeze(1).expand_as(offset_flat)  # [BT, 2, hw, hw]
        if thing_mask_exp.any():
            loss_offset = F.l1_loss(offset_flat[thing_mask_exp],
                                    offset_gt[thing_mask_exp])
        else:
            loss_offset = offset_preds.sum() * 0.0

        # ── Inter-frame contrastive loss ──
        loss_contrast = self._contrastive_loss(embeddings, inst_gt)

        total = (loss_sem
                 + self._LAMBDA_CENTER   * loss_center
                 + self._LAMBDA_OFFSET   * loss_offset
                 + self._LAMBDA_CONTRAST * loss_contrast)

        return {
            "loss":          total,
            "loss_sem":      loss_sem,
            "loss_center":   loss_center,
            "loss_offset":   loss_offset,
            "loss_contrast": loss_contrast,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Task head 4 — 2D Camera Object Detection (DETR-style)
# ══════════════════════════════════════════════════════════════════════════════

class _DetHead(nn.Module):
    """
    DETR-style detection base (shared by DetHead2D).

    Per-query learnable scalar no-object bias concatenated after cosine-sim,
    keeping no-object score outside the text embedding space.
    Loss: Hungarian matching + focal (cls) + GIoU + L1 (box).
    """

    _FOCAL_ALPHA = 0.25
    _FOCAL_GAMMA = 2.0
    _LAMBDA_BOX  = 5.0
    _LAMBDA_GIOU = 2.0

    def __init__(
        self,
        num_classes:        int,
        num_queries:        int,
        box_dim:            int,
        num_decoder_layers: int   = 2,
        nhead:              int   = 8,
        dim_feedforward:    int   = 1024,
        dropout:            float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.box_dim     = box_dim

        self.log_scale   = nn.Parameter(torch.tensor(0.0))
        self.no_obj_bias = nn.Parameter(torch.zeros(num_queries, 1))
        self.query_embed = nn.Embedding(num_queries, CLIP_DIM)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=CLIP_DIM, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder  = nn.TransformerDecoder(decoder_layer,
                                              num_layers=num_decoder_layers)
        self.box_head = nn.Linear(CLIP_DIM, box_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.query_embed.weight, -0.1, 0.1)
        nn.init.zeros_(self.box_head.bias)
        nn.init.normal_(self.box_head.weight, std=0.01)

    def forward(
        self,
        patch_tokens: torch.Tensor,   # [B, 196, 512]
        text_embs:    torch.Tensor,   # [C, 512]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_logits : [B, N, num_classes + 1]   (last = no-object)
            box_preds  : [B, N, box_dim]            (sigmoid-normalised)
        """
        B       = patch_tokens.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        out     = self.decoder(tgt=queries, memory=patch_tokens)       # [B, N, 512]

        cos_logits = _cosine_logits(out, text_embs, self.log_scale,
                                    spatial=False)                      # [B, N, C]
        no_obj     = self.no_obj_bias.unsqueeze(0).expand(B, -1, -1)   # [B, N, 1]
        cls_logits = torch.cat([cos_logits, no_obj], dim=-1)            # [B, N, C+1]

        box_preds  = self.box_head(out).sigmoid()                       # [B, N, box_dim]
        return cls_logits, box_preds

    @torch.no_grad()
    def _match(
        self,
        cls_logits: torch.Tensor,   # [N, C+1]
        box_preds:  torch.Tensor,   # [N, 4]
        gt_cls:     torch.Tensor,   # [M]
        gt_box:     torch.Tensor,   # [M, 4]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        M = gt_cls.shape[0]
        if M == 0:
            empty = torch.zeros(0, dtype=torch.long, device=cls_logits.device)
            return empty, empty

        prob     = cls_logits.softmax(-1)
        cost_cls = -prob[:, gt_cls]                                         # [N, M]
        cost_l1  = torch.cdist(box_preds.float(), gt_box.float(), p=1)     # [N, M]

        # GIoU cost
        def _to_xyxy(b):
            cx, cy, w, h = b.unbind(-1)
            return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

        pred_xyxy = _to_xyxy(box_preds.float())   # [N, 4]
        gt_xyxy   = _to_xyxy(gt_box.float())      # [M, 4]
        giou      = generalized_box_iou(pred_xyxy, gt_xyxy)   # [N, M]
        cost_giou = 1.0 - giou

        cost = cost_cls + self._LAMBDA_BOX * cost_l1 + self._LAMBDA_GIOU * cost_giou

        rows, cols = linear_sum_assignment(cost.float().cpu().numpy())
        dev = cls_logits.device
        return (torch.as_tensor(rows, dtype=torch.long, device=dev),
                torch.as_tensor(cols, dtype=torch.long, device=dev))

    def compute_loss(
        self,
        cls_logits: torch.Tensor,
        box_preds:  torch.Tensor,
        targets:    List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        bg_idx = self.num_classes
        cls_losses, box_losses, giou_losses = [], [], []

        for b in range(cls_logits.shape[0]):
            log_b    = cls_logits[b]
            box_b    = box_preds[b]
            gt_cls_b = targets[b]["labels"].to(log_b.device)
            gt_box_b = targets[b]["boxes"].to(box_b.device)
            M        = gt_cls_b.shape[0]

            pred_i, gt_i = self._match(log_b, box_b, gt_cls_b, gt_box_b)

            tgt_cls = torch.full((self.num_queries,), bg_idx,
                                 dtype=torch.long, device=log_b.device)
            if M > 0:
                tgt_cls[pred_i] = gt_cls_b[gt_i]

            # Focal classification loss
            one_hot = F.one_hot(tgt_cls, self.num_classes + 1).float()
            p       = log_b.sigmoid()
            bce     = F.binary_cross_entropy_with_logits(log_b, one_hot, reduction="none")
            p_t     = p * one_hot + (1 - p) * (1 - one_hot)
            a_t     = self._FOCAL_ALPHA * one_hot + (1 - self._FOCAL_ALPHA) * (1 - one_hot)
            focal   = a_t * (1 - p_t) ** self._FOCAL_GAMMA * bce
            cls_losses.append(focal.sum() / max(M, 1))

            if M > 0:
                matched_pred = box_b[pred_i]         # [K, 4]
                matched_gt   = gt_box_b[gt_i].to(box_b)  # [K, 4]
                box_losses.append(F.l1_loss(matched_pred, matched_gt))

                # GIoU loss on matched pairs
                def _to_xyxy(bxs):
                    cx, cy, w, h = bxs.unbind(-1)
                    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

                giou = torch.diag(generalized_box_iou(
                    _to_xyxy(matched_pred.float()),
                    _to_xyxy(matched_gt.float()),
                ))                                       # [K]
                giou_losses.append((1 - giou).mean())
            else:
                box_losses.append(box_b.sum() * 0.0)
                giou_losses.append(box_b.sum() * 0.0)

        loss_cls  = torch.stack(cls_losses).mean()
        loss_box  = torch.stack(box_losses).mean()
        loss_giou = torch.stack(giou_losses).mean()
        return {
            "loss":      loss_cls + self._LAMBDA_BOX * loss_box + self._LAMBDA_GIOU * loss_giou,
            "loss_cls":  loss_cls,
            "loss_box":  loss_box,
            "loss_giou": loss_giou,
        }


class DetHead2D(_DetHead):
    """Camera 2D detection — 3 classes (vehicle/pedestrian/cyclist)."""

    NUM_CLASSES = 3
    NUM_QUERIES = 100
    BOX_DIM     = 4

    def __init__(self, num_decoder_layers: int = 2, num_queries: int = 0):
        super().__init__(
            num_classes        = self.NUM_CLASSES,
            num_queries        = num_queries or self.NUM_QUERIES,
            box_dim            = self.BOX_DIM,
            num_decoder_layers = num_decoder_layers,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_task_heads(
    num_decoder_layers: int = 2,
    num_queries_3d:     int = 300,
    num_queries_2d:     int = 100,
    num_queries_pan:    int = 100,
) -> nn.ModuleDict:
    """
    Construct all four task heads.

    text_embs are not passed here — heads receive them at each forward call
    from CustomCLIP.  Fusion of image+LiDAR tokens for 3D tasks is handled
    upstream by ImageLidarFusion (models/fusion.py), not inside the heads.

    Args:
        num_decoder_layers : DETR decoder depth for DetHead3D and DetHead2D.
        num_queries_3d     : number of object queries for DetHead3D.
        num_queries_2d     : number of object queries for DetHead2D.
        num_queries_pan    : unused in PanHead2D; kept for config compatibility.
    """
    return nn.ModuleDict({
        "seg3d": SegHead3D(),
        "det3d": DetHead3D(num_decoder_layers, num_queries=num_queries_3d),
        "pan2d": PanHead2D(num_queries=num_queries_pan),
        "det2d": DetHead2D(num_decoder_layers, num_queries=num_queries_2d),
    })
