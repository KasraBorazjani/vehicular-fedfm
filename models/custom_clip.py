"""
models/custom_clip.py
---------------------
CustomCLIP: three-branch multi-task CLIP model for FedMMA.

Architecture
------------
Three encoders share a common CLIP ViT-B/16 backbone:

  image  → frozen CLIP ViT-B/16 + visual adapters δv  (layers adapter_start–12)
  lidar  → frozen CLIP ViT-B/16 copy + LiDARInputStem + LiDAR adapters δℓ
  text   → frozen CLIP text transformer + text adapters δt (layers adapter_start–12)

Per-task forward:
  seg3d / det3d : image_patches + lidar_patches → ImageLidarFusion → fused tokens
                  → task head.  Fusion is decoupled from the heads so it can be
                  swapped independently (see models/fusion.py).
  pan2d         : image [B,T,3,224,224] flattened → encoder → SharedTaskAdapter
                  → PanHead2D (FPN per-frame, contrastive across T frames)
  det2d         : image_patches → SharedTaskAdapter → DetHead2D

Text embeddings are recomputed at every forward pass (no caching) so that
gradients flow back through the text adapter parameters.
"""

from typing import Optional

import torch
import torch.nn as nn

from clip import clip
from models.adapters import AdapterBottleneck, SharedTaskAdapter
from models.fusion import ImageLidarFusion
from models.lidar_encoder import LiDAREncoder
from models.task_heads import build_task_heads
from utils.text_templates import TASKS, TASK_CLASSES, TASK_TEMPLATES, TEXT_SLICES, TASK_MODALITIES


# ── CLIP ViT-B/16 architecture constants ──────────────────────────────────────

_CLIP_TRANSFORMER_WIDTH = 768
_CLIP_OUTPUT_DIM        = 512
_CLIP_NUM_LAYERS        = 12


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_modality_adapters(
    width:         int,
    num_layers:    int,
    adapter_start: int,
    adapter_dim:   int,
) -> nn.ModuleList:
    adapters = [None] * (num_layers + 1)
    for i in range(adapter_start, num_layers + 1):
        adapters[i] = AdapterBottleneck(width, adapter_dim)
    return nn.ModuleList(adapters)


# ── Text encoder ──────────────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.transformer          = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final             = clip_model.ln_final
        self.text_projection      = clip_model.text_projection
        self.dtype                = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, adapter_func=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        if adapter_func is not None:
            x = self.transformer([x, adapter_func])
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        x = x @ self.text_projection
        return x


# ── CustomCLIP ────────────────────────────────────────────────────────────────

class CustomCLIP(nn.Module):
    """
    Three-branch multi-task model for FedMMA task onboarding.

    Args
    ----
    clip_model             : loaded CLIP model (clip.build_model_MMA output).
    arriving_task          : task being onboarded ("seg3d"|"det3d"|"pan2d"|"det2d").
    adapter_dim            : bottleneck width for δv / δt / δℓ adapters.
    shared_adapter_dim     : bottleneck width for SharedTaskAdapter (2D tasks only).
    adapter_start          : first transformer layer to inject adapters (1-indexed).
    adapter_scale          : gradient scale coefficient.
    num_decoder_layers     : DETR decoder depth for DetHead2D.
    use_task_adapter       : False → nn.Identity for 2D tasks (NTA baseline).
    freeze_pretrained_heads: True → non-arriving heads frozen at construction.
    """

    def __init__(
        self,
        clip_model,
        arriving_task:             str,
        adapter_dim:               int   = 64,
        shared_adapter_dim:        int   = 64,
        adapter_start:             int   = 5,
        adapter_scale:             float = 1.0,
        num_decoder_layers:        int   = 2,
        num_queries_3d:            int   = 300,
        num_queries_2d:            int   = 100,
        num_queries_pan:           int   = 100,
        use_task_adapter:          bool  = True,
        freeze_pretrained_heads:   bool  = False,
    ):
        super().__init__()

        self.arriving_task = arriving_task
        self.adapter_scale = adapter_scale
        self.adapter_start = adapter_start
        self.dtype         = clip_model.dtype

        clip_model.requires_grad_(False)

        self.image_encoder = clip_model.visual

        self.lidar_encoder = LiDAREncoder(
            clip_visual   = clip_model.visual,
            adapter_dim   = adapter_dim,
            adapter_start = adapter_start,
        )

        self.text_encoder = TextEncoder(clip_model)

        self.vis_adapters = _build_modality_adapters(
            _CLIP_TRANSFORMER_WIDTH, _CLIP_NUM_LAYERS, adapter_start, adapter_dim,
        )
        self.txt_adapters = _build_modality_adapters(
            _CLIP_TRANSFORMER_WIDTH, _CLIP_NUM_LAYERS, adapter_start, adapter_dim,
        )

        # Fusion module for 3D tasks: merges image + LiDAR patch tokens.
        # Lives here (not in the heads) so the mechanism is swappable via models/fusion.py.
        self.image_lidar_fusion = ImageLidarFusion()

        # SharedTaskAdapter: used only for 2D tasks (pan2d, det2d).
        self.task_adapter = (
            SharedTaskAdapter(dim=_CLIP_OUTPUT_DIM, shared_adapter_dim=shared_adapter_dim)
            if use_task_adapter
            else nn.Identity()
        )

        self._build_text_prompts(clip_model)

        self.task_heads = build_task_heads(
            num_decoder_layers = num_decoder_layers,
            num_queries_3d     = num_queries_3d,
            num_queries_2d     = num_queries_2d,
            num_queries_pan    = num_queries_pan,
        )

        if freeze_pretrained_heads:
            self.freeze_pretrained_task_heads()

    # ── Text prompt construction ──────────────────────────────────────────────

    def _build_text_prompts(self, clip_model: nn.Module):
        dtype   = clip_model.dtype
        prompts = []
        for task in TASKS:
            template = TASK_TEMPLATES[task]
            for name in TASK_CLASSES[task]:
                prompts.append(template.format(name.replace("_", " ")))

        tokenized = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(dtype)

        self.register_buffer("token_embedding",   embedding)
        self.register_buffer("tokenized_prompts", tokenized)

    # ── Phase-transition helpers ──────────────────────────────────────────────

    def freeze_lidar_stem(self):
        self.lidar_encoder.freeze_stem()

    def freeze_pretrained_task_heads(self):
        for task, head in self.task_heads.items():
            if task != self.arriving_task:
                head.requires_grad_(False)

    # ── Adapter function closures ─────────────────────────────────────────────

    def _vis_adapter_func(self, layer_idx: int):
        return self.vis_adapters[layer_idx], None, self.adapter_scale

    def _txt_adapter_func(self, layer_idx: int):
        return self.txt_adapters[layer_idx], None, self.adapter_scale

    # ── Text encoding ─────────────────────────────────────────────────────────

    def _encode_text(self) -> torch.Tensor:
        return self.text_encoder(
            self.token_embedding,
            self.tokenized_prompts,
            adapter_func=self._txt_adapter_func,
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        task:  str,
        image: torch.Tensor,
        lidar: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            task      : "seg3d" | "det3d" | "pan2d" | "det2d"
            image     : [B, 3, 224, 224]       for seg3d / det3d / det2d
                        [B, T, 3, 224, 224]    for pan2d
            lidar     : [B, 3, 224, 224]       for 3D tasks; None for 2D

        Returns:
            seg3d  : logits [B, 22, 224, 224]
            det3d  : (cls_logits [B,Q,5], box_preds [B,Q,7])
            pan2d  : (sem_logits [B,T,28,112,112], center [B,T,1,112,112],
                      offset [B,T,2,112,112], embeddings [B,T,64,112,112])
            det2d  : (cls_logits [B,N,4], box_preds [B,N,4])
        """
        text_embs      = self._encode_text()
        task_text_embs = text_embs[TEXT_SLICES[task]]

        # ── pan2d: video sequence path ────────────────────────────────────────
        if task == "pan2d":
            B, T = image.shape[:2]
            img_flat     = image.view(B * T, *image.shape[2:])
            patches_flat = self.image_encoder(
                [img_flat.type(self.dtype), self._vis_adapter_func]
            )                                                  # [B*T, 196, 512]
            fused_flat   = self.task_adapter(patches_flat)    # [B*T, 196, 512]
            fused        = fused_flat.view(B, T, 196, _CLIP_OUTPUT_DIM)
            return self.task_heads["pan2d"](fused, task_text_embs)

        # ── seg3d: image + LiDAR → ImageLidarFusion → SegHead3D ──────────────
        if task == "seg3d":
            if lidar is None:
                raise ValueError("seg3d requires a lidar tensor")
            image_patches = self.image_encoder(
                [image.type(self.dtype), self._vis_adapter_func]
            )
            lidar_patches = self.lidar_encoder(
                lidar, shared_adapter=None, scale=self.adapter_scale,
            )
            fused = self.image_lidar_fusion(image_patches, lidar_patches)
            return self.task_heads["seg3d"](fused, task_text_embs)

        # ── det3d: image + LiDAR → ImageLidarFusion → DetHead3D (DETR) ───────
        if task == "det3d":
            if lidar is None:
                raise ValueError("det3d requires a lidar tensor")
            image_patches = self.image_encoder(
                [image.type(self.dtype), self._vis_adapter_func]
            )
            lidar_patches = self.lidar_encoder(
                lidar, shared_adapter=None, scale=self.adapter_scale,
            )
            fused = self.image_lidar_fusion(image_patches, lidar_patches)
            return self.task_heads["det3d"](fused, task_text_embs)

        # ── det2d: image only, SharedTaskAdapter ─────────────────────────────
        image_patches = self.image_encoder(
            [image.type(self.dtype), self._vis_adapter_func]
        )
        fused = self.task_adapter(image_patches)
        return self.task_heads["det2d"](fused, task_text_embs)
