"""
clip/lidar_encoder.py
---------------------
LiDAR branch encoder for FedMMA.

Architecture
------------
LiDAREncoder
├── lidar_modules  (LiDARTrainableModules — federation unit)
│   ├── stem      (LiDARInputStem        — conv3×3 + BN, aggregated in phase 1 only)
│   └── adapters  (nn.ModuleList × 8    — LiDARAdapterBottleneck, aggregated in both phases)
└── backbone      (VisionTransformer     — frozen deepcopy of CLIP image encoder)

Federated aggregation
---------------------
Phase 1:  agg_module = encoder.lidar_modules           # stem + adapters
Phase 2:  agg_module = encoder.lidar_modules.adapters  # adapters only
          encoder.freeze_stem() must be called at the phase-2 boundary.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn


# ── Input stem ────────────────────────────────────────────────────────────────

class LiDARInputStem(nn.Module):
    """Learnable 3×3 conv + BatchNorm that remaps LiDAR channel statistics
    (range, intensity, elongation) into a distribution compatible with CLIP's
    RGB-pretrained patch embedding.

    Initialized as near-identity: the centre pixel of each kernel copies its
    corresponding input channel, all others are zero. BatchNorm defaults to
    weight=1 / bias=0. Together this means the stem starts as a pass-through
    and departs from identity only as training updates its parameters.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(3)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.conv.weight)
        with torch.no_grad():
            for i in range(3):
                self.conv.weight[i, i, 1, 1] = 1.0   # [out_ch, in_ch, kH, kW]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep stem in fp32 — caller casts to backbone dtype after this.
        return self.norm(self.conv(x))


# ── Adapter bottleneck ────────────────────────────────────────────────────────

class LiDARAdapterBottleneck(nn.Module):
    """Per-layer bottleneck adapter injected into transformer blocks adapter_start–12.

    Follows the down / up protocol consumed by ModifiedResidualAttentionBlock:

        seq_adapter, shared_adapter, scale = adapter_func(layer_idx)
        y = seq_adapter.down(x)   # compress: width → adapter_dim
        y = seq_adapter.up(y)     # expand:   adapter_dim → width

    up is zero-initialized so the adapter contributes nothing at the start of
    training — the frozen backbone output passes through unchanged.
    """

    def __init__(self, width: int, adapter_dim: int):
        super().__init__()
        self.down = nn.Linear(width, adapter_dim)
        self.up   = nn.Linear(adapter_dim, width)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)


# ── Trainable container (federation unit) ────────────────────────────────────

class LiDARTrainableModules(nn.Module):
    """Groups the input stem and all adapter bottlenecks so federated
    aggregation has a single, phase-aware handle.

    Usage in federated_main.py
    --------------------------
    if phase == 1:
        agg_module = encoder.lidar_modules           # stem + adapters
    else:
        agg_module = encoder.lidar_modules.adapters  # adapters only
    # then: sd = agg_module.state_dict() / agg_module.load_state_dict(sd)
    """

    def __init__(self, width: int, adapter_dim: int, num_adapter_layers: int):
        super().__init__()
        self.stem     = LiDARInputStem()
        self.adapters = nn.ModuleList([
            LiDARAdapterBottleneck(width, adapter_dim)
            for _ in range(num_adapter_layers)
        ])


# ── LiDAR encoder ─────────────────────────────────────────────────────────────

class LiDAREncoder(nn.Module):
    """LiDAR branch of FedMMA.

    Wraps a frozen copy of CLIP's VisionTransformer with:
      - a trainable input stem  (LiDARInputStem)   that remaps LiDAR channel
        statistics into a distribution compatible with CLIP's patch embedding
      - trainable adapter bottlenecks (LiDARAdapterBottleneck) injected at
        transformer blocks ``adapter_start`` through 12 (inclusive, 1-indexed,
        matching ModifiedResidualAttentionBlock.layer_index)

    Both components live inside ``self.lidar_modules`` for clean federation.

    Args:
        clip_visual:   CLIP's VisionTransformer instance. Weights are deep-copied
                       and frozen; the original is not modified.
        adapter_dim:   Bottleneck width for each adapter layer (default 64).
        adapter_start: First transformer block index to inject adapters (default 5).
    """

    def __init__(
        self,
        clip_visual: nn.Module,
        adapter_dim: int = 64,
        adapter_start: int = 5,
    ):
        super().__init__()

        # ── frozen backbone ───────────────────────────────────────────────────
        self.backbone = copy.deepcopy(clip_visual)
        self.backbone.requires_grad_(False)

        # ── architecture constants ────────────────────────────────────────────
        width              = clip_visual.transformer.width          # 768 for ViT-B/16
        num_blocks         = self.backbone.transformer.layers       # 12  for ViT-B/16
        num_adapter_layers = num_blocks - adapter_start + 1        # 8   (blocks 5–12)
        self.adapter_start = adapter_start

        # ── trainable LiDAR params (federation unit) ──────────────────────────
        self.lidar_modules = LiDARTrainableModules(
            width              = width,
            adapter_dim        = adapter_dim,
            num_adapter_layers = num_adapter_layers,
        )

    # ── phase transition ──────────────────────────────────────────────────────

    def freeze_stem(self):
        """Freeze the input stem at the start of phase 2.

        Call this once before the phase-2 training loop begins.  After this:
          - the stem's parameters receive no gradients
          - federated aggregation should switch to encoder.lidar_modules.adapters
        """
        self.lidar_modules.stem.requires_grad_(False)

    # ── dtype helper ─────────────────────────────────────────────────────────

    @property
    def dtype(self):
        """Dtype of the frozen backbone (fp16 after convert_weights)."""
        return self.backbone.conv1.weight.dtype

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        shared_adapter: Optional[nn.Module] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            x:              LiDAR range image  [B, 3, H, W]  (range / intensity / elongation).
            shared_adapter: Optional cross-modal shared adapter passed in from
                            mmadapter.py; None during standalone testing.
            scale:          Gradient scale forwarded to the adapter injection logic
                            inside ModifiedResidualAttentionBlock.

        Returns:
            Patch tokens  [B, 196, output_dim]  (14×14 spatial grid, dim=512 for ViT-B/16).
        """
        # 1. Remap LiDAR statistics → RGB-compatible range (runs in fp32)
        x = self.lidar_modules.stem(x)

        # 2. Cast to backbone dtype (fp16 after convert_weights, matching CLIP.encode_image)
        x = x.type(self.dtype)

        # 3. Build the adapter_func closure consumed by VisionTransformer.forward
        #    and ModifiedResidualAttentionBlock.
        adapters = self.lidar_modules.adapters
        start    = self.adapter_start

        def adapter_func(layer_idx: int):
            # layer_idx 0        → pre-transformer (ln_pre region), no adapter
            # layer_idx 1…start-1 → early blocks,                    no adapter
            # layer_idx start…12 → late blocks,                       LiDAR adapter
            if layer_idx < start:
                return None, None, scale
            return adapters[layer_idx - start], shared_adapter, scale

        # 4. Run frozen backbone with adapter injection; returns [B, 196, output_dim]
        return self.backbone([x, adapter_func])
