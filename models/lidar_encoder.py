"""
models/lidar_encoder.py
-----------------------
LiDAR branch encoder for FedMMA.

Architecture
------------
LiDAREncoder
├── lidar_modules  (LiDARTrainableModules — federation unit)
│   ├── stem      (LiDARInputStem — conv3×3 + BN, aggregated in phase 1 only)
│   └── adapters  (nn.ModuleList of AdapterBottleneck, aggregated in both phases)
└── backbone      (VisionTransformer — frozen deepcopy of CLIP image encoder)

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

from models.adapters import AdapterBottleneck


class LiDARInputStem(nn.Module):
    """
    Learnable 3×3 conv + BatchNorm that remaps LiDAR channel statistics
    (range, intensity, elongation) into a distribution compatible with CLIP's
    RGB-pretrained patch embedding.

    Near-identity initialisation: the centre pixel of each 3×3 kernel copies its
    corresponding input channel; all others start at zero. BatchNorm starts as
    weight=1 / bias=0. The stem begins as a pass-through and drifts from identity
    only as training updates its parameters.
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
        # Stem runs in fp32; caller casts to backbone dtype after this.
        return self.norm(self.conv(x))


class LiDARTrainableModules(nn.Module):
    """
    Federation unit grouping the input stem and adapter bottlenecks.

    Usage in federated_main.py:
        phase 1: agg_module = encoder.lidar_modules           # stem + adapters
        phase 2: agg_module = encoder.lidar_modules.adapters  # adapters only
        # sd = agg_module.state_dict() / agg_module.load_state_dict(sd)
    """

    def __init__(self, width: int, adapter_dim: int, num_adapter_layers: int):
        super().__init__()
        self.stem     = LiDARInputStem()
        self.adapters = nn.ModuleList([
            AdapterBottleneck(width, adapter_dim)
            for _ in range(num_adapter_layers)
        ])


class LiDAREncoder(nn.Module):
    """
    LiDAR branch of FedMMA.

    Wraps a frozen deepcopy of CLIP's VisionTransformer with a trainable input
    stem (LiDARInputStem) and AdapterBottleneck layers injected at transformer
    blocks adapter_start through 12 (inclusive, 1-indexed).

    All trainable components live inside self.lidar_modules for clean federation.

    Args:
        clip_visual   : CLIP's VisionTransformer. Weights are deep-copied and
                        frozen; the original is not modified.
        adapter_dim   : bottleneck width for each AdapterBottleneck layer.
        adapter_start : first transformer block index to inject adapters (1-indexed).
    """

    def __init__(
        self,
        clip_visual:   nn.Module,
        adapter_dim:   int = 64,
        adapter_start: int = 5,
    ):
        super().__init__()

        self.backbone      = copy.deepcopy(clip_visual)
        self.backbone.requires_grad_(False)

        width              = clip_visual.transformer.width          # 768 for ViT-B/16
        num_blocks         = self.backbone.transformer.layers       # 12  for ViT-B/16
        num_adapter_layers = num_blocks - adapter_start + 1        # 8   (blocks 5–12)
        self.adapter_start = adapter_start

        self.lidar_modules = LiDARTrainableModules(
            width              = width,
            adapter_dim        = adapter_dim,
            num_adapter_layers = num_adapter_layers,
        )

    def freeze_stem(self):
        """Freeze the input stem at the phase-2 boundary.

        After this call, the stem receives no gradients and federated
        aggregation should switch to encoder.lidar_modules.adapters.
        """
        self.lidar_modules.stem.requires_grad_(False)

    @property
    def dtype(self):
        """Dtype of the frozen backbone (fp16 by default after convert_weights)."""
        return self.backbone.conv1.weight.dtype

    def forward(
        self,
        x:              torch.Tensor,
        shared_adapter: Optional[nn.Module] = None,
        scale:          float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            x             : LiDAR range image [B, 3, H, W]
            shared_adapter: intra-block shared adapter (None in current design)
            scale         : gradient scale passed through adapter_func closure

        Returns:
            patch tokens [B, 196, output_dim]
        """
        x        = self.lidar_modules.stem(x)
        x        = x.type(self.dtype)
        adapters = self.lidar_modules.adapters
        start    = self.adapter_start

        def adapter_func(layer_idx: int):
            if layer_idx < start:
                return None, None, scale
            return adapters[layer_idx - start], shared_adapter, scale

        return self.backbone([x, adapter_func])
