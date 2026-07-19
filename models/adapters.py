"""
models/adapters.py
------------------
Shared adapter modules for all three branches of CustomCLIP.

AdapterBottleneck  — per-layer bottleneck injected into transformer blocks;
                     used identically for visual (δv), text (δt), and LiDAR (δℓ)
SharedTaskAdapter  — post-fusion cross-task bottleneck applied to fused patch tokens
"""

import torch
import torch.nn as nn


class AdapterBottleneck(nn.Module):
    """
    Bottleneck adapter injected into transformer blocks.

    Protocol consumed by ModifiedResidualAttentionBlock:

        seq_adapter, shared_adapter, scale = adapter_func(layer_idx)
        y = seq_adapter.down(x)   # width → adapter_dim  (Linear + ReLU)
        y = seq_adapter.up(y)     # adapter_dim → width  (Linear, zero-init)
        x = x + y * scale

    up is zero-initialised so the adapter is a no-op at the start of training.
    Used identically for all three modality branches (δv / δt / δℓ).
    """

    def __init__(self, width: int, adapter_dim: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Linear(width, adapter_dim),
            nn.ReLU(),
        )
        self.up = nn.Linear(adapter_dim, width)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.down[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.down[0].bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)


class SharedTaskAdapter(nn.Module):
    """
    Post-fusion cross-task adapter applied to fused image+LiDAR patch tokens.

    Residual bottleneck (dim → shared_adapter_dim → dim) that accumulates
    cross-task knowledge and acts as a transfer anchor for arriving tasks.
    Zero-initialised up projection ensures identity at training start.

    Args:
        dim               : token width (512 for CLIP ViT-B/16 output_dim)
        shared_adapter_dim: bottleneck hidden dim — kept independent from the
                            per-branch adapter_dim to allow separate ablations
    """

    def __init__(self, dim: int = 512, shared_adapter_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, shared_adapter_dim)
        self.act  = nn.GELU()
        self.up   = nn.Linear(shared_adapter_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, 196, 512] → [B, 196, 512]"""
        return x + self.up(self.act(self.down(x)))
