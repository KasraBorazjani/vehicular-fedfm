"""
models/fusion.py
----------------
Standalone image-LiDAR patch-token fusion module.

Extracted from the 3D task heads so the fusion mechanism can be replaced
independently — swap this file to try cross-attention, gated fusion, etc.
The downstream heads (SegHead3D, DetHead3D) receive already-fused tokens
and have no knowledge of the fusion strategy used.
"""

import torch
import torch.nn as nn

CLIP_DIM = 512


class ImageLidarFusion(nn.Module):
    """
    Concat-project fusion: [image_patches | lidar_patches] → fused_patches.

    Takes image and LiDAR patch tokens of equal shape and projects their
    concatenation back to the original dimension.

    Input  : image_patches [B, N, D],  lidar_patches [B, N, D]
    Output : fused_patches [B, N, D]
    """

    def __init__(self, dim: int = CLIP_DIM):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        image_patches: torch.Tensor,   # [B, N, D]
        lidar_patches: torch.Tensor,   # [B, N, D]
    ) -> torch.Tensor:                  # [B, N, D]
        return self.proj(torch.cat([image_patches, lidar_patches], dim=-1))
