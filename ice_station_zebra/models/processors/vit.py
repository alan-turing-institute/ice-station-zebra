"""Vision Transformer implementation.

Author: Erin Quan

Description:
    Vision Transformer (ViT) model for sea ice forecasting that predicts future sea ice
    concentration from meteorological data. Takes multi-channel input images, converts
    them to patch embeddings, processes through transformer encoder blocks, and outputs
    spatially-resolved predictions for specified forecast horizons.
"""

from typing import Any

import torch
from torch import nn

from ice_station_zebra.models.common import PatchEmbedding, TransformerEncoderBlock
from ice_station_zebra.types import TensorNCHW

from .base_processor import BaseProcessor


class VitProcessor(BaseProcessor):
    def __init__(
        self,
        *,
        depth: int,
        dropout: float,
        emb_dim: int,
        heads: int,
        mlp_dim: int,
        patch_size: int,
        **kwargs: Any,
    ) -> None:
        """Initialize Vision Transformer model for sea ice forecasting."""
        super().__init__(**kwargs)

        # Ensure input is square
        if self.data_space.shape[0] != self.data_space.shape[1]:
            msg = "The height and width of the input are not equal."
            raise ValueError(msg)

        self.img_size = self.data_space.shape[0]
        self.patch_size = patch_size
        self.out_channels = self.data_space.channels

        self.patch_embed = PatchEmbedding(
            self.data_space.channels, patch_size, emb_dim, self.img_size
        )
        num_patches = (self.img_size // patch_size) ** 2

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, emb_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(emb_dim, heads, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(emb_dim)

        # (B, N, patch_size * patch_size * out_channels *)
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, patch_size * patch_size * self.out_channels),
        )

        self.smooth = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=3, padding=1
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward pass through the ViT model for a single timestep.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        """
        batch, _, height, _ = x.shape

        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x)  # (B, N, D)
        x = self.decoder(x)  # (B, N, out_channels**patch_size*patch_size)

        h_patches = w_patches = height // self.patch_size
        x = x.reshape(
            batch,
            h_patches,
            w_patches,
            self.out_channels,
            self.patch_size,
            self.patch_size,
        )
        # Shape is batch, out_channels, h_patches, patch_size, w_patches, patch_size
        x = x.permute(0, 3, 1, 4, 2, 5)

        # Shape is batch, out_channels, height, width
        x = x.reshape(batch, self.out_channels, self.img_size, self.img_size)

        return self.smooth(x)
