"""
Vision Transformer implementation

Author: Erin Quan

Description:
    Vision Transformer (ViT) model for sea ice forecasting that predicts future sea ice 
    concentration from meteorological data. Takes multi-channel input images, converts 
    them to patch embeddings, processes through transformer encoder blocks, and outputs 
    spatially-resolved predictions for specified forecast horizons.
"""
import torch
import torch.nn as nn
import numpy as np
from ice_station_zebra.models.common.patchembed import PatchEmbedding
from ice_station_zebra.models.common.transformerblock import TransformerEncoderBlock

class VitProcessor(nn.Module):
    def __init__(
        self,
        n_latent_channels: int,
        start_out_channels: int,
        img_size: int,
        patch_size: int,
        emb_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float
    ) -> None:
        """Initialize Vision Transformer model for sea ice forecasting."""
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_per_side = img_size // patch_size
        self.out_channels = start_out_channels
        self.emb_dim = emb_dim

    
        
        self.patch_embed = PatchEmbedding(n_latent_channels, patch_size, emb_dim, img_size)
        num_patches = (img_size // patch_size) ** 2
        
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, emb_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(emb_dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )
        
        self.norm = nn.LayerNorm(emb_dim)

        # (B, N, patch_size * patch_size * out_channels *)
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, patch_size * patch_size * self.out_channels),
        )
        
    def forward(self, x):  
        """
        Forward pass through the ViT model for sea ice forecasting before decoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x)  # (B, N, D)

        x = self.decoder(x)  # (B, N, out_channels**patch_size*patch_size)
        
        H_patches = W_patches = int(np.sqrt(x.shape[1]))
        
        x = x.reshape(B, H_patches, W_patches, self.out_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, out_channels, H_patches, patch_size, W_patches, patch_size)
        x = x.reshape(B, self.out_channels, self.img_size, self.img_size) # (B, C, H, W)

        
        return x