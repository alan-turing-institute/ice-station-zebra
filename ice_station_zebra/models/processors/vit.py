# import torch
# import torch.nn as nn
# from torch_ema import ExponentialMovingAverage
# from ice_station_zebra.models.vit.vit import ViTForSeaIce
# from ice_station_zebra.types import TensorNCHW


# class VitProcessor(nn.Module):
#     def __init__(self,
#                  img_size: int,
#                  patch_size: int,
#                  in_channels:int,
#                  out_channels:int,
#                  emb_dim: int,
#                  depth: int,
#                  heads: int,
#                  mlp_dim: int,
#                  dropout: float,
#                  n_forecast_days: int,
#                  ema_decay: float):
#         super().__init__()

#         self.model = ViTForSeaIce(
#             img_size=img_size,
#             patch_size=patch_size,
#             in_channels=in_channels,
#             out_channels=out_channels,
#             emb_dim=emb_dim,
#             depth=depth,
#             heads=heads,
#             mlp_dim=mlp_dim,
#             dropout=dropout,
#             n_forecast_days=n_forecast_days
#         )

#         self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
#         self.n_forecast_days = n_forecast_days
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#     def forward(self, x: TensorNCHW) -> TensorNCHW:
#         with self.ema.average_parameters():
#             y_hat = torch.sigmoid(self.model(x)) # model output (B, H, W, out_channels)
#             y_hat = y_hat.movedim(-1,1) # (B, out_channel, H, W)
            
#             return y_hat
"""
Vision Transformer implementation

Author: Wei (Erin) Quan

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
        
        # Hardcode argument values for now
        # img_size = 192
        # patch_size = 48
        # in_channels = n_latent_channels
        # emb_dim = 256
        # depth = 4
        # heads = 2
        # mlp_dim = 512
        # dropout = 0.66
        
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

        # (B, N, patch_size * patch_size * out_channels * n_forecast_days)
        # self.decoder = nn.Sequential(
        #     nn.Linear(emb_dim, patch_size * patch_size * (out_channels * n_forecast_days)),
        # )
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
        # Input: (B, H, W, C)
        # B, H, W, C = x.shape
        # already in B, C, H, W
        B, C, H, W = x.shape
        
        # Convert to (B, C, H, W) for patch embedding
        # already B, C, H, W, then comment out next line
        # x = x.permute(0, 3, 1, 2)
        
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
        
        # x = x.permute(0, 3, 4, 2, 1)  # (B, H, W, n_forecast_days, out_channels) old
        
        return x