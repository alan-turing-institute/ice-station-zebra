import torch
import torch.nn as nn
import numpy as np


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, img_size):
        """
        Initialize patch embedding layer that converts image patches into embeddings.
        
        Args:
            in_channels (int): Number of input channels in the image
            patch_size (int): Size of each square patch
            emb_dim (int): Embedding dimension for each patch
            img_size (int): Size of input image (assume 432)
        """
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.patches_per_side = img_size // patch_size
        self.num_patches = self.patches_per_side ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """
        Convert input image into patch embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Patch embeddings of shape [B, N, emb_dim] where N is number of patches
        """
        x = self.proj(x)  # [B, emb_dim, patches_per_side, patches_per_side]
        x = x.flatten(2).transpose(1, 2)  # [B, N, emb_dim]
        return x
