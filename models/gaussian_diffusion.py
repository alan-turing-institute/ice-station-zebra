"""
GaussianDiffusion: Gaussian Diffusion Process Implementation

Author: Maria Carolina Novitasari 

Description:
    Implements the forward and reverse processes of a Denoising Diffusion Probabilistic Model (DDPM),
    with support for both cosine and linear beta schedules.

    Key Definitions:
    - Beta (βₜ): Variance schedule controlling the amount of noise added at each timestep.
      A small βₜ means less noise; a large βₜ means more noise.
    - Alpha (αₜ): Defined as (1 - βₜ), representing the retained signal at each step.
    - Alpha Cumprod (ᾱₜ): Cumulative product of alphas over time, representing the overall
      signal preservation from step 0 to t.
"""


import torch
import torch.nn.functional as F
import math
from typing import Tuple

class GaussianDiffusion:
    """
    Implements the forward and reverse processes of a Denoising Diffusion Probabilistic Model (DDPM),
    including support for cosine and linear beta schedules.
    """
    
    def __init__(self, timesteps: int = 1000, beta_schedule: str = 'cosine'):
        """
        Initialize diffusion parameters and precompute useful constants.

        Args:
            timesteps (int): Total number of diffusion steps.
            beta_schedule (str): Type of beta schedule to use. Options: 'linear', 'cosine'.
        """
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(timesteps)
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Compute beta schedule using a cosine function.
    
        Args:
            timesteps (int): Total number of timesteps.
            s (float): Small offset to prevent singularities near 0.
                      Controls how fast the signal decays: a smaller s results in faster
                      corruption early on; a larger s gives a smoother, slower decay.
    
        Returns:
            torch.Tensor: Beta values of shape (timesteps,).
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        return torch.clip(betas, 0, 0.999)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Add noise to x_start at timestep t, using the forward diffusion process.

        Args:
            x_start (torch.Tensor): Original input tensor (clean image).
            t (torch.Tensor): Timesteps for each sample in the batch (shape: [B]).
            noise (torch.Tensor, optional): Noise to add. If None, standard Gaussian noise is used.

        Returns:
            torch.Tensor: Noisy sample at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        """
        Perform a single reverse diffusion (denoising) step.

        Args:
            x (torch.Tensor): Current noisy sample at timestep t.
            t (torch.Tensor): Timesteps for each sample in the batch (shape: [B]).
            pred_noise (torch.Tensor): Model's predicted noise (εθ) for x at timestep t.

        Returns:
            torch.Tensor: Sample from the previous timestep (x_{t-1}).
        """
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper (our pred_noise is εθ)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
        
        # Create mask for where t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        # Only add noise if t != 0
        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise
        
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]) -> torch.Tensor:
        """
        Extract values from a tensor at specific timesteps t and reshape for broadcasting.

        Args:
            a (torch.Tensor): 1D tensor containing precomputed values (e.g., alpha or beta schedule).
            t (torch.Tensor): Timesteps for each sample in the batch (shape: [B]).
            x_shape (Tuple[int]): Target shape for broadcasting (same as input sample x).

        Returns:
            torch.Tensor: Extracted and reshaped values for each timestep in the batch.
        """
        a = a.to(t.device)
        out = a[t]  # (batch_size,) # Reshape for broadcasting: [batch_size, 1, 1, 1, 1]
      
        return out.view((t.shape[0],) + (1,) * (len(x_shape) - 1))