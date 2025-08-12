import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage

from ice_station_zebra.models.diffusion import GaussianDiffusion, UNetDiffusion
from ice_station_zebra.types import TensorNCHW


class DDPMProcessor(nn.Module):
    """Denoising Diffusion Probabilistic Model (DDPM)

    Operations all occur in latent space:
        TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)
    """

    def __init__(self, n_latent_channels: int, timesteps: int = 1000) -> None:
        super().__init__()
        self.model = UNetDiffusion(n_latent_channels, timesteps)
        self.timesteps = timesteps
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        self.diffusion = GaussianDiffusion(timesteps=timesteps)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """
        Transformation summary

        Args:
            x: TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)
        """
        x_bhwc = x.movedim(1, -1)
        sample_weight = torch.ones_like(x_bhwc[..., :1])

        y_bhwc = self.sample(x_bhwc, sample_weight)

        y_bchw = y_bhwc.movedim(-1, 1)
        y_hat = (y_bchw + 1.0) / 2.0
        y_hat = torch.clamp(y_hat, 0, 1)

        return y_hat

    def sample(self, x, sample_weight, num_samples=1):
        """
        Perform reverse diffusion sampling starting from noise.

        Args:
            x (torch.Tensor): Conditioning input [B, H, W, C].
            sample_weight (torch.Tensor or None): Optional weights.
            num_samples (int): Not used (for future batching).

        Returns:
            torch.Tensor: Denoised output of shape [B, H, W, C].
        """
        # Start from pure noise
        y = torch.rand_like(x)

        # Use EMA weights for sampling
        with self.ema.average_parameters():
            for t in reversed(range(0, self.timesteps)):
                t_batch = torch.full_like(x[:, 0, 0, 0], t, dtype=torch.long)
                pred_v: torch.Tensor = self.model(y, t_batch, x, sample_weight)
                pred_v = pred_v.squeeze(3) if pred_v.dim() > 3 else pred_v.squeeze()
                y = self.diffusion.p_sample(y, t_batch, pred_v)

        return y
