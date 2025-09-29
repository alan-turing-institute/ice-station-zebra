from typing import Any

import torch
from torch_ema import ExponentialMovingAverage  # type: ignore[import]

from ice_station_zebra.models.diffusion import GaussianDiffusion, UNetDiffusion
from ice_station_zebra.types import TensorNCHW

from .base_processor import BaseProcessor


class DDPMProcessor(BaseProcessor):
    """Denoising Diffusion Probabilistic Model (DDPM).

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
    """

    def __init__(self, timesteps: int = 1000, **kwargs: Any) -> None:
        """Initialize the DDPM processor.

        Args:
            timesteps: Num diffusion timesteps (default=1000).
            kwargs: Arguments to BaseProcessor.

        """
        super().__init__(**kwargs)
        self.model = UNetDiffusion(self.data_space.channels, timesteps)
        self.timesteps = timesteps
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        self.diffusion = GaussianDiffusion(timesteps=timesteps)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Generate NCHW output with diffusion for a single timestep.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        """
        sample_weight = torch.ones_like(x[..., :1])

        y_bchw = self.sample(x, sample_weight)

        return (y_bchw + 1.0) / 2.0

    def sample(
        self,
        x: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Perform reverse diffusion sampling starting from noise.

        Args:
            x (torch.Tensor): Conditioning input [B, H, W, C].
            sample_weight (torch.Tensor or None): Optional weights.

        Returns:
            torch.Tensor: Denoised output of shape [B, H, W, C].

        """
        # Start from pure noise
        y = torch.rand_like(x)

        dim_threshold = 3

        # Use EMA weights for sampling
        with self.ema.average_parameters():
            for t in reversed(range(self.timesteps)):
                t_batch = torch.full_like(x[:, 0, 0, 0], t, dtype=torch.long)
                pred_v: torch.Tensor = self.model(y, t_batch, x, sample_weight)
                pred_v = (
                    pred_v.squeeze(3)
                    if pred_v.dim() > dim_threshold
                    else pred_v.squeeze()
                )
                y = self.diffusion.p_sample(y, t_batch, pred_v)

        return y
