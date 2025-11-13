import os

# Unset SLURM_NTASKS if it's causing issues
if "SLURM_NTASKS" in os.environ:
    del os.environ["SLURM_NTASKS"]

# Optionally, set SLURM_NTASKS_PER_NODE if needed
os.environ["SLURM_NTASKS_PER_NODE"] = "1"  # or whatever value is appropriate


from typing import Any

import torch
# Force all new tensors to be float32 by default
torch.set_default_dtype(torch.float32)
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage  # type: ignore[import]
from torchmetrics import MetricCollection

from ice_station_zebra.models.diffusion import GaussianDiffusion, UNetDiffusion
from ice_station_zebra.types import ModelTestOutput, TensorNTCHW, DataSpace

from .losses import WeightedMSELoss
from .metrics import IceNetAccuracy, SIEError
from .zebra_model import ZebraModel


class DDPM(ZebraModel):
    """Denoising Diffusion Probabilistic Model (DDPM).

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
    """

    # def __init__(self, timesteps: int = 1000, **kwargs: Any) -> None:
    def __init__(self, 
                 timesteps: int = 1000, 
                 learning_rate: float = 5e-4, 
                 start_out_channels: int = 32, 
                 kernel_size: int = 3, 
                 activation: str = "SiLU", 
                 normalization: str = "groupnorm",
                 n_output_classes: int = 1,
                 time_embed_dim:int = 256,
                 dropout_rate:float = 0.1,
                 **kwargs: Any) -> None:

        """Initialize the DDPM processor.

        Args:
            timesteps: Num diffusion timesteps (default=1000).
            kwargs: Arguments to BaseProcessor.

        """
        super().__init__(**kwargs)

        era5_space = next(space for space in self.input_spaces if 
                          (space['name'] if isinstance(space, dict) else space.name) == 'era5')
        
        # Get channels from either dict or object
        if isinstance(era5_space, dict):
            self.era5_space = era5_space['channels']
        else:
            self.era5_space = era5_space.channels

        # osisaf channels after squeezing: T_osisaf = self.n_history_steps (probably 1 here)
        osisaf_channels = self.n_history_steps  # keep T dimension as channels
        
        # era5 channels after merging time*channels
        era5_channels = self.n_history_steps * self.era5_space  # T * C2
        
        # total channels for UNet
        self.input_channels = osisaf_channels + era5_channels
        self.n_output_classes = n_output_classes
        self.output_channels = self.n_forecast_steps * self.n_output_classes
        self.timesteps = timesteps
        
        # self.model = UNetDiffusion(self.input_channels, self.output_channels, self.timesteps)
        self.model = UNetDiffusion(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            timesteps=self.timesteps,
            kernel_size=kernel_size,
            start_out_channels=start_out_channels,
            time_embed_dim=time_embed_dim,  
            normalization=normalization,
            activation=activation,
            dropout_rate=dropout_rate
        )
 
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        
        self.diffusion = GaussianDiffusion(timesteps=timesteps)

        # Move all diffusion schedules to the correct device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.diffusion.betas = self.diffusion.betas.to(device)
            self.diffusion.alphas = self.diffusion.alphas.to(device)
            self.diffusion.alphas_cumprod = self.diffusion.alphas_cumprod.to(device)
            self.diffusion.alphas_cumprod_prev = self.diffusion.alphas_cumprod_prev.to(device)
            self.diffusion.sqrt_alphas_cumprod = self.diffusion.sqrt_alphas_cumprod.to(device)
            self.diffusion.sqrt_one_minus_alphas_cumprod = self.diffusion.sqrt_one_minus_alphas_cumprod.to(device)
            self.diffusion.sqrt_recip_alphas = self.diffusion.sqrt_recip_alphas.to(device)
            self.diffusion.posterior_variance = self.diffusion.posterior_variance.to(device)
            self.diffusion.posterior_log_variance_clipped = self.diffusion.posterior_log_variance_clipped.to(device)
            self.diffusion.posterior_mean_coef1 = self.diffusion.posterior_mean_coef1.to(device)
            self.diffusion.posterior_mean_coef2 = self.diffusion.posterior_mean_coef2.to(device)
            self.ema.to(device)

        self.learning_rate = learning_rate

        metrics = {
            "val_accuracy": IceNetAccuracy(
                leadtimes_to_evaluate=list(range(self.n_forecast_steps))
            ),
            "val_sieerror": SIEError(
                leadtimes_to_evaluate=list(range(self.n_forecast_steps))
            ),
        }
        for i in range(self.n_forecast_steps):
            metrics[f"val_accuracy_{i}"] = IceNetAccuracy(leadtimes_to_evaluate=[i])
            metrics[f"val_sieerror_{i}"] = SIEError(leadtimes_to_evaluate=[i])
        self.metrics = MetricCollection(metrics)

        test_metrics = {
            "test_accuracy": IceNetAccuracy(
                leadtimes_to_evaluate=list(range(self.n_forecast_steps))
            ),
            "test_sieerror": SIEError(
                leadtimes_to_evaluate=list(range(self.n_forecast_steps))
            ),
        }
        for i in range(self.n_forecast_steps):
            test_metrics[f"test_accuracy_{i}"] = IceNetAccuracy(
                leadtimes_to_evaluate=[i]
            )
            test_metrics[f"test_sieerror_{i}"] = SIEError(leadtimes_to_evaluate=[i])
        self.test_metrics = MetricCollection(test_metrics)

        self.save_hyperparameters()

    def forward(self, batch: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Generate a single NCHW output with diffusion.

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
            x (torch.Tensor): Conditioning input [B, C, H, W].
            sample_weight (torch.Tensor or None): Optional weights.

        Returns:
            torch.Tensor: Denoised output of shape [B, C, H, W].

        """
        shape = (x.shape[0], self.n_forecast_steps * self.n_output_classes, *x.shape[-2:])
        device = x.device

        # Start from pure noise
        y = torch.randn(shape, device=device)

        dim_threshold = 3

        # Use EMA weights for sampling
        with self.ema.average_parameters():
            for t in reversed(range(self.timesteps)):
                t_batch = torch.full_like(x[:, 0, 0, 0], t, dtype=torch.long, device=device)
                pred_v: torch.Tensor = self.model(y, t_batch, x)
                pred_v = (
                    pred_v.squeeze(3)
                    if pred_v.dim() > dim_threshold
                    else pred_v.squeeze()
                )
                y = self.diffusion.p_sample(y, t_batch, pred_v)

        return y

  