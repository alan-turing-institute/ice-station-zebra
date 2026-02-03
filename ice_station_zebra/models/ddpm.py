# mypy: ignore-errors
import os
from collections.abc import Callable
from typing import Any, NoReturn

import torch
import torch.nn.functional as F  # noqa: N812
from torch.optim import Optimizer
from torch_ema import ExponentialMovingAverage  # type: ignore[import]
from torchmetrics import MetricCollection

from ice_station_zebra.models.diffusion import GaussianDiffusion, UNetDiffusion
from ice_station_zebra.types import ModelTestOutput, TensorNTCHW

from .losses import WeightedMSELoss
from .metrics import IceNetAccuracy, SIEError
from .zebra_model import ZebraModel

# Unset SLURM_NTASKS if it's causing issues
if "SLURM_NTASKS" in os.environ:
    del os.environ["SLURM_NTASKS"]

# Optionally, set SLURM_NTASKS_PER_NODE if needed
os.environ["SLURM_NTASKS_PER_NODE"] = "1"

# Force all new tensors to be float32 by default
torch.set_default_dtype(torch.float32)


class DDPM(ZebraModel):
    """Denoising Diffusion Probabilistic Model (DDPM).

    Input space:
        TensorNTCHW with shape (batch_size, n_history_steps + n_history_steps * n_era5_channels, height, width)
        - OSISAF input: T historical steps, singleton channel squeezed
        - ERA5 input: T historical steps times number of channels, resized to OSISAF resolution

    Output space:
        TensorNTCHW with shape (batch_size, n_forecast_steps * n_output_channels, height, width)
        - Forecasted outputs per timestep and channel, flattened along the channel dimension
    """

    def __init__(  # noqa: PLR0913
        self,
        timesteps: int = 1000,
        learning_rate: float = 5e-4,
        start_out_channels: int = 32,
        kernel_size: int = 3,
        activation: str = "SiLU",
        normalization: str = "groupnorm",
        time_embed_dim: int = 256,
        dropout_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize the DDPM processor.

        Args:
            timesteps (int): Number of diffusion timesteps. Default is 1000.
            learning_rate (float): Optimizer learning rate for training. Default is 5e-4.
            start_out_channels (int): Base number of channels in the first UNet block.
            kernel_size (int): Convolution kernel size used in the UNet.
            activation (str): Activation function used throughout the network (e.g., "SiLU").
            normalization (str): Normalization layer type (e.g., "groupnorm").
            time_embed_dim (int): Dimensionality of the timestep embedding.
            dropout_rate (float): Dropout probability applied inside the UNet blocks.
            **kwargs: Additional arguments passed to ``ZebraModel``.

        """
        super().__init__(**kwargs)

        self.osisaf_key = self.output_space.name

        # era5_space = next(
        #     space
        #     for space in self.input_spaces
        #     if (space["name"] if isinstance(space, dict) else space.name) == "era5"
        # )

        # # Get channels from either dict or object
        # if isinstance(era5_space, dict):
        #     self.era5_space = era5_space["channels"]
        # else:
        #     self.era5_space = era5_space.channels

        # # Downweight ERA5 conditioning 
        # self.era5_weight = 0.25 #0.1 #0.25

        # # Get the base output channels from output_space
        # if isinstance(self.output_space, dict):
        #     self.base_output_channels = self.output_space["channels"]
        # else:
        #     self.base_output_channels = self.output_space.channels
        self.base_output_channels = self.output_space.channels

        # osisaf channels after squeezing: T_osisaf = self.n_history_steps
        osisaf_channels = self.n_history_steps  # keep T dimension as channels

        # # era5 channels after merging time*channels
        # era5_channels = self.n_history_steps * self.era5_space  # T * C2

        # total channels for UNet
        # self.input_channels = osisaf_channels + era5_channels
        self.input_channels = osisaf_channels #+ era5_channels
        self.output_channels = self.n_forecast_steps * self.base_output_channels
        self.timesteps = timesteps

        self.model = UNetDiffusion(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            timesteps=self.timesteps,
            kernel_size=kernel_size,
            start_out_channels=start_out_channels,
            time_embed_dim=time_embed_dim,
            normalization=normalization,
            activation=activation,
            dropout_rate=dropout_rate,
        )

        self.diffusion = GaussianDiffusion(timesteps=timesteps)

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

    def forward(self, *args: Any, **kwargs: Any) -> NoReturn:
        msg = "This model uses `training_step`, `validation_step`, and `test_step` instead of `forward()`"
        raise NotImplementedError(msg)

    def sample(
        self,
        x: torch.Tensor,
        sample_weight: torch.Tensor | None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Perform reverse diffusion sampling starting from noise.

        Args:
            x (torch.Tensor): Conditioning input [B, C, H, W].
            sample_weight (torch.Tensor or None): Optional weights.

        Returns:
            torch.Tensor: Denoised output of shape [B, C, H, W].

        """
        shape = (
            x.shape[0],
            self.n_forecast_steps * self.base_output_channels,
            *x.shape[-2:],
        )
        device = x.device

        # Start from pure noise
        y = torch.randn(shape, device=self.device)

        dim_threshold = 3

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full_like(
                x[:, 0, 0, 0], t, dtype=torch.long, device=self.device
            )
            pred_v: torch.Tensor = self.model(y, t_batch, x)
            pred_v = (
                pred_v.squeeze(3)
                if pred_v.dim() > dim_threshold
                else pred_v.squeeze()
            )
            y = self.diffusion.p_sample(y, t_batch, pred_v)

        return y

    def loss(
        self,
        prediction: TensorNTCHW,
        target: TensorNTCHW,
        sample_weight: TensorNTCHW | None = None,
    ) -> torch.Tensor:
        if sample_weight is None:
            sample_weight = torch.ones_like(prediction)
        return WeightedMSELoss(reduction="none")(prediction, target, sample_weight)

    def prepare_inputs(self, batch: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """
        Prepare OSI-SAF-only inputs.
    
        Returns:
            Tensor of shape [B, T, H, W]
        """
        osisaf = batch[self.osisaf_key]  # [B, T, 1, H, W]
    
        # Squeeze singleton channel
        osisaf_squeezed = osisaf.squeeze(2)  # [B, T, H, W]
    
        return osisaf_squeezed


    def training_step(self, batch: dict[str, TensorNTCHW]) -> dict:
        """One training step using DDPM loss (predicted noise vs. true noise)."""

        # Prepare input tensor by combining osisaf-south and era5
        x = self.prepare_inputs(batch) # [B, T, C_combined, H, W]

        # Extract target
        y = batch["target"].squeeze(2) 

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=self.device).long() #look into this

        # Create noisy version using scaled target
        noise = torch.randn_like(y)
        noisy_y = self.diffusion.q_sample(y, t, noise)

        # Predict v
        pred_v = self.model(noisy_y, t, x)

        # Compute target v using scaled data
        target_v = self.diffusion.calculate_v(y, noise, t)

        # Compute loss
        loss = F.mse_loss(pred_v, target_v)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {"loss": loss}

    def validation_step(self, batch: dict[str, TensorNTCHW]) -> dict:
        """One validation step using the specified loss function defined in the criterion."""
        # Prepare input tensor
        x = self.prepare_inputs(batch)  # [B, T, C_combined, H, W]

        # Extract target and optional weights
        y = batch["target"].squeeze(2)  # [B, T, H, W]
        sample_weight = batch.get("sample_weight", torch.ones_like(y))

        # Generate samples
        outputs = self.sample(x, sample_weight)

        y_hat = torch.clamp(outputs, 0, 1)

        # Calculate loss 
        loss = self.loss(y_hat, y, sample_weight)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Update metrics
        self.metrics.update(y_hat, y, sample_weight)
        return {"val_loss": loss}

    def test_step(
        self,
        batch: dict[str, TensorNTCHW],
        _batch_idx: int,  # noqa: PT019
    ) -> ModelTestOutput:
        """One test step using the specified loss function and full metric evaluation.

        Args:
            batch (tuple): (x, y, sample_weight)
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.

        """
        x = self.prepare_inputs(batch)  # [B, T, C_combined, H, W]
        y = batch["target"].squeeze(2)
        sample_weight = batch.get("sample_weight", torch.ones_like(y))

        outputs = self.sample(x, sample_weight)

        y_hat = torch.clamp(outputs, 0, 1).unsqueeze(2)
        
        y = y.unsqueeze(2)
        sample_weight = sample_weight.unsqueeze(2)

        loss = self.loss(y_hat, y, sample_weight)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.test_metrics.update(y_hat, y, sample_weight)

        return ModelTestOutput(prediction=y_hat, target=y, loss=loss)
