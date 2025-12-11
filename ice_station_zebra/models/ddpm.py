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
        - ERA5 input: T historical steps × number of channels, resized to OSISAF resolution

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

        era5_space = next(
            space
            for space in self.input_spaces
            if (space["name"] if isinstance(space, dict) else space.name) == "era5"
        )

        # Get channels from either dict or object
        if isinstance(era5_space, dict):
            self.era5_space = era5_space["channels"]
        else:
            self.era5_space = era5_space.channels

        # Get the base output channels from output_space
        if isinstance(self.output_space, dict):
            self.base_output_channels = self.output_space["channels"]
        else:
            self.base_output_channels = self.output_space.channels

        # osisaf channels after squeezing: T_osisaf = self.n_history_steps
        osisaf_channels = self.n_history_steps  # keep T dimension as channels

        # era5 channels after merging time*channels
        era5_channels = self.n_history_steps * self.era5_space  # T * C2

        # total channels for UNet
        self.input_channels = osisaf_channels + era5_channels
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

        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)

        self.diffusion = GaussianDiffusion(timesteps=timesteps)

        # Move all diffusion schedules to the correct device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.diffusion.betas = self.diffusion.betas.to(device)
            self.diffusion.alphas = self.diffusion.alphas.to(device)
            self.diffusion.alphas_cumprod = self.diffusion.alphas_cumprod.to(device)
            self.diffusion.alphas_cumprod_prev = self.diffusion.alphas_cumprod_prev.to(
                device
            )
            self.diffusion.sqrt_alphas_cumprod = self.diffusion.sqrt_alphas_cumprod.to(
                device
            )
            self.diffusion.sqrt_one_minus_alphas_cumprod = (
                self.diffusion.sqrt_one_minus_alphas_cumprod.to(device)
            )
            self.diffusion.sqrt_recip_alphas = self.diffusion.sqrt_recip_alphas.to(
                device
            )
            self.diffusion.posterior_variance = self.diffusion.posterior_variance.to(
                device
            )
            self.diffusion.posterior_log_variance_clipped = (
                self.diffusion.posterior_log_variance_clipped.to(device)
            )
            self.diffusion.posterior_mean_coef1 = (
                self.diffusion.posterior_mean_coef1.to(device)
            )
            self.diffusion.posterior_mean_coef2 = (
                self.diffusion.posterior_mean_coef2.to(device)
            )
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
        y = torch.randn(shape, device=device)

        dim_threshold = 3

        # Use EMA weights for sampling
        with self.ema.average_parameters():
            for t in reversed(range(self.timesteps)):
                t_batch = torch.full_like(
                    x[:, 0, 0, 0], t, dtype=torch.long, device=device
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
        """Merge time dimension into channels for ERA5, squeeze singleton channels for osisaf,and combine them for forecasting together.

        Args:
            batch: Dictionary containing keys 'osisaf-south' and 'era5'.

        Returns:
            TensorNTCHW: Combined input of shape [B, C_combined, H, W].

        """
        osisaf = batch[self.osisaf_key]  # [B, T, 1, H, W]
        era5 = batch["era5"]  # [B, T, 27, H2, W2]

        # pylint: disable=invalid-name
        B, T, C2, H2, W2 = era5.shape  # noqa: N806
        H_target, W_target = osisaf.shape[-2:]  # noqa: N806

        # Squeeze singleton channel for osisaf: [B, T, H, W]
        osisaf_squeezed = osisaf.squeeze(2)

        # Merge ERA5 time and channels: [B, T*C2, H2, W2]
        era5_merged = era5.view(B, T * C2, H2, W2)

        # Resize to osisaf spatial resolution
        era5_resized = F.interpolate(
            era5_merged, size=(H_target, W_target), mode="bilinear", align_corners=False
        )

        # Flatten osisaf time dimension into channels: [B, T, H, W] → [B, T, H, W] already
        # Concatenate along channels: [B, T + T*C2, H, W]
        return torch.cat([osisaf_squeezed, era5_resized], dim=1)

    def training_step(self, batch: dict[str, TensorNTCHW]) -> dict:
        """One training step using DDPM loss (predicted noise vs. true noise)."""
        device = batch["target"].device

        # Prepare input tensor by combining osisaf-south and era5
        x = self.prepare_inputs(batch).to(device)  # [B, T, C_combined, H, W]

        # Extract target
        y = batch["target"].squeeze(2).to(device)

        # Scale target to [-1, 1]
        y_scaled = 2.0 * y - 1.0

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device).long()

        # Create noisy version using scaled target
        noise = torch.randn_like(y_scaled)
        noisy_y = self.diffusion.q_sample(y_scaled, t, noise)

        # Predict v
        pred_v = self.model(noisy_y, t, x)  # .squeeze()

        # Compute target v using scaled data
        target_v = self.diffusion.calculate_v(y_scaled, noise, t)

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

        # Generate samples (returns [-1, 1] range)
        outputs = self.sample(x, sample_weight)

        # Convert to [0, 1] for metrics and loss
        y_hat = (outputs + 1.0) / 2.0
        y_hat = torch.clamp(y_hat, 0, 1)

        # Calculate loss in [0, 1] space
        loss = self.loss(y_hat, y, sample_weight)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Update metrics in [0, 1] space
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

        # Convert to [0, 1] for metrics and loss
        y_hat = (outputs + 1.0) / 2.0
        y_hat = torch.clamp(y_hat, 0, 1)

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

    def configure_optimizers(self) -> dict[str, Any]:
        """Set up the optimizer.

        Returns:
            torch.optim.Optimizer: Adam optimizer.

        """
        # Add weight decay to discourage large weights (helps generalization)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100, eta_min=self.learning_rate * 0.01
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "lr",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def optimizer_step(
        self,
        epoch: int,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
        optimizer: Optimizer,
        optimizer_closure: Callable[[], None],
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Custom optimizer step used in PyTorch Lightning.

        This method performs a single optimization step using the provided closure,
        and updates the Exponential Moving Average (EMA) of the model weights.

        Args:
            epoch (int): Current epoch number.
            batch_idx (int): Index of the current batch.
            optimizer (Optimizer): The optimizer instance (e.g., Adam).
            optimizer_closure (callable): A closure that reevaluates the model and returns the loss.
            **kwargs: Additional arguments passed by PyTorch Lightning (not used here).

        Notes:
            - This method is automatically called by PyTorch Lightning during training.
            - `self.ema.update()` updates the EMA shadow weights after the optimizer step.

        """
        optimizer.step(closure=optimizer_closure)
        self.ema.update()
