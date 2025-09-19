from typing import Any

import torch
from torch_ema import ExponentialMovingAverage  # type: ignore[import]
import torch.nn.functional as F

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
        self.model = UNetDiffusion(self.n_latent_channels_total, timesteps)
        self.timesteps = timesteps
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        self.diffusion = GaussianDiffusion(timesteps=timesteps)

    def rollout(self, x: TensorNCHW) -> TensorNCHW:
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

    def training_step(self, batch):
        """
        One training step using DDPM loss (predicted noise vs. true noise).

        Args:
            batch (tuple): (x, y, sample_weight).

        Returns:
            dict: {"loss": loss}
        """
        x, y, sample_weight = batch
        y = y.squeeze(-1)
        
        # Scale target to [-1, 1] 
        y_scaled = 2.0 * y - 1.0
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device).long()
        
        # Create noisy version using scaled target
        noise = torch.randn_like(y_scaled)
        noisy_y = self.diffusion.q_sample(y_scaled, t, noise)
        
        # Predict v
        pred_v = self.model(noisy_y, t, x, sample_weight)
        pred_v = pred_v.squeeze()
        
        # Compute target v using scaled data
        target_v = self.diffusion.calculate_v(y_scaled, noise, t)
        
        loss = F.mse_loss(pred_v, target_v)
        
        if self.global_step % 100 == 0:
            print(f"Step {self.global_step}: Loss {loss.item():.4f}")
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch):
        """
        One validation step using the specified loss function defined in the criterion.

        Args:
            batch (tuple): (x, y, sample_weight)

        Returns:
            dict: {"val_loss": loss}
        """
        x, y, sample_weight = batch
        y = y.squeeze(-1)  # Removes the last dimension (size 1)
        
        # Generate samples (returns [-1, 1] range)
        outputs = self.sample(x, sample_weight)
        
        # Convert to [0, 1] for metrics and loss
        y_hat = (outputs + 1.0) / 2.0
        y_hat = torch.clamp(y_hat, 0, 1)
        
        # Calculate loss in [0, 1] space
        loss = self.criterion(y_hat, y, sample_weight)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Update metrics in [0, 1] space
        self.metrics.update(y_hat, y, sample_weight)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        """
        One test step using the specified loss function and full metric evaluation.

        Args:
            batch (tuple): (x, y, sample_weight)
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y, sample_weight = batch
        y = y.squeeze(-1)
        
        outputs = self.sample(x, sample_weight)
        
        # Convert to [0, 1] for metrics and loss
        y_hat = (outputs + 1.0) / 2.0
        y_hat = torch.clamp(y_hat, 0, 1)
        
        loss = self.criterion(y_hat, y, sample_weight)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.test_metrics.update(y_hat, y, sample_weight)
        
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Perform prediction on a single batch.

        Args:
            batch (tuple): (x, y, sample_weight).
            batch_idx (int): Batch index.
            dataloader_idx (int): Index of the DataLoader (if using multiple).

        Returns:
            torch.Tensor: Prediction tensor [B, H, W, C_out].
        """
        x, y, sample_weight = batch
        y = y.squeeze(-1)  # Removes the last dimension (size 1)
        
        outputs = self.sample(x, sample_weight)
        
        # Convert to [0, 1] for final predictions
        y_hat = (outputs + 1.0) / 2.0
        y_hat = torch.clamp(y_hat, 0, 1)
        
        return y_hat

    def configure_optimizers(self):
        """
        Set up the optimizer.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        # Add weight decay to discourage large weights (helps generalization)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,  
            betas=(0.9, 0.95),  
            eps=1e-8
        )
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=100,
                eta_min=self.learning_rate * 0.01
            ),
            'interval': 'epoch',
            'frequency': 1,
            'name': 'lr'
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        """
        Custom optimizer step used in PyTorch Lightning.
    
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