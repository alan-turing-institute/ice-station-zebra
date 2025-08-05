"""
LitDiffusion: PyTorch Lightning Module for DDPM-based Forecasting

Author: Maria Carolina Novitasari

Description:
    PyTorch Lightning wrapper for training, validating, and testing a conditional
    denoising diffusion probabilistic model (DDPM), designed for geophysical forecasting tasks
    (e.g., sea ice concentration). Handles training loops, sampling logic, loss,
    and Exponential Moving Average (EMA) updates. Supports metric evaluation and 
    forecast generation conditioned on external inputs (e.g., meteorological variables).
"""

# Core PyTorch & Lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

# Metrics & Training Utilities
from torchmetrics import MetricCollection
from torch_ema import ExponentialMovingAverage

# Relative Imports (assuming your project structure)
from .unet_diffusion import UNetDiffusion  # Your model class
from .gaussian_diffusion import GaussianDiffusion
from utils import IceNetAccuracy, SIEError  # Custom metrics
from utils import WeightedMSELoss  # Or your chosen loss

class LitDiffusion(pl.LightningModule):
    """
    PyTorch Lightning wrapper for training and evaluating a diffusion-based model
    (e.g., DDPM) for conditional forecasting. Handles training loop, sampling, 
    metrics, and optimizer configuration.
    """
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float,
                 criterion: callable,
                 timesteps: int = 1000):
        """
        Initialize the LightningModule for DDPM training.

        Args:
            model (nn.Module): The U-Net-style diffusion model.
            learning_rate (float): Optimizer learning rate.
            criterion (callable): Loss function used for evaluation (e.g., WeightedBCE or WeightedMSE).
            timesteps (int): Number of diffusion steps (T).
        """
        super().__init__()
        
        self.model = model
        self.learning_rate = learning_rate
        self.timesteps = timesteps
        self.diffusion = GaussianDiffusion(timesteps=timesteps)
        self.criterion = criterion
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)  

        self.n_output_classes = model.n_output_classes

        metrics = {
            "val_accuracy": IceNetAccuracy(leadtimes_to_evaluate=list(range(self.model.n_forecast_days))),
            "val_sieerror": SIEError(leadtimes_to_evaluate=list(range(self.model.n_forecast_days)))
        }
        for i in range(self.model.n_forecast_days):
            metrics[f"val_accuracy_{i}"] = IceNetAccuracy(leadtimes_to_evaluate=[i])
            metrics[f"val_sieerror_{i}"] = SIEError(leadtimes_to_evaluate=[i])
        self.metrics = MetricCollection(metrics)

        test_metrics = {
            "test_accuracy": IceNetAccuracy(leadtimes_to_evaluate=list(range(self.model.n_forecast_days))),
            "test_sieerror": SIEError(leadtimes_to_evaluate=list(range(self.model.n_forecast_days)))
        }
        for i in range(self.model.n_forecast_days):
            test_metrics[f"test_accuracy_{i}"] = IceNetAccuracy(leadtimes_to_evaluate=[i])
            test_metrics[f"test_sieerror_{i}"] = SIEError(leadtimes_to_evaluate=[i])
        self.test_metrics = MetricCollection(test_metrics)

        self.save_hyperparameters()

    def on_fit_start(self):
        """
        Hook called at the beginning of training (`fit()`).
    
        Ensures that the shadow parameters used in the Exponential Moving Average (EMA)
        are moved to the same device as the model. This prevents device mismatch errors 
        during EMA updates, which require both the shadow and actual model parameters 
        to be on the same device.
    
        Notes:
            - `torch_ema.ExponentialMovingAverage` stores shadow parameters in a list.
            - This method is called by PyTorch Lightning before the first training step.
        """
        device = next(self.model.parameters()).device
        print(f"[LitDiffusion] Moving EMA shadow parameters to device: {device}")
        self.ema.to(device)

    def forward(self, x):
        """
        Run the model in inference mode.

        Args:
            x (torch.Tensor or tuple): Either a tensor [B, H, W, C] or a batch from DataLoader.

        Returns:
            torch.Tensor: Generated sample(s).
        """
        if isinstance(x, (list, tuple)):
            x = x[0]
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)

        if sample_weight is None:
            sample_weight = torch.ones_like(x[..., :1])

        outputs = self.sample(x, sample_weight)
        
        y_hat = (outputs + 1.0) / 2.0
        y_hat = torch.clamp(y_hat, 0, 1)
        
        return y_hat

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

    def sample(self, x, sample_weight, num_samples=1):
        """
        Perform reverse diffusion sampling starting from noise.

        Args:
            x (torch.Tensor): Conditioning input [B, H, W, C].
            sample_weight (torch.Tensor or None): Optional weights.
            num_samples (int): Not used (for future batching).

        Returns:
            torch.Tensor: Final denoised output [B, H, W, C_out].
        """
        shape = (x.shape[0], *x.shape[1:-1], self.model.n_forecast_days * self.n_output_classes)
        device = x.device
        
        # Start from pure noise
        y = torch.randn(shape, device=device)
        
        for t in reversed(range(0, self.timesteps)):
            t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
            # pred_noise = self.model(y, t_batch, x, sample_weight)
            # pred_noise = pred_noise.squeeze(3)
            # y = self.diffusion.p_sample(y, t_batch, pred_noise)
            
            pred_v = self.model(y, t_batch, x, sample_weight)
            pred_v = pred_v.squeeze(3)  # Remove extra dimension if needed
            y = self.diffusion.p_sample(y, t_batch, pred_v)
            
        return y

    def on_validation_epoch_end(self):
        """
        Called at the end of validation to log all collected metrics.
        """
        self.log_dict(self.metrics.compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.metrics.reset()

    def on_test_epoch_end(self):
        """
        Called at the end of test loop to log all collected metrics.
        """
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics.reset()

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
            weight_decay=1e-5,  # L2 regularization strength
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',           # For val_loss
                factor=0.8,           # If the metric stops improving, reduce LR by half
                patience=10,           # Number of epochs with no improvement to wait before reducing LR
                verbose=True
            ),
            'monitor': 'val_loss',    # Metric to watch
            'interval': 'epoch',
            'frequency': 1,
            'name': 'lr'
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_closure,
                       on_tpu=False,
                       using_native_amp=False,
                       using_lbfgs=False):
        """
        Custom optimizer step used in PyTorch Lightning.
    
        This method performs a single optimization step and updates the 
        Exponential Moving Average (EMA) of the model weights.
    
        Args:
            epoch (int): Current epoch number.
            batch_idx (int): Index of the current batch.
            optimizer (Optimizer): The optimizer being used (e.g., Adam).
            optimizer_idx (int): Index of the optimizer (useful when multiple optimizers are used).
            optimizer_closure (callable): A closure that reevaluates the model and returns the loss.
            on_tpu (bool): Whether the training is running on a TPU.
            using_native_amp (bool): Whether automatic mixed precision (AMP) is being used.
            using_lbfgs (bool): Whether the LBFGS optimizer is being used.
    
        Notes:
            - PyTorch Lightning calls this method automatically during training.
            - `self.ema.update()` updates the EMA shadow weights after the optimizer step.
        """
        optimizer.step(closure=optimizer_closure)
        self.ema.update()  # Update EMA weights after optimizer step
        
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
        y_hat = torch.sigmoid(outputs)
            
        loss = self.criterion(y_hat, y, sample_weight)

        return y_hat
