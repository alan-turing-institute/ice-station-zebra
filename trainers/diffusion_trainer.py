"""
trainers/train_diffusion_hydra.py: Hydra-integrated Trainer for the DDPM

Author: Maria Carolina Novitasari 

Description:
    Defines the `DiffusionTrainer` class to train a UNet-based Denoising Diffusion Probabilistic Model (DDPM) for sea ice concentration forecasting using PyTorch Lightning and Hydra configuration management.

    This training utility handles:
    - Hydra configuration management
    - Dataset loading for training and validation modes
    - DataLoader preparation with reproducibility and batching
    - UNetDiffusion model instantiation with configurable architecture
    - Loss function and training callbacks (checkpointing and early stopping)
    - PyTorch Lightning trainer setup and model fitting
"""

import logging
from pathlib import Path
import hydra
from lightning import Callback, Trainer
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from icenet.data.dataset import IceNetDataSetPyTorch
from models.unet_diffusion import UNetDiffusion
from models.lit_diffusion import LitDiffusion  
from models.gaussian_diffusion import GaussianDiffusion
from utils.losses import WeightedMSELoss

logger = logging.getLogger(__name__)

class DiffusionTrainer:
    """A wrapper for PyTorch Lightning diffusion model training with Hydra configuration"""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize the Diffusion trainer."""
        self.config = config
        
        # Set seed for reproducibility
        pl.seed_everything(config.train.seed)
        
        # Configure datasets and dataloaders
        self._setup_data()
        
        # Construct the diffusion model
        self._setup_model()
        
        # Setup logging and callbacks
        self._setup_logging_and_callbacks()
        
        # Construct the trainer
        self._setup_trainer()
        
    def _setup_data(self) -> None:
        """Setup training and validation datasets and dataloaders."""
        logger.info("Setting up datasets and dataloaders...")
        
        # Create datasets
        self.train_dataset = IceNetDataSetPyTorch(
            self.config.data.configuration_path, 
            mode="train"
        )
        self.val_dataset = IceNetDataSetPyTorch(
            self.config.data.configuration_path, 
            mode="val"
        )
        
        # Create dataloaders
        dataloader_config = self.config.train.dataloader
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=dataloader_config.batch_size,
            num_workers=dataloader_config.n_workers,
            persistent_workers=dataloader_config.get('persistent_workers', True),
            shuffle=dataloader_config.get('shuffle', False)
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size=dataloader_config.batch_size,
            num_workers=dataloader_config.n_workers,
            persistent_workers=dataloader_config.get('persistent_workers', True),
            shuffle=False
        )
        
        # Debug: Check batch shape
        for batch in self.train_dataloader:
            x, y, sample_weight = batch
            logger.info(f"Batch shapes - X: {x.shape}, Y: {y.shape}, Weights: {sample_weight.shape}")
            break
            
    def _setup_model(self) -> None:
        """Setup the diffusion model and Lightning module."""
        logger.info("Setting up diffusion model...")
        
        # Construct UNet diffusion model
        model_config = self.config.model
        self.model = UNetDiffusion(
            input_channels=self.train_dataset._num_channels,
            filter_size=model_config.filter_size,
            n_filters_factor=model_config.n_filters_factor,
            n_forecast_days=self.train_dataset._n_forecast_days,
            timesteps=model_config.timesteps
        )
        
        # Setup loss function
        self.criterion = WeightedMSELoss(reduction="none")
        
        # Configure PyTorch Lightning module
        self.lit_module = LitDiffusion(
            model=self.model,
            learning_rate=self.config.train.optimizer.learning_rate,
            criterion=self.criterion,
            timesteps=model_config.timesteps
        )
        
    def _setup_logging_and_callbacks(self) -> None:
        """Setup logging and training callbacks."""
        logger.info("Setting up logging and callbacks...")
        
        # Construct lightning loggers
        self.lightning_loggers = []
        # if "loggers" in self.config:
        if hasattr(self.config, "loggers") and self.config.loggers is not None:
            for logger_name, logger_config in self.config.loggers.items():
                self.lightning_loggers.append(
                    hydra.utils.instantiate(
                        dict(
                            {
                                "job_type": "train",
                                "project": f"diffusion_{self.config.model.get('name', 'icenet')}",
                            },
                            **logger_config,
                        )
                    )
                )
        
        # Setup run directory
        self.run_directory = self._get_run_directory()
        
        # Setup callbacks
        self.callbacks = []
        
        # Checkpoint callback
        checkpoint_config = self.config.train.callbacks.get('checkpoint', {})
        self.checkpoint_callback = ModelCheckpoint(
            monitor=checkpoint_config.get('monitor', 'val_accuracy'),
            mode=checkpoint_config.get('mode', 'max'),
            dirpath=self.run_directory / "checkpoints",
            # **{k: v for k, v in checkpoint_config.items() if k not in ['monitor', 'mode']}
            **{k: v for k, v in checkpoint_config.items() if k not in ['monitor', 'mode', '_target_']}
        )
        self.callbacks.append(self.checkpoint_callback)
        
        # Early stopping callback
        if 'early_stopping' in self.config.train.callbacks:
            early_stop_config = self.config.train.callbacks.early_stopping
            early_stop_callback = EarlyStopping(
                monitor=early_stop_config.get('monitor', 'val_accuracy'),
                patience=early_stop_config.get('patience', 25),
                verbose=early_stop_config.get('verbose', True),
                mode=early_stop_config.get('mode', 'max')
            )
            self.callbacks.append(early_stop_callback)
            
        # Add any additional callbacks from config
        for callback_name, callback_config in self.config.train.callbacks.items():
            if callback_name not in ['checkpoint', 'early_stopping']:
                self.callbacks.append(hydra.utils.instantiate(callback_config))
                
    def _get_run_directory(self) -> Path:
        """Get the run directory for saving outputs."""
        # Try to get from wandb logger if available
        wandb_logger = self._get_wandb_logger()
        if wandb_logger:
            return Path(wandb_logger.experiment._settings.sync_dir)
        else:
            # Generate local run directory
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return Path("./training/local") / f"diffusion_run_{timestamp}"
            
    def _get_wandb_logger(self):
        """Get wandb logger if available."""
        for logger in self.lightning_loggers:
            if hasattr(logger, 'experiment') and hasattr(logger.experiment, '_settings'):
                return logger
        return None
        
    def _setup_trainer(self) -> None:
        """Setup PyTorch Lightning trainer."""
        logger.info("Setting up PyTorch Lightning trainer...")
        
        # Construct the trainer with Hydra configuration
        trainer_config = self.config.train.trainer
        self.trainer = Trainer(
            accelerator=trainer_config.get('accelerator', 'auto'),
            devices=trainer_config.get('devices', -1),
            log_every_n_steps=trainer_config.get('log_every_n_steps', 5),
            max_epochs=trainer_config.get('max_epochs', 100),
            num_sanity_val_steps=trainer_config.get('num_sanity_val_steps', 1),
            fast_dev_run=trainer_config.get('fast_dev_run', False),
            callbacks=self.callbacks,
            logger=self.lightning_loggers if self.lightning_loggers else None,
            **{k: v for k, v in trainer_config.items() 
               if k not in ['accelerator', 'devices', 'log_every_n_steps', 'max_epochs', 
                           'num_sanity_val_steps', 'fast_dev_run']}
        )
        
        # Save config to the output directory
        self.run_directory.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, self.run_directory / "model_config.yaml")
        
    def train(self) -> tuple:
        """
        Train the diffusion model.
        
        Returns:
            tuple: (model, trainer, checkpoint_callback)
        """
        logger.info("Starting training...")
        logger.info(f"Training {len(self.train_dataset)} examples / {len(self.train_dataloader)} batches "
                   f"(batch size {self.config.train.dataloader.batch_size}).")
        logger.info(f"Validating {len(self.val_dataset)} examples / {len(self.val_dataloader)} batches "
                   f"(batch size {self.config.train.dataloader.batch_size}).")
        
        # Train model
        self.trainer.fit(
            model=self.lit_module,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )
        
        return self.model, self.trainer, self.checkpoint_callback
