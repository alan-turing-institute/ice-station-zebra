"""
train_diffusion_icenet: Trainer for Denoising Diffusion Model

Author: Maria Carolina Novitasari

Description:
    Defines the `train_diffusion_icenet` function to train a UNet-based denoising diffusion model (DDPM)
    for sea ice concentration forecasting using PyTorch Lightning.

    This training utility handles:
    - Dataset loading for training and validation modes
    - DataLoader preparation with reproducibility and batching
    - UNetDiffusion model instantiation with configurable architecture
    - Loss function and training callbacks (checkpointing and early stopping)
    - PyTorch Lightning trainer setup and model fitting
"""

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from icenet.data.dataset import IceNetDataSetPyTorch

from models import UNetDiffusion, LitDiffusion, GaussianDiffusion
from utils import WeightedMSELoss

def train_diffusion_icenet(configuration_path,
                          learning_rate,
                          max_epochs,
                          batch_size,
                          n_workers,
                          filter_size,
                          n_filters_factor,
                          seed,
                          timesteps=1000,
                          persistent_workers=True):
    """
    Train IceNet diffusion model using the specified parameters.

    Args:
        configuration_path (str): Path to IceNet configuration YAML file.
        learning_rate (float): Learning rate for optimizer.
        max_epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        n_workers (int): Number of workers for data loading.
        filter_size (int): Convolution kernel size used in UNet layers.
        n_filters_factor (float): Scaling factor for number of filters in UNet.
        seed (int): Random seed for reproducibility.
        timesteps (int): Number of diffusion steps (T).

    Returns:
        tuple: (model, trainer, checkpoint_callback)
            model (UNetDiffusion): Trained model.
            trainer (pl.Trainer): PyTorch Lightning trainer used for training.
            checkpoint_callback (ModelCheckpoint): Callback used for saving the best model.
    """
    # init
    pl.seed_everything(seed)
    
    # configure datasets and dataloaders
    train_dataset = IceNetDataSetPyTorch(configuration_path, mode="train")
    val_dataset = IceNetDataSetPyTorch(configuration_path, mode="val")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=n_workers,
                                 persistent_workers=persistent_workers, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=n_workers,
                               persistent_workers=persistent_workers, shuffle=False)

    #mc debug
    # Check the shape of a batch of data from the train dataloader
    for batch in train_dataloader:
        # Assuming the batch contains (x, y, sample_weight)
        x, y, sample_weight = batch
        break  # We only need to inspect one batch

    # construct diffusion model
    model = UNetDiffusion(
        input_channels=train_dataset._num_channels,
        filter_size=filter_size,
        n_filters_factor=n_filters_factor,
        n_forecast_days=train_dataset._n_forecast_days,
        timesteps=timesteps
    )
    
    criterion = WeightedMSELoss(reduction="none")
    # checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
    
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',  
        patience=150,
        verbose=True,
        mode='max')
    
    # trainer.callbacks.append(checkpoint_callback, early_stop_callback)
    
    # configure PyTorch Lightning module
    lit_module = LitDiffusion(
        model=model,
        learning_rate=learning_rate,
        criterion=criterion,
        timesteps=timesteps
    )

    # set up trainer configuration
    trainer = pl.Trainer(
        accelerator="auto",
        devices=-1,
        log_every_n_steps=5,
        max_epochs=max_epochs,
        num_sanity_val_steps=1,
        fast_dev_run=False,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # train model
    print(f"Training {len(train_dataset)} examples / {len(train_dataloader)} batches (batch size {batch_size}).")
    print(f"Validating {len(val_dataset)} examples / {len(val_dataloader)} batches (batch size {batch_size}).")
    trainer.fit(lit_module, train_dataloader, val_dataloader)

    return model, trainer, checkpoint_callback