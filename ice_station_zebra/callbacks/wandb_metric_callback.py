import wandb
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class WandbMetric(Callback):
    """Configure Wandb to plot train and val losses together."""

    def on_train_start(self, _trainer: Trainer, _pl_module: LightningModule) -> None:
        """Configure metric grouping when training starts."""
        if wandb.run is not None:
            wandb.define_metric("train_loss", step_metric="epoch")
            wandb.define_metric("validation_loss", step_metric="epoch")
