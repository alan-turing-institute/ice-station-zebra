from lightning.pytorch.callbacks import Callback
import wandb


class WandbMetricConfig(Callback):
    """Configure Wandb to plot train and val losses together."""
    
    def on_train_start(self, trainer, pl_module):
        """Configure metric grouping when training starts."""
        if wandb.run is not None:
            wandb.define_metric("loss/*", step_metric="epoch")