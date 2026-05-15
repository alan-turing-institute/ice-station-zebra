import logging
from typing import TYPE_CHECKING

import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torchmetrics import MetricCollection

from icenet_mp.utils import get_wandb_run

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class MetricSummaryCallback(Callback):
    """A callback to summarise metrics during evaluation."""

    def log_per_epoch_metrics(
        self, trainer: Trainer, metrics: MetricCollection
    ) -> None:
        """Log per-epoch metrics to W&B."""
        for name, metric in metrics.items():
            # Compute the metric value (e.g., SIEError) across all batches
            values: Tensor = metric.compute()

            # Log the mean value of the metric across all days
            for logger_ in trainer.loggers:
                logger_.log_metrics(
                    {
                        f"{name} (mean)": values.mean().item(),
                        "epoch": trainer.current_epoch,
                    }
                )

    def log_per_run_metrics(self, trainer: Trainer, metrics: MetricCollection) -> None:
        """Log per-run metrics to W&B."""
        for name, metric in metrics.items():
            # Compute the metric value (e.g., SIEError) across all batches
            values: Tensor = metric.compute()

            # Check if WandB is being used as a logger and metrics are calculated for
            # multiple days if so, log the metric values as a table and plot.
            if (
                isinstance(run := get_wandb_run(trainer), wandb.Run)
                and values.numel() > 1
            ):
                table = wandb.Table(
                    data=list(enumerate(values.tolist(), start=1)),
                    columns=["day", name],
                )
                plot_name = name + " per day"
                run.log(
                    {plot_name: wandb.plot.line(table, "day", name, title=plot_name)}
                )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of a test epoch."""
        if isinstance(pl_module.test_metrics, MetricCollection):
            self.log_per_epoch_metrics(trainer, pl_module.test_metrics)
        else:
            logger.warning("Could not load train metrics!")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of testing."""
        if isinstance(pl_module.test_metrics, MetricCollection):
            self.log_per_run_metrics(trainer, pl_module.test_metrics)
        else:
            logger.warning("Could not load test metrics!")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of a training epoch."""
        if isinstance(pl_module.train_metrics, MetricCollection):
            self.log_per_epoch_metrics(trainer, pl_module.train_metrics)
        else:
            logger.warning("Could not load train metrics!")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of training."""
        if isinstance(pl_module.train_metrics, MetricCollection):
            self.log_per_run_metrics(trainer, pl_module.train_metrics)
        else:
            logger.warning("Could not load train metrics!")
