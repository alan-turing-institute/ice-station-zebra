import logging
import statistics
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

if TYPE_CHECKING:
    from torchmetrics import MetricCollection

from icenet_mp.types import ModelTestOutput

logger = logging.getLogger(__name__)


class MetricSummaryCallback(Callback):
    """A callback to summarise metrics during evaluation."""

    def __init__(self, *, average_loss: bool = True) -> None:
        """Summarise metrics during evaluation.

        Args:
            average_loss: Whether to log average loss

        """
        self.metrics: dict[str, list[float]] = {}
        if average_loss:
            self.metrics["average_loss"] = []

    def on_test_batch_end(
        self,
        _trainer: Trainer,
        _module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        _batch: Any,  # noqa: ANN401
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends."""
        if not isinstance(outputs, ModelTestOutput):
            msg = f"Output is of type {type(outputs)}, skipping metric accumulation."
            logger.warning(msg)
            return

        if "average_loss" in self.metrics:
            self.metrics["average_loss"].append(outputs.loss.item())

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        _module: LightningModule,
    ) -> None:
        """Called at the end of the test epoch."""
        # Post-process accumulated metrics into a single value
        metrics_: dict[str, float] = {}
        for name, values in self.metrics.items():
            if name.startswith("average_") and values:
                metrics_[name] = statistics.mean(values)

        # Log metrics to each logger
        for logger_ in trainer.loggers:
            logger_.log_metrics(metrics_)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of testing."""
        test_metrics: MetricCollection = pl_module.test_metrics  # type: ignore[assignment]
        if not hasattr(test_metrics, "items"):
            logger.warning(
                "test_metrics does not have an items() method, skipping metric summary."
            )
            return

        for name, metric in test_metrics.items():
            # Compute the metric value (e.g., SIEError) across all batches and log it
            values = metric.compute()

            for logger_ in trainer.loggers:
                # check if WandB is being used as a logger, and if so, log the metric values as a table and plot
                if isinstance(logger_, WandbLogger):
                    # If the metric returns a value for each day, log it as a W&B table and plot it
                    if values.numel() > 1:
                        table = wandb.Table(
                            data=list(enumerate(values.tolist(), start=1)),
                            columns=["day", name],
                        )
                    plot_name = name + " per day"
                    wandb.log(
                        {
                            plot_name: wandb.plot.line(
                                table, "day", name, title=plot_name
                            )
                        }
                    )

                # Log the mean value of the metric across all days
                logger_.log_metrics({f"{name} (mean)": values.mean().item()})
