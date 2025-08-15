import statistics
from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch import Callback

from ice_station_zebra.types import ModelTestOutput


class MetricSummaryCallback(Callback):
    """A callback to summarise metrics during evaluation."""

    def __init__(self, average_loss: bool = True) -> None:
        """Summarise metrics during evaluation.

        Args:
            average_loss: Whether to log average loss
        """
        self.metrics: dict[str, list[float]] = {}
        if average_loss:
            self.metrics["average_loss"] = []

    def on_test_batch_end(
        self,
        trainer: Trainer,
        module: LightningModule,
        outputs: ModelTestOutput,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends."""
        if "average_loss" in self.metrics:
            self.metrics["average_loss"].append(outputs.loss.item())

    def on_test_epoch_end(self, trainer: Trainer, module: LightningModule) -> None:
        """Called at the end of the test epoch."""
        # Post-process accumulated metrics into a single value
        metrics_: dict[str, float] = {}
        for name, values in self.metrics.items():
            if name.startswith("average_"):
                metrics_[name] = statistics.mean(values)

        # Log metrics to each logger
        for logger in trainer.loggers:
            logger.log_metrics(metrics_)
