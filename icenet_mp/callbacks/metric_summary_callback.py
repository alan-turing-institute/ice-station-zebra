import logging
import statistics
from collections.abc import Mapping
from typing import Any

import torch
import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torch import Tensor

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

        prediction = outputs.prediction
        target = outputs.target
        print(f"shapes: prediction={prediction.shape}, target={target.shape}")
        
    def on_test_epoch_end(
        self,
        trainer: Trainer,
        _module: LightningModule,
    ) -> None:
        """Called at the end of the test epoch."""
        # Post-process accumulated metrics into a single value
        metrics_: dict[str, float] = {}
        for name, values in self.metrics.items():
            if name.startswith("average_"): 
                metrics_[name] = statistics.mean(values)

        print(self.metrics)
        print(metrics_)

        # Log metrics to each logger
        for logger in trainer.loggers:
            print("Logging metrics to logger: ", logger)
            logger.log_metrics(metrics_)

         
    def on_test_end(
        self, trainer: Trainer, pl_module: LightningModule ) -> None: 
        """Called at the end of testing.""" 
        test_metrics_: dict[str, float] = {}
        for name, metric in pl_module.test_metrics.items():
            print("name: ", name)
            print("metric: ", metric)
            
            # Compute the metric value (e.g., SIEError) across all batches and log it
            values = metric.compute()
        
            print("test end values:", values)
            # If the metric returns a value for each day, log it as a W&B table and plot it
            if values.numel() > 1:
                table = wandb.Table(data=list(enumerate(values.tolist(), start=1)), columns=["day", name])
                print(table.data)
                plot_name = name + " per day"
                wandb.log(
                    {plot_name: wandb.plot.line(table, "day", name, title=plot_name)}
                )
                
                test_metrics_[name] = torch.mean(values).item()  # Log the mean value across days

                print("test_metrics_:", test_metrics_)

        # Log metrics to each logger
        for logger in trainer.loggers:
            print("Logging metrics to logger: ", logger)
            logger.log_metrics(test_metrics_)