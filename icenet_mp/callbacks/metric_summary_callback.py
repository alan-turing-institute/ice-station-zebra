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
        # Per-day MAE over forecast horizon (shape: [B, T, C, H, W])
        per_day_mae = torch.mean(torch.abs(prediction - target), dim=(0, 2, 3, 4))
        for t, mae in enumerate(per_day_mae, start=1):
            key = f"mae_day_{t}"
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(mae.detach().cpu().item())

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        _module: LightningModule,
    ) -> None:
        """Called at the end of the test epoch."""
        # Post-process accumulated metrics into a single value
        metrics_: dict[str, float] = {}
        for name, values in self.metrics.items():
            if name.startswith("average_") or name.startswith("mae_day_"):
                metrics_[name] = statistics.mean(values)

        print(self.metrics)
        print(metrics_)

        # Log metrics to each logger
        for logger in trainer.loggers:
            logger.log_metrics(metrics_)

        # Build W&B table for per-day MAE and plot results
        mae_rows: list[list[float]] = []
        for name, value in metrics_.items():
            if name.startswith("mae_day_"):
                day = int(name.split("_")[-1])
                mae_rows.append([day, value])

        if mae_rows:
            mae_rows.sort(key=lambda r: r[0])
            table = wandb.Table(data=mae_rows, columns=["day", "mae"])
            print(table.data)
            # wandb.log({"mae_by_day": table})
            plot_name = "MAE per day"
            wandb.log(
                {plot_name: wandb.plot.line(table, "day", "mae", title=plot_name)}
            )
