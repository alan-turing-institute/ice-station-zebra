from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torch import Tensor

from ice_station_zebra.visualisations import plot_sic_comparison
from ice_station_zebra.utils import get_wandb_logger


class PlottingCallback(Callback):
    """A callback to create plots during evaluation."""

    def on_test_batch_end(
        self,
        trainer: Trainer,
        module: LightningModule,
        outputs: dict[str, Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends."""
        # Plot the sea ice concentration for every 10 batches
        if batch_idx % 10:
            batch_size = outputs["target"].shape[0]
            try:
                dataloader = trainer.test_dataloaders[dataloader_idx]
            except TypeError:
                dataloader = trainer.test_dataloaders
            date_ = dataloader.dataset.date_from_index(batch_size * batch_idx)
            np_ground_truth = outputs["target"].cpu().numpy()[0, 0, :, :]
            np_prediction = outputs["output"].cpu().numpy()[0, 0, :, :]
            img_sic_comparison = plot_sic_comparison(
                target=np_ground_truth, prediction=np_prediction, date=date_
            )
            if wandb_logger := get_wandb_logger(trainer.loggers):
                wandb_logger.log_image(key="sea-ice-comparison", images=[img_sic_comparison])
