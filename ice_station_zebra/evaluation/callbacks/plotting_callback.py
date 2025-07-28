import logging
from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torch import Tensor
from torch.utils.data import DataLoader

from ice_station_zebra.visualisations import plot_sic_comparison
from ice_station_zebra.data.lightning import CombinedDataset

logger = logging.getLogger(__name__)


class PlottingCallback(Callback):
    """A callback to create plots during evaluation."""

    def __init__(
        self, frequency: int = 10, plot_sea_ice_concentration: bool = True
    ) -> None:
        """Create plots during evaluation.

        Args:
            frequency: Create a new plot every `frequency` batches.
            plot_sea_ice_concentration: Whether to plot sea ice concentration.
        """
        super().__init__()
        self.frequency = frequency
        self.plot_fns = {}
        if plot_sea_ice_concentration:
            self.plot_fns["sea-ice-comparison"] = plot_sic_comparison

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
        # Run plotting every `frequency` batches
        if batch_idx % self.frequency == 0:
            # Get date for this batch
            batch_size = outputs["target"].shape[0]
            test_dataloaders: DataLoader | list[DataLoader] | None = (
                trainer.test_dataloaders
            )
            if test_dataloaders is None:
                logger.debug("No test dataloaders found, skipping plotting.")
                return
            dataset: CombinedDataset = (
                test_dataloaders[dataloader_idx]
                if isinstance(test_dataloaders, list)
                else test_dataloaders
            ).dataset  # type: ignore[assignment]
            date_ = dataset.date_from_index(batch_size * batch_idx)

            # Load the ground truth and prediction
            np_ground_truth = outputs["target"].cpu().numpy()[0, 0, :, :]
            np_prediction = outputs["output"].cpu().numpy()[0, 0, :, :]

            # Create each requested plot
            images = {
                name: plot_fn(np_ground_truth, np_prediction, date_)
                for name, plot_fn in self.plot_fns.items()
            }

            # Log images to each logger
            for lightning_logger in trainer.loggers:
                for key, image_list in images.items():
                    if hasattr(lightning_logger, "log_image"):
                        lightning_logger.log_image(key=key, images=image_list)
                    else:
                        logger.debug(
                            f"Logger {lightning_logger.name} does not support logging images."
                        )
