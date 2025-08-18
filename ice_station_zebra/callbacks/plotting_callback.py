import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torch import Tensor

from ice_station_zebra.data_loaders import CombinedDataset
from ice_station_zebra.types import ModelTestOutput
from ice_station_zebra.visualisations import plot_sic_comparison

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PlottingCallback(Callback):
    """A callback to create plots during evaluation."""

    def __init__(
        self, *, frequency: int = 10, plot_sea_ice_concentration: bool = True
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
        _module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        _batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends."""
        # Only run plotting every `frequency` batches
        if batch_idx % self.frequency:
            return

        # Check that outputs is a ModelTestOutput
        if not isinstance(outputs, ModelTestOutput):
            msg = f"Output is of type {type(outputs)}, skipping plotting."
            logger.warning(msg)
            return

        # Get date for this batch
        dl: DataLoader | list[DataLoader] | None = trainer.test_dataloaders
        if dl is None:
            logger.warning("No test dataloaders found, skipping plotting.")
            return
        dataset = (dl[dataloader_idx] if isinstance(dl, Sequence) else dl).dataset
        if not isinstance(dataset, CombinedDataset):
            logger.warning("Dataset is not a CombinedDataset, skipping plotting.")
            return
        batch_size = outputs.target.shape[0]
        date_ = dataset.date_from_index(batch_size * batch_idx)

        # Load the ground truth and prediction
        # Both prediction and target are TensorNTCHW
        np_ground_truth = outputs.target.cpu().numpy()[0, 0, 0, :, :]
        np_prediction = outputs.prediction.cpu().numpy()[0, 0, 0, :, :]

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
                        "Logger %s does not support logging images.",
                        lightning_logger.name,
                    )
