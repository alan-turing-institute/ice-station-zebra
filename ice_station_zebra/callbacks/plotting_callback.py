import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from omegaconf import DictConfig
from torch import Tensor

from ice_station_zebra.data_loaders import CombinedDataset
from ice_station_zebra.types import ModelTestOutput
from ice_station_zebra.utils import datetime_from_npdatetime
from ice_station_zebra.visualisations import DEFAULT_SIC_SPEC, PlotSpec, Plotter

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PlottingCallback(Callback):
    """A callback to create plots during evaluation."""

    def __init__(
        self,
        *,
        frequency: int = 10,
        make_static_plots: bool = True,
        make_video_plots: bool = True,
        plot_spec: PlotSpec | None = None,
        base_path: str,
    ) -> None:
        """Create plots during evaluation.

        Args:
            frequency: Create a new plot every `frequency` batches.
            make_static_plots: Whether to create static plots.
            make_video_plots: Whether to create video plots.
            video_fps: Frames per second for video plots.
            video_format: Format for video plots (mp4 or gif).
            plot_spec: Plotting specification to use (contains difference settings,
                      timestep selection, etc.).
            base_path: Base path for finding land masks.

        """
        super().__init__()
        self.frequency = int(max(1, frequency))
        self.make_static_plots = make_static_plots
        self.make_video_plots = make_video_plots
        self.plotter = Plotter(base_path, plot_spec or DEFAULT_SIC_SPEC)

    def set_metadata(self, config: DictConfig, model_name: str) -> None:
        """Set metadata for the plotter."""
        self.plotter.set_metadata(config, model_name)

    # --- Lightning Hook ---
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,  # noqa: ANN401, ARG002
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
            msg = f"Dataset is of type {type(dataset)}, skipping plotting."
            logger.warning(msg)
            return

        # Get sequence dates for static and video plots
        batch_size = int(outputs.target.shape[0])
        n_timesteps = int(outputs.target.shape[1])
        dates = [
            datetime_from_npdatetime(dataset.dates[batch_size * batch_idx + tt])
            for tt in range(n_timesteps)
        ]

        self.plotter.set_hemisphere(dataset.hemisphere)

        if self.make_static_plots:
            self.plotter.log_static_plots(outputs, dates, trainer.loggers)

        if self.make_video_plots:
            self.plotter.log_video_plots(outputs, dates, trainer.loggers)
