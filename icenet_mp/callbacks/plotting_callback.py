import logging
from collections.abc import Mapping, Sequence
from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from icenet_mp.data_loaders import CombinedDataset
from icenet_mp.models import BaseModel
from icenet_mp.types import ModelStepOutput, PlotSpec
from icenet_mp.utils import datetime_from_npdatetime
from icenet_mp.visualisations import DEFAULT_SIC_SPEC, Plotter

logger = logging.getLogger(__name__)


class PlottingCallback(Callback):
    """A callback to create plots during evaluation."""

    def __init__(
        self,
        *,
        frequency: int = 5,
        make_input_plots: bool = False,
        make_static_plots: bool = True,
        make_video_plots: bool = True,
        plot_spec: PlotSpec | None = None,
        base_path: str | None = None,
    ) -> None:
        """Create plots during evaluation.

        Args:
            frequency: Create a new plot every `frequency` batches.
            make_input_plots: Whether to plot the raw inputs.
            make_static_plots: Whether to create static plots.
            make_video_plots: Whether to create video plots.
            plot_spec: Plotting specification to use (contains difference settings, timestep selection, etc.).
            base_path: Base path for finding land masks.

        """
        super().__init__()
        self.frequency = int(max(1, frequency))
        self.make_input_plots = make_input_plots
        self.make_static_plots = make_static_plots
        self.make_video_plots = make_video_plots
        self.plotter = Plotter(base_path, plot_spec or DEFAULT_SIC_SPEC)

    def load_dataset(
        self, dataloader: DataLoader | list[DataLoader] | None, dataloader_idx: int
    ) -> CombinedDataset | None:
        """Load the dataset for the given dataloader index."""
        if dataloader is None:
            return None
        dataset = (
            dataloader[dataloader_idx]
            if isinstance(dataloader, Sequence)
            else dataloader
        ).dataset
        if not isinstance(dataset, CombinedDataset):
            logger.warning("Dataset is of type %s not CombinedDataset", type(dataset))
            return None
        return dataset

    def make_plots(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *,
        batch_idx: int,
        dataset: CombinedDataset,
        outputs: Tensor | Mapping[str, Any] | None,
    ) -> None:
        # Ensure that outputs is a ModelStepOutput
        if isinstance(outputs, Mapping):
            outputs = ModelStepOutput(**outputs)
        else:
            logger.warning("Could not load outputs, skipping plotting.")
            return

        # Load dates from the dataset
        batch_size = int(outputs.target.shape[0])
        start_date = dataset.dates[batch_size * batch_idx]
        dates = list(
            map(datetime_from_npdatetime, dataset.get_forecast_steps(start_date))
        )

        # Set hemisphere for plotting based on dataset
        if not isinstance(pl_module, BaseModel):
            msg = f"Lightning module is of type {type(pl_module)}, skipping plotting."
            logger.warning(msg)
            return
        self.plotter.set_hemisphere(pl_module.hemisphere)

        # Get loggers that support image and video logging
        image_loggers = [ll for ll in trainer.loggers if hasattr(ll, "log_image")]
        video_loggers = [ll for ll in trainer.loggers if hasattr(ll, "log_video")]

        if self.make_static_plots:
            self.plotter.log_static_outputs(outputs, dates, image_loggers)
            if self.make_input_plots:
                self.plotter.log_static_inputs(dataset.inputs, dates, image_loggers)

        if self.make_video_plots:
            self.plotter.log_video_outputs(outputs, dates, video_loggers)
            if self.make_input_plots:
                self.plotter.log_video_inputs(dataset.inputs, dates, video_loggers)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,  # noqa: ANN401, ARG002
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called at the end of each test batch."""
        # Only run plotting every `frequency` batches
        if batch_idx % self.frequency:
            return

        # Load the dataset
        if not (dataset := self.load_dataset(trainer.test_dataloaders, dataloader_idx)):
            logger.warning("Could not load dataset, skipping plotting.")
            return

        # Make the plots
        self.make_plots(
            trainer,
            pl_module,
            batch_idx=batch_idx,
            dataset=dataset,
            outputs=outputs,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,  # noqa: ANN401, ARG002
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called at the end of each train batch."""
        # Only run plotting every `frequency` batches
        if batch_idx % self.frequency:
            return

        # Load the dataset
        if not (dataset := self.load_dataset(trainer.train_dataloader, dataloader_idx)):
            logger.warning("Could not load dataset, skipping plotting.")
            return

        # Make the plots
        self.make_plots(
            trainer,
            pl_module,
            batch_idx=batch_idx,
            dataset=dataset,
            outputs=outputs,
        )

    def set_metadata(self, config: DictConfig, model_name: str) -> None:
        """Set metadata for the plotter."""
        self.plotter.set_metadata(config, model_name)
