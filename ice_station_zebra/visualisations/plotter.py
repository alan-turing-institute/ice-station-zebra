import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from omegaconf import DictConfig

from ice_station_zebra.data_loaders import ZebraDataset
from ice_station_zebra.exceptions import InvalidArrayError, VideoRenderError
from ice_station_zebra.types import ModelTestOutput, PlotSpec

from .land_mask import LandMask
from .metadata import build_metadata, format_metadata_subtitle
from .plotting_static import plot_static_inputs, plot_static_prediction
from .plotting_video import plot_video_inputs, plot_video_prediction

logger = logging.getLogger(__name__)


class Plotter:
    def __init__(self, base_path: str | None, plot_spec: PlotSpec) -> None:
        """A helper class to create and log plots."""
        self.base_path = Path(base_path) if base_path else None
        self.plot_spec = plot_spec

    def set_hemisphere(self, hemisphere: Literal["north", "south"]) -> None:
        """Set the hemisphere and update the plot spec accordingly."""
        self.plot_spec.hemisphere = hemisphere
        self.land_mask = LandMask(self.base_path, hemisphere)

    def set_metadata(self, config: DictConfig, model_name: str) -> None:
        """Set metadata for the plotter based on the model test output."""
        metadata = build_metadata(config, model_name)
        self.plot_spec.metadata_subtitle = format_metadata_subtitle(metadata)

    def log_static_inputs(
        self, inputs: list[ZebraDataset], dates: list[datetime], image_loggers: list
    ) -> None:
        """Extract and log raw input plots."""
        try:
            date = dates[self.plot_spec.selected_timestep]
            for input_ds in inputs:
                # Get static data for this timestep
                channels = {
                    variable_name: input_ds[self.plot_spec.selected_timestep][
                        channel, :
                    ]
                    for channel, variable_name in enumerate(input_ds.variable_names)
                }
                # Plot and log input static images
                for key, image_list in plot_static_inputs(
                    channels=channels,
                    land_mask=self.land_mask,
                    plot_spec_base=self.plot_spec,
                    save_dir=Path(
                        "/Users/jrobinson/Developer/forecasting/sea-ice/ice-station-zebra/outputs/"
                    ),
                    when=date,
                ).items():
                    for image_logger in image_loggers:
                        image_logger.log_image(key=key, images=image_list)
        except InvalidArrayError as exc:
            logger.warning("Static plotting skipped due to invalid arrays: %s", exc)
        except (IndexError, ValueError, MemoryError, OSError) as exc:
            logger.warning("Static plotting failed: %s", exc)

    def log_static_outputs(
        self, outputs: ModelTestOutput, dates: list[datetime], image_loggers: list
    ) -> None:
        """Create and log static image plots."""
        try:
            date = dates[self.plot_spec.selected_timestep]
            # Use the first batch, first channel -> [H,W]
            np_ground_truth_hw = (
                outputs.target[0, self.plot_spec.selected_timestep, 0]
                .detach()
                .cpu()
                .numpy()
            )
            np_prediction_hw = (
                outputs.prediction[0, self.plot_spec.selected_timestep, 0]
                .detach()
                .cpu()
                .numpy()
            )
            # Plot and log output static images
            for key, image_list in plot_static_prediction(
                date=date,
                ground_truth_hw=np_ground_truth_hw,
                land_mask=self.land_mask,
                plot_spec=self.plot_spec,
                prediction_hw=np_prediction_hw,
            ).items():
                for image_logger in image_loggers:
                    image_logger.log_image(key=key, images=image_list)
        except InvalidArrayError as err:
            logger.warning("Static plotting skipped due to invalid arrays: %s", err)
        except (IndexError, ValueError, MemoryError, OSError) as exc:
            logger.warning("Static plotting failed: %s", exc)

    def log_video_inputs(
        self, inputs: list[ZebraDataset], dates: list[datetime], video_loggers: list
    ) -> None:
        """Extract and log raw input plots."""
        for input_ds in inputs:
            # Create animations for all variables
            np_dates = [np.datetime64(date.replace(tzinfo=None)) for date in dates]
            channels = {
                variable_name: input_ds.get_tchw(np_dates)[:, channel, :]
                for channel, variable_name in enumerate(input_ds.variable_names)
            }
            videos = plot_video_inputs(
                channels=channels,
                dates=dates,
                plot_spec=self.plot_spec,
                land_mask=self.land_mask,
                save_dir=Path(
                    "/Users/jrobinson/Developer/forecasting/sea-ice/ice-station-zebra/outputs/videos"
                ),
            )

            # Log input videos
            for video_logger in video_loggers:
                for key, video_buffer in videos.items():
                    video_buffer.seek(0)
                    video_logger.log_video(
                        key=key,
                        videos=[video_buffer],
                        format=[self.plot_spec.video_format],
                    )

    def log_video_outputs(
        self, outputs: ModelTestOutput, dates: list[datetime], video_loggers: list
    ) -> None:
        """Create and log video plots."""
        try:
            # Use the first batch, first channel -> [T,H,W]
            np_ground_truth_thw = outputs.target[0, :, 0].detach().cpu().numpy()
            np_prediction_thw = outputs.prediction[0, :, 0].detach().cpu().numpy()
            video_data = plot_video_prediction(
                dates=dates,
                fps=self.plot_spec.video_fps,
                ground_truth_stream=np_ground_truth_thw,
                land_mask=self.land_mask,
                plot_spec=self.plot_spec,
                prediction_stream=np_prediction_thw,
                video_format=self.plot_spec.video_format,
            )
            for video_logger in video_loggers:
                for key, video_buffer in video_data.items():
                    video_buffer.seek(0)
                    video_logger.log_video(
                        key=key,
                        videos=[video_buffer],
                        format=[self.plot_spec.video_format],
                    )
        except (InvalidArrayError, VideoRenderError) as err:
            logger.warning("Video plotting skipped: %s", err)
        except (IndexError, ValueError, MemoryError, OSError):
            logger.exception("Video plotting failed")
