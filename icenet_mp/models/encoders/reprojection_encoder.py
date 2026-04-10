import logging
import time
from functools import cache
from itertools import product
from typing import Any

import numpy as np
import torch
from haversine import haversine_vector

from icenet_mp.types import TensorNCHW

from .base_encoder import BaseEncoder

logger = logging.getLogger(__name__)


class ReprojectionEncoder(BaseEncoder):
    """Encoder that reprojects data from a source projection to a target projection.

    Each cell in the target projection takes its value from the nearest neighbour in the
    source.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, latent_height, latent_width)
    """

    def __init__(self, project_to: str, **kwargs: Any) -> None:
        """Initialise a ReprojectionEncoder."""
        super().__init__(**kwargs)

        # Name of the output data space to project to
        self.project_to = project_to

        # In order to avoid input/output latitudes and longitudes being recorded as
        # model parameters, we set them later on
        self.input_latitudes: list[float] = []
        self.input_longitudes: list[float] = []
        self.output_latitudes: list[float] = []
        self.output_longitudes: list[float] = []

        # Add a cached method for calculating nearest neighbours
        # Adding this as a decorator would lead to memory leaks
        self.cached_nearest_neighbours = cache(self.nearest_neighbours)

    def nearest_neighbours(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the nearest neighbour input cell for each cell in the output grid.

        Returns:
            Tuple of (nn_indices_h, nn_indices_w) where each is a tensor of shape
            [output_height, output_width] containing, for each output cell, the index in
            the H and W dimensions of the nearest neighbour input cell that should be
            used as the source.

        """
        # Validate that provided latitudes and longitudes are consistent with expected sizes from data spaces
        if (
            not self.input_latitudes
            or not self.input_longitudes
            or not self.output_latitudes
            or not self.output_longitudes
        ):
            msg = "Input/output latitudes and longitudes must be set before calculating nearest neighbours."
            raise ValueError(msg)
        if len(self.input_latitudes) != self.data_space_in.area:
            msg = f"Number of input latitudes {len(self.input_latitudes)} does not match expected size from data space {self.data_space_in.shape}"
            raise ValueError(msg)
        if len(self.input_longitudes) != self.data_space_in.area:
            msg = f"Number of input longitudes {len(self.input_longitudes)} does not match expected size from data space {self.data_space_in.shape}"
            raise ValueError(msg)
        if len(self.output_latitudes) != self.data_space_out.area:
            msg = f"Number of output latitudes {len(self.output_latitudes)} does not match expected size from data space {self.data_space_out.shape}"
            raise ValueError(msg)
        if len(self.output_longitudes) != self.data_space_out.area:
            msg = f"Number of output longitudes {len(self.output_longitudes)} does not match expected size from data space {self.data_space_out.shape}"
            raise ValueError(msg)

        # Get all indices for each cell in the input grid
        input_indices = list(
            product(
                range(self.data_space_in.shape[0]), range(self.data_space_in.shape[1])
            )
        )

        # Get lat/lon values for each cell in the input and output grids
        input_latlons = np.array(
            list(zip(self.input_latitudes, self.input_longitudes, strict=False)),
            dtype=np.float32,
        )
        output_latlons = np.array(
            list(zip(self.output_latitudes, self.output_longitudes, strict=False)),
            dtype=np.float32,
        )

        # We record the time taken in reprojection as this can be slow
        start = time.perf_counter()
        logger.warning(
            "Calculating reprojection from %d input grid points to %d output grid points...",
            len(input_latlons),
            len(output_latlons),
        )

        # We want to find the closest input grid point for each output grid point. If we
        # try to fully vectorise the call, generating a single array of shape
        # [data_space_out.area, data_space_in.area], then we will run out of memory.
        # Instead we loop over the output points, using argmin to reduce to the closest
        # source point for each output point. We then look up the source grid indices
        # for that source point and store each of the height and width indices in an
        # array of [data_space_out_h, data_space_out_w]. This allows easy application of
        # the index lookup during the forward pass.
        closest_src_point_indices = np.array(
            [
                input_indices[
                    np.argmin(haversine_vector(input_latlons, output_latlon, comb=True))
                ]
                for output_latlon in output_latlons
            ]
        )  # [output_h * output_w, 2]
        nn_indices_h = torch.tensor(
            closest_src_point_indices[:, 0].reshape(self.data_space_out.shape),
            dtype=torch.int,
            device=device,
        )  # [output_h, output_w]
        nn_indices_w = torch.tensor(
            closest_src_point_indices[:, 1].reshape(self.data_space_out.shape),
            dtype=torch.int,
            device=device,
        )  # [output_h, output_w]
        logger.info(
            "Reprojection calculation took %.2f seconds", time.perf_counter() - start
        )

        return (nn_indices_h, nn_indices_w)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: encode input space into latent space by splitting into patches.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return x[:, :, *self.cached_nearest_neighbours(x.device)]

    def set_latlon(
        self, name: str, latitudes: list[float], longitudes: list[float]
    ) -> None:
        if name == self.name:
            self.input_latitudes = latitudes
            self.input_longitudes = longitudes
        if name == self.project_to:
            self.output_latitudes = latitudes
            self.output_longitudes = longitudes
