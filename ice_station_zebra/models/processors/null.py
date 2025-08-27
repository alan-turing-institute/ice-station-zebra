from itertools import cycle
from typing import Any

from torch import nn, stack

from ice_station_zebra.types import TensorNTCHW

from .base_processor import BaseProcessor


class NullProcessor(BaseProcessor):
    """Null model that simply returns input.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels, latent_height, latent_width)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise a NullProcessor."""
        super().__init__(**kwargs)
        self.model = nn.Identity()

    def forward(self, x: TensorNTCHW) -> TensorNTCHW:
        """Forward step: process in latent space.

        Args:
            x: TensorNTCHW with (batch_size, n_history_steps, n_latent_channels, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels, latent_height, latent_width)

        """
        return stack(
            [
                self.model(x[:, next(cycle(range(self.n_history_steps))), :, :, :])
                for _ in range(self.n_forecast_steps)
            ],
            dim=1,
        )
