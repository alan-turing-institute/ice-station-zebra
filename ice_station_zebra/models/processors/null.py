from typing import Any

from torch import nn

from ice_station_zebra.types import TensorNCHW, TensorNTCHW

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

        Uses the default timestep-by-timestep rollout to iterate over NCHW input.

        Args:
            x: TensorNTCHW with (batch_size, n_history_steps, n_latent_channels, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels, latent_height, latent_width)

        """
        return self.rollout(x)

    def rollout_step(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
