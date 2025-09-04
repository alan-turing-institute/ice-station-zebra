from typing import Any

from torch import nn

from ice_station_zebra.types import TensorNTCHW

from .base_processor import BaseProcessor


class NullProcessor(BaseProcessor):
    """Null model that simply returns input.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise a NullProcessor.

        Args:
            kwargs: Arguments to BaseProcessor.

        """
        super().__init__(**kwargs)
        self.model = nn.Identity()

    def rollout(self, x: TensorNTCHW) -> TensorNTCHW:
        """Apply identity to NCHW tensor.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        """
        return self.model(x)
