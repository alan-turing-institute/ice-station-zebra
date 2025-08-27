from abc import ABC, abstractmethod

from torch import nn

from ice_station_zebra.types import TensorNTCHW


class BaseProcessor(nn.Module, ABC):
    """Processor that converts latent input into latent output.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels, latent_height, latent_width)
    """

    def __init__(
        self, *, n_forecast_steps: int, n_history_steps: int, n_latent_channels: int
    ) -> None:
        """Initialise a NullProcessor."""
        super().__init__()
        self.n_forecast_steps = n_forecast_steps
        self.n_history_steps = n_history_steps
        self.n_latent_channels = n_latent_channels

    @abstractmethod
    def forward(self, x: TensorNTCHW) -> TensorNTCHW:
        """Forward step: process in latent space.

        Args:
            x: TensorNTCHW with (batch_size, n_history_steps, n_latent_channels, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels, latent_height, latent_width)

        """
