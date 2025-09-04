from abc import ABC, abstractmethod

from torch import nn

from ice_station_zebra.types import TensorNTCHW


class BaseDecoder(nn.Module, ABC):
    """Decoder that takes data in a latent space and translates it to a larger output space.

    Latent space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)
    """

    def __init__(self, *, n_forecast_steps: int, n_latent_channels_total: int) -> None:
        """Initialise a BaseDecoder."""
        super().__init__()
        self.n_forecast_steps = n_forecast_steps
        self.n_latent_channels_total = n_latent_channels_total

    @abstractmethod
    def forward(self, x: TensorNTCHW) -> TensorNTCHW:
        """Forward step: decode latent space into output space.

        Args:
            x: TensorNTCHW with (batch_size, n_forecast_steps, latent_channels, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)

        """
