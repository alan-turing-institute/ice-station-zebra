from abc import ABC, abstractmethod

from torch import nn

from ice_station_zebra.types import TensorNCHW, TensorNTCHW


class BaseDecoder(nn.Module, ABC):
    """
    Decoder that takes data in a latent space and translates it to a larger output space.

    Latent space:
        TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)
    """

    def __init__(self, *, n_forecast_steps: int) -> None:
        super().__init__()
        self.n_forecast_steps = n_forecast_steps

    @abstractmethod
    def forward(self, x: TensorNCHW) -> TensorNTCHW:
        """
        Forward step: decode latent space into output space.

        Args:
            x: TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)
        """
