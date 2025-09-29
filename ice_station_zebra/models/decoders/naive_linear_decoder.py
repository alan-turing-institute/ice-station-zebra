from typing import Any

from torch import nn

from ice_station_zebra.types import DataSpace, TensorNCHW

from .base_decoder import BaseDecoder


class NaiveLinearDecoder(BaseDecoder):
    """Naive, linear decoder that takes data in a latent space and translates it to a larger output space.

    Latent space:
        TensorNTCHW with (batch_size, n_forecast_steps, latent_channels, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)
    """

    def __init__(
        self,
        *,
        output_space: DataSpace,
        **kwargs: Any,
    ) -> None:
        """Initialise a NaiveLinearDecoder."""
        super().__init__(**kwargs)

        # List of layers
        layers: list[nn.Module] = []

        # Convolve to the desired number of output channels
        layers.append(nn.Conv2d(self.n_latent_channels_total, output_space.channels, 1))

        # Resample to the desired output shape
        layers.append(nn.Upsample(output_space.shape))

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def rollout(self, x: TensorNCHW) -> TensorNCHW:
        """Single rollout step: decode NCHW latent data into NCHW output.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, output_channels, output_height, output_width)

        """
        return self.model(x)
