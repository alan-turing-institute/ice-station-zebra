import math
from typing import Any

from torch import nn

from ice_station_zebra.types import DataSpace, TensorNCHW, TensorNTCHW

from .base_decoder import BaseDecoder


class NaiveLatentSpaceDecoder(BaseDecoder):
    """Naive, linear decoder that takes data in a latent space and translates it to a larger output space.

    Latent space:
        TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)
    """

    def __init__(
        self, *, latent_space: DataSpace, output_space: DataSpace, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # List of layers
        layers: list[nn.Module] = []

        # Add size-increasing convolutional layers until we are larger than the output shape
        n_conv_layers = math.floor(
            math.log2(min(*output_space.shape) / max(*latent_space.shape))
        )
        n_channels = latent_space.channels
        for _ in range(n_conv_layers):
            layers.append(
                nn.ConvTranspose2d(
                    n_channels, n_channels // 2, kernel_size=4, stride=2, padding=1
                )
            )
            n_channels //= 2

        # Resample to the desired output shape
        layers.append(nn.Upsample(output_space.shape))

        # Convolve to the desired number of output channels
        layers.append(
            nn.Conv2d(n_channels, output_space.channels * self.n_forecast_steps, 1)
        )

        # Unflatten the time and channels
        layers.append(nn.Unflatten(1, [self.n_forecast_steps, output_space.channels]))

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNTCHW:
        """Forward step: decode latent space into output space.

        Args:
            x: TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)
        """
        return self.model(x)
