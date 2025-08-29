import math
from typing import Any

from torch import nn, stack

from ice_station_zebra.types import DataSpace, TensorNTCHW

from .base_decoder import BaseDecoder


class NaiveLatentSpaceDecoder(BaseDecoder):
    """Naive, linear decoder that takes data in a latent space and translates it to a larger output space.

    Latent space:
        TensorNTCHW with (batch_size, n_forecast_steps, latent_channels, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)
    """

    def __init__(
        self, *, latent_space: DataSpace, output_space: DataSpace, **kwargs: Any
    ) -> None:
        """Initialise a NaiveLatentSpaceDecoder."""
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
        layers.append(nn.Conv2d(n_channels, output_space.channels, 1))

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNTCHW) -> TensorNTCHW:
        """Forward step: decode latent space into output space.

        Args:
            x: TensorNTCHW with (batch_size, n_forecast_steps, latent_channels, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)

        """
        return stack(
            [
                self.model(x[:, idx_t, :, :, :])  # cut the NTCHW input into NCHW slices
                for idx_t in range(self.n_forecast_steps)
            ],
            dim=1,
        )
