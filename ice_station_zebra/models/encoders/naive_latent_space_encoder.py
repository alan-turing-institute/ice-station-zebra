import math
from typing import Any

from torch import nn

from ice_station_zebra.types import DataSpace, TensorNCHW, TensorNTCHW

from .base_encoder import BaseEncoder


class NaiveLatentSpaceEncoder(BaseEncoder):
    """
    Naive, linear encoder that takes data in an input space and translates it to a smaller latent space.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

    Latent space:
        TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)
    """

    def __init__(
        self, *, input_space: DataSpace, latent_space: DataSpace, **kwargs: Any
    ) -> None:
        super().__init__(name=input_space.name, **kwargs)

        # Construct list of layers
        layers: list[nn.Module] = []

        # Start by flattening the time and channels
        layers.append(nn.Flatten(1, 2))
        n_channels = input_space.channels * self.n_history_steps

        # Add size-reducing convolutional layers while we are larger than the latent shape
        n_conv_layers = math.floor(
            math.log2(min(*input_space.shape) / max(*latent_space.shape))
        )
        for _ in range(n_conv_layers):
            layers.append(
                nn.Conv2d(
                    n_channels, 2 * n_channels, kernel_size=4, stride=2, padding=1
                )
            )
            n_channels *= 2

        # Resample to the desired latent shape
        layers.append(nn.Upsample(latent_space.shape))

        # Convolve to the desired number of latent channels
        layers.append(nn.Conv2d(n_channels, latent_space.channels, 1))

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNTCHW) -> TensorNCHW:
        """
        Forward step: encode input space into latent space.

        Args:
            x: TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)
        """
        return self.model(x)
