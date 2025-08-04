import math

import torch.nn as nn
from torch import Tensor

from ice_station_zebra.types import DataSpace


class NaiveLatentSpaceEncoder(nn.Module):
    """
    Encoder that takes data in an input space and translates it to a smaller latent space

    Input:
        Tensor of (batch_size, input_channels, input_height, input_width)

    Output:
        Tensor of (batch_size, latent_channels, latent_height, latent_width)
    """

    def __init__(self, *, input_space: DataSpace, latent_space: DataSpace) -> None:
        super().__init__()
        self.name = input_space.name

        # Construct list of layers
        layers: list[nn.Module] = []

        # Add size-reducing convolutional layers while we are larger than the latent shape
        n_conv_layers = math.floor(
            math.log2(min(*input_space.shape) / max(*latent_space.shape))
        )
        n_channels = input_space.channels
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Transformation summary

        Input:
            x: Tensor of (batch_size, input_channels, input_height, input_width)

        Output:
            Tensor of (batch_size, latent_channels, latent_height, latent_width)
        """
        return self.model(x)
