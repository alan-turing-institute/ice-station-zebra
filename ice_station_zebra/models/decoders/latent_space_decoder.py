import math

import torch.nn as nn
from torch import Tensor

from ice_station_zebra.types import ZebraDataSpace


class LatentSpaceDecoder(nn.Module):
    """
    Decoder that takes data in a latent space and translates it to a larger output space

    Input:
        Tensor of (batch_size, latent_channels, latent_shape_x, latent_shape_y)

    Output:
        Tensor of (batch_size, output_channels, output_shape_x, output_shape_y)
    """

    def __init__(
        self,
        *,
        latent_space: ZebraDataSpace,
        output_space: ZebraDataSpace,
    ) -> None:
        super().__init__()

        # List of layers
        layers = []

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

        # Upsample to the desired output shape
        layers.append(nn.Upsample(output_space.shape))

        # Convolve to the desired number of output channels
        layers.append(nn.Conv2d(n_channels, output_space.channels, 1))

        # Combine the layers into a list
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Transformation summary

        Inputs:
            x: Tensor(batch_size, channels, input_shape[0], input_shape[1])

        Outputs:
            Tensor(batch_size, output_channels, 2^latent_size, 2^latent_size)
        """
        return self.model(x)
