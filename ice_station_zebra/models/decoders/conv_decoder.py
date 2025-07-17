import math

import torch.nn as nn
from torch import Tensor

from .base_decoder import BaseDecoder


class ConvDecoder(BaseDecoder):
    """
    Encoder that takes 2D input, upscales and applies multiple 2D convoluations

    Input:
        Tensor of (batch_size, channels, input_shape[0], input_shape[1])

    Output:
        Tensor of (batch_size, output_channels, 2^latent_size, 2^latent_size)
    """

    def __init__(
        self,
        output_shape: tuple[int, int],
        output_channels: int,
        *,
        latent_channels: int,
        latent_scale: int,
    ) -> None:
        super().__init__(output_shape, output_channels)

        # List of layers
        layers = []

        # Downsampling scale is the smallest power of 2 larger than the output shape
        downsample_scale = math.ceil(math.log2(max(*self.output_shape)))

        # Apply Conv2D layers until we reach the downsample size
        n_channels = latent_channels
        for _ in range(downsample_scale - latent_scale):
            layers.append(
                nn.ConvTranspose2d(
                    n_channels, n_channels // 2, kernel_size=4, stride=2, padding=1
                )
            )
            n_channels //= 2

        # Downsample to the desired output shape
        layers.append(nn.Upsample(self.output_shape))

        # Convolve back to the desired number of output channels
        layers.append(nn.Conv2d(n_channels, self.output_channels, 1))

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
