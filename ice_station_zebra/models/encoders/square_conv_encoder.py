import math

import torch.nn as nn
from torch import Tensor

from .base_encoder import BaseEncoder


class SquareConvEncoder(BaseEncoder):
    """
    Encoder that takes 2D input, upscales and applies multiple 2D convoluations

    Input:
        Tensor of (batch_size, channels, input_shape[0], input_shape[1])

    Output:
        Tensor of (batch_size, output_channels, 2^latent_size, 2^latent_size)
    """

    def __init__(
        self, input_shape: tuple[int, int], input_channels: int, *, latent_scale: int
    ) -> None:
        super().__init__()

        # Upsampling scale is the smallest power of 2 larger than the input shape
        upsample_scale = math.ceil(math.log2(max(*input_shape)))
        upsample_length = int(math.pow(2, upsample_scale))

        # Construct list of layers
        layers = []

        # Upsample to a square of length upsample_length
        layers.append(nn.Upsample((upsample_length, upsample_length)))

        # Now add Conv2D layers until we are at 2^latent_scale x 2^latent_scale
        n_channels = input_channels
        for _ in range(upsample_scale - latent_scale):
            layers.append(
                nn.Conv2d(
                    n_channels, 2 * n_channels, kernel_size=3, stride=2, padding=1
                )
            )
            n_channels *= 2

        # Combine the layers into a list
        self.model = nn.Sequential(*layers)

        # Keep track of the number of latent channels for use by the decoder
        self.latent_channels = n_channels

    def forward(self, x: Tensor) -> Tensor:
        """
        Transformation summary

        Inputs:
            x: Tensor(batch_size, channels, input_shape[0], input_shape[1])

        Outputs:
            Tensor(batch_size, output_channels, 2^latent_size, 2^latent_size)
        """
        return self.model(x)
