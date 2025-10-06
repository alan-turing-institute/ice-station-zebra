from typing import Any

import torch
from torch import nn

from ice_station_zebra.models.common import CommonConvBlock, ConvBlockUpsampleNaive
from ice_station_zebra.types import TensorNCHW

from .base_processor import BaseProcessor


class UNetProcessor(BaseProcessor):
    """UNet model that processes input through a UNet architecture.

    Structure based on Andersson et al. (2021) Nature Communications
    https://doi.org/10.1038/s41467-021-25257-4

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        kernel_size: int,
        start_out_channels: int,
        **kwargs: Any,
    ) -> None:
        """Initialise a UNetProcessor.

        Args:
            kernel_size: Size of the convolutional filters.
            start_out_channels: Number of output channels in the first layer.
            kwargs: Arguments to BaseProcessor.

        """
        super().__init__(**kwargs)

        if kernel_size <= 0:
            msg = "Kernel size must be greater than 0."
            raise ValueError(msg)

        if start_out_channels <= 0:
            msg = "Start out channels must be greater than 0."
            raise ValueError(msg)

        channels = [start_out_channels * 2**exponent for exponent in range(4)]

        # Encoder
        self.conv1 = CommonConvBlock(
            self.data_space.channels, channels[0], kernel_size=kernel_size
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = CommonConvBlock(channels[0], channels[1], kernel_size=kernel_size)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = CommonConvBlock(channels[1], channels[2], kernel_size=kernel_size)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = CommonConvBlock(channels[2], channels[2], kernel_size=kernel_size)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.conv5 = CommonConvBlock(channels[2], channels[3], kernel_size=kernel_size)

        # Decoder
        self.up6 = ConvBlockUpsampleNaive(channels[3], channels[2])
        self.up7 = ConvBlockUpsampleNaive(channels[2], channels[2])
        self.up8 = ConvBlockUpsampleNaive(channels[2], channels[1])
        self.up9 = ConvBlockUpsampleNaive(channels[1], channels[0])

        self.up6b = CommonConvBlock(channels[3], channels[2], kernel_size=kernel_size)
        self.up7b = CommonConvBlock(channels[3], channels[2], kernel_size=kernel_size)
        self.up8b = CommonConvBlock(channels[2], channels[1], kernel_size=kernel_size)
        self.up9b = CommonConvBlock(
            channels[1], channels[0], kernel_size=kernel_size, n_subblocks=3
        )

        # Final layer
        self.final_layer = nn.Conv2d(
            channels[0], self.data_space.channels, kernel_size=1, padding="same"
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: apply UNet model to NCHW tensor for a single timestep.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        """
        _, _, h, w = x.shape
        if h % 16 or h <= 16 or w % 16 or w <= 16:  # noqa: PLR2004
            msg = f"Latent space height and width must be divisible by 16 with a factor more than 1, got {h} and {w}."
            raise ValueError(msg)

        # Encoder
        bn1 = self.conv1(x)
        conv1 = self.maxpool1(bn1)
        bn2 = self.conv2(conv1)
        conv2 = self.maxpool2(bn2)
        bn3 = self.conv3(conv2)
        conv3 = self.maxpool3(bn3)
        bn4 = self.conv4(conv3)
        conv4 = self.maxpool4(bn4)

        # Bottleneck
        bn5 = self.conv5(conv4)

        # Decoder
        up6 = self.up6b(torch.cat([bn4, self.up6(bn5)], dim=1))
        up7 = self.up7b(torch.cat([bn3, self.up7(up6)], dim=1))
        up8 = self.up8b(torch.cat([bn2, self.up8(up7)], dim=1))
        up9 = self.up9b(torch.cat([bn1, self.up9(up8)], dim=1))

        # Apply final layer and return
        return self.final_layer(up9)
