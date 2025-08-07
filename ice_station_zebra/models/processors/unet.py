import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F

from ice_station_zebra.models.common.convblock import ConvBlock
from ice_station_zebra.models.common.bottleneckblock import BottleneckBlock
from ice_station_zebra.models.common.upconvblock import UpconvBlock


class UNetProcessor(nn.Module):
    """UNet model that processes input through a UNet architecture"""

    def __init__(
        self,
        n_latent_channels: int,
        filter_size: int,
        n_filters_factor: float,
    ) -> None:
        super().__init__()

        start_out_channels = 64

        reduced_channels = int(start_out_channels * n_filters_factor)

        channels = [reduced_channels * 2**pow for pow in range(4)]

        # Encoder
        self.conv1 = ConvBlock(n_latent_channels, channels[0], filter_size=filter_size)
        self.conv2 = ConvBlock(channels[0], channels[1], filter_size=filter_size)
        self.conv3 = ConvBlock(channels[1], channels[2], filter_size=filter_size)
        self.conv4 = ConvBlock(channels[2], channels[2], filter_size=filter_size)

        # Bottleneck
        self.conv5 = BottleneckBlock(channels[2], channels[3], filter_size=filter_size)

        # Decoder
        self.up6 = UpconvBlock(channels[3], channels[2])
        self.up7 = UpconvBlock(channels[2], channels[2])
        self.up8 = UpconvBlock(channels[2], channels[1])
        self.up9 = UpconvBlock(channels[1], channels[0])

        self.up6b = ConvBlock(channels[3], channels[2], filter_size=filter_size)
        self.up7b = ConvBlock(channels[3], channels[2], filter_size=filter_size)
        self.up8b = ConvBlock(channels[2], channels[1], filter_size=filter_size)
        self.up9b = ConvBlock(
            channels[1], channels[0], filter_size=filter_size, final=True
        )

        # Final layer
        self.final_layer = nn.Conv2d(
            channels[0], n_latent_channels, kernel_size=1, padding="same"
        )

    def forward(self, x: Tensor) -> Tensor:
        # Process in latent space: tensor with (batch_size, all_variables, latent_height, latent_width)

        # Encoder
        bn1 = self.conv1(x)
        conv1 = F.max_pool2d(bn1, kernel_size=2)
        bn2 = self.conv2(conv1)
        conv2 = F.max_pool2d(bn2, kernel_size=2)
        bn3 = self.conv3(conv2)
        conv3 = F.max_pool2d(bn3, kernel_size=2)
        bn4 = self.conv4(conv3)
        conv4 = F.max_pool2d(bn4, kernel_size=2)

        # Bottleneck
        bn5 = self.conv5(conv4)

        # Decoder
        up6 = self.up6b(torch.cat([bn4, self.up6(bn5)], dim=1))
        up7 = self.up7b(torch.cat([bn3, self.up7(up6)], dim=1))
        up8 = self.up8b(torch.cat([bn2, self.up8(up7)], dim=1))
        up9 = self.up9b(torch.cat([bn1, self.up9(up8)], dim=1))

        # Final layer
        output = self.final_layer(up9)

        return output
