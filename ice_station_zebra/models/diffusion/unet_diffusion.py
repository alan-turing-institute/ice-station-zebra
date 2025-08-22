"""UNetDiffusion: Conditional U-Net for DDPM-based Forecasting.

Author: Maria Carolina Novitasari

Description:
    U-Net architecture for use in conditional denoising diffusion probabilistic models (DDPM),
    designed for geophysical forecasting tasks such as sea ice concentration prediction.
    Inputs include noisy predictions, diffusion timestep embeddings, and meteorological
    conditioning inputs. Supports configurable number of forecast days and output classes via constructor parameters.

"""

import math

import torch
import torch.nn as nn

from ice_station_zebra.models.common.bottleneckblock import BottleneckBlock
from ice_station_zebra.models.common.convblock import ConvBlock
from ice_station_zebra.models.common.timeembed import TimeEmbed
from ice_station_zebra.models.common.upconvblock import UpconvBlock


class UNetDiffusion(nn.Module):
    """U-Net architecture for conditional DDPM-based forecasting.
    
    Inputs include noisy predictions, time step embeddings, and conditioning inputs.
    Supports configurable depth, filter size, and number of forecast days/classes.
    """
    
    def __init__(
        self,
        input_channels: int,
        timesteps: int = 1000,
        filter_size: int = 3,
        start_out_channels: int = 64,
) -> None:
        """Initialize the U-Net diffusion model.

        Args:
            input_channels (int): Number of input conditioning channels (e.g., meteorological variables).
            timesteps (int): Number of diffusion timesteps.
            filter_size (int): Convolution kernel size for all conv layers.
            start_out_channels (int): Number of output channels in the first convolution block. Defaults to 64.
            
        """
        super().__init__()

        self.filter_size = filter_size
        self.start_out_channels = start_out_channels
        self.timesteps = timesteps

        # Time embedding
        self.time_embed_dim = 256
        self.time_embed = TimeEmbed(self.time_embed_dim)

        # Channel calculations
        channels = [start_out_channels * 2**i for i in range(4)]

        output_channels = input_channels
        self.initial_conv_channels = input_channels + output_channels

        # Encoder
        self.conv1 = ConvBlock(
            self.initial_conv_channels, channels[0], filter_size=filter_size
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = ConvBlock(channels[0], channels[1], filter_size=filter_size)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = ConvBlock(channels[1], channels[2], filter_size=filter_size)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = ConvBlock(channels[2], channels[2], filter_size=filter_size)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.conv5 = BottleneckBlock(channels[2], channels[3], filter_size=filter_size)

        # Decoder
        self.up6 = UpconvBlock(channels[3], channels[2])
        self.up7 = UpconvBlock(channels[2], channels[2])
        self.up8 = UpconvBlock(channels[2], channels[1])
        self.up9 = UpconvBlock(channels[1], channels[0])

        self.up6b = ConvBlock(
            channels[3] + self.time_embed_dim, channels[2], filter_size=filter_size
        )
        self.up7b = ConvBlock(
            channels[3] + self.time_embed_dim, channels[2], filter_size=filter_size
        )
        self.up8b = ConvBlock(
            channels[2] + self.time_embed_dim, channels[1], filter_size=filter_size
        )
        self.up9b = ConvBlock(
            channels[1] + self.time_embed_dim,
            channels[0],
            filter_size=filter_size,
            final=True,
        )

        # Final layer
        self.final_layer = nn.Conv2d(
            channels[0], output_channels, kernel_size=1, padding="same"
        )

    def forward(
        self,
        noise: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the U-Net diffusion model.

        Args:
            noise (torch.Tensor): Noisy forecast tensor of shape [B, H, W, n_classes, n_forecast_days].
            t (torch.Tensor): Diffusion timestep tensor of shape [B].
            conditioning (torch.Tensor): Conditioning input tensor of shape [B, H, W, input_channels].

        Returns:
            torch.Tensor: Predicted denoised forecast of shape [B, H, W, n_classes, n_forecast_days].
        """
        noise = 2.0 * noise - 1.0
        conditioning = 2.0 * conditioning - 1.0

        # Time embedding
        t = self._timestep_embedding(t)
        t = self.time_embed(t)

        # Concatenate with conditional input
        noise = torch.cat([noise, conditioning], dim=-1)  # [b,h,w,(d*c)+input_channels]

        # Convert to channel-first format
        noise = torch.movedim(noise, -1, 1)  # [b,channels,h,w]

        # Encoder pathway
        bn1 = self.conv1(noise)
        conv1 = self.maxpool1(bn1)
        bn2 = self.conv2(conv1)
        conv2 = self.maxpool2(bn2)
        bn3 = self.conv3(conv2)
        conv3 = self.maxpool3(bn3)
        bn4 = self.conv4(conv3)
        conv4 = self.maxpool4(bn4)

        # Bottleneck
        bn5 = self.conv5(conv4)

        # Decoder with time embedding
        up6 = self.up6(bn5)
        up6 = torch.cat([bn4, up6], dim=1)
        up6 = self._add_time_embedding(up6, t)
        up6 = self.up6b(up6)

        up7 = self.up7(up6)
        up7 = torch.cat([bn3, up7], dim=1)
        up7 = self._add_time_embedding(up7, t)
        up7 = self.up7b(up7)

        up8 = self.up8(up7)
        up8 = torch.cat([bn2, up8], dim=1)
        up8 = self._add_time_embedding(up8, t)
        up8 = self.up8b(up8)

        up9 = self.up9(up8)
        up9 = torch.cat([bn1, up9], dim=1)
        up9 = self._add_time_embedding(up9, t)
        up9 = self.up9b(up9)

        # Final output
        output = self.final_layer(up9)  # [b, c_out, h, w]

        return torch.movedim(output, 1, -1)  # [b, h, w, c_out]

    def _timestep_embedding(
        self, timesteps: torch.Tensor, dim: int = 256, max_period: int = 10000
    ) -> torch.Tensor:

        """Converts timestep integers into sinusoidal positional embeddings.

        Args:
            timesteps (torch.Tensor): Timestep tensor [B].
            dim (int): Embedding dimension.
            max_period (int): Frequency range.

        Returns:
            torch.Tensor: Embedding tensor of shape [B, dim].
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=timesteps.device
            )
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
            
        return embedding

    def _add_time_embedding(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        """Concatenates time embedding across spatial dimensions.

        Args:
            x (torch.Tensor): Feature map tensor [B, C, H, W].
            t (torch.Tensor): Time embedding tensor [B, D].

        Returns:
            torch.Tensor: Time-conditioned feature map [B, C+D, H, W].
        """
        b, c, h, w = x.shape
        t = t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        
        return torch.cat([x, t], dim=1)
