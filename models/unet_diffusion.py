"""
UNetDiffusion: Conditional U-Net for DDPM-based Forecasting

Author: Maria Carolina Novitasari 

Description:
    U-Net architecture for use in conditional denoising diffusion probabilistic models (DDPM),
    designed for geophysical forecasting tasks such as sea ice concentration prediction.
    Inputs include noisy predictions, diffusion timestep embeddings, and meteorological
    conditioning inputs. Supports configurable number of forecast days and output classes via constructor parameters.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

from .common import CommonConvBlock, TimeEmbed, UpConvBlock

        
class UNetDiffusion(nn.Module):
    """
    U-Net architecture for conditional DDPM-based forecasting.
    Inputs include noisy predictions, time step embeddings, and conditioning inputs.
    Supports configurable depth, filter size, and number of forecast days/classes.
    """
    
    def __init__(self,
                 input_channels,
                 filter_size=3,
                 # n_filters_factor=1,
                 n_forecast_days=7,
                 n_output_classes=1,
                 timesteps=1000,
                 start_out_channels=32,
                 activation: str = "SiLU",
                 normalization: str = "groupnorm",
                 dropout_rate: float = 0.1,
                 **kwargs):
        """
        Initialize the U-Net diffusion model.

        Args:
            input_channels (int): Number of input conditioning channels (e.g., meteorological variables).
            filter_size (int): Convolution kernel size for all conv layers.
            n_filters_factor (float): Scaling factor for channel depth across the network.
            n_forecast_days (int): Number of days to forecast.
            n_output_classes (int): Number of output regression targets per forecast day.
            timesteps (int): Number of diffusion timesteps.
            start_out_channels (int): Number of output channels in the first convolution block. Defaults to 32.
            activation (str): Activation function to use (e.g., SiLU, ReLU, LeakyReLU).
            normalization (str): Normalization strategy (e.g., groupnorm, batchnorm).
            dropout_rate (float): Dropout probability applied in convolutional blocks.
            **kwargs: Additional arguments (ignored).
        """
        super(UNetDiffusion, self).__init__()

        self.input_channels = input_channels
        self.filter_size = filter_size
        # self.n_filters_factor = n_filters_factor
        self.n_forecast_days = n_forecast_days
        self.n_output_classes = n_output_classes
        self.timesteps = timesteps
        self.activation = activation
        self.normalization = normalization
        self.dropout_rate = dropout_rate
        
        # Time embedding
        self.time_embed_dim = 256
        self.time_embed = TimeEmbed(self.time_embed_dim)
      
        channels = [start_out_channels * 2**pow for pow in range(4)]

        self.initial_conv_channels = (n_output_classes * n_forecast_days) + input_channels
        
        # Encoder
        self.conv1 = CommonConvBlock(
            in_channels=self.initial_conv_channels,
            out_channels=channels[0],
            kernel_size=self.filter_size,
            norm_type=self.normalization,
            activation=self.activation,
        )
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = CommonConvBlock(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=self.filter_size,
            norm_type=self.normalization,
            activation=self.activation,
        )
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = CommonConvBlock(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=self.filter_size,
            norm_type=self.normalization,
            activation=self.activation,
        )
        self.maxpool3 = nn.MaxPool2d(2)
        self.conv4 = CommonConvBlock(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=self.filter_size,
            norm_type=self.normalization,
            activation=self.activation,
        )
        self.maxpool4 = nn.MaxPool2d(2)
        

        # Bottleneck
        self.conv5 = CommonConvBlock(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=self.filter_size,  # You'll need to pass this from self.filter_size
            norm_type=self.normalization,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
        )

        # Decoder
        self.up6 = UpConvBlock(
            in_channels=channels[3],
            out_channels=channels[2],
            norm_type=self.normalization,
            activation=self.activation,
        )
        self.up7 = UpConvBlock(
            in_channels=channels[2],
            out_channels=channels[2],
            norm_type=self.normalization,
            activation=self.activation,
        )
        self.up8 = UpConvBlock(
            in_channels=channels[2],
            out_channels=channels[1],
            norm_type=self.normalization,
            activation=self.activation,
        )
        self.up9 = UpConvBlock(
            in_channels=channels[1],
            out_channels=channels[0],
            norm_type=self.normalization,
            activation=self.activation,
        )

        self.up6b = CommonConvBlock(
            in_channels=channels[3] + self.time_embed_dim,
            out_channels=channels[2],
            kernel_size=self.filter_size,
            norm_type=self.normalization,
            activation=self.activation,
        )
        self.up7b = CommonConvBlock(
            in_channels=channels[3] + self.time_embed_dim,
            out_channels=channels[2],
            kernel_size=self.filter_size,
            norm_type=self.normalization,
            activation=self.activation,
        )
        self.up8b = CommonConvBlock(
            in_channels=channels[2] + self.time_embed_dim,
            out_channels=channels[1],
            kernel_size=self.filter_size,
            norm_type=self.normalization,
            activation=self.activation,
        )
        self.up9b = CommonConvBlock(
            in_channels=channels[1] + self.time_embed_dim,
            out_channels=channels[0],
            kernel_size=self.filter_size,
            norm_type=self.normalization,
            activation=self.activation,
            final=True,
        )

        # Final layer
        self.final_layer = nn.Conv2d(
            channels[0], n_output_classes * n_forecast_days, kernel_size=1, padding="same"
        )

    def forward(self, x, t, y, sample_weight):
        """
        Forward pass of the U-Net diffusion model.

        Args:
            x (torch.Tensor): Noisy forecast tensor of shape [B, H, W, n_classes, n_forecast_days].
            t (torch.Tensor): Diffusion timestep tensor of shape [B].
            y (torch.Tensor): Conditioning input tensor of shape [B, H, W, input_channels].
            sample_weight (torch.Tensor or None): Optional weighting mask [B, H, W, n_classes, n_forecast_days].

        Returns:
            torch.Tensor: Predicted denoised forecast of shape [B, H, W, n_classes, n_forecast_days].
        """
        x = 2.0 * x - 1.0    
        y = 2.0 * y - 1.0    
    
        # Time embedding
        t = self._timestep_embedding(t)
        t = self.time_embed(t)
        
        # Concatenate with conditional input
        x = torch.cat([x, y], dim=-1)  # [b,h,w,(d*c)+input_channels]
        
        # Convert to channel-first format
        x = torch.movedim(x, -1, 1)  # [b,channels,h,w]

        # Encoder pathway
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
        output = torch.movedim(output, 1, -1)  # [b, h, w, c_out]

        b, h, w, c = output.shape
        output = output.reshape((b, h, w, self.n_output_classes, self.n_forecast_days))

        return output
        
    def _timestep_embedding(self, timesteps, dim=256, max_period=10000):
        """
        Converts timestep integers into sinusoidal positional embeddings.

        Args:
            timesteps (torch.Tensor): Timestep tensor [B].
            dim (int): Embedding dimension.
            max_period (int): Frequency range.

        Returns:
            torch.Tensor: Embedding tensor of shape [B, dim].
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def _add_time_embedding(self, x, t):
        """
        Concatenates time embedding across spatial dimensions.

        Args:
            x (torch.Tensor): Feature map tensor [B, C, H, W].
            t (torch.Tensor): Time embedding tensor [B, D].

        Returns:
            torch.Tensor: Time-conditioned feature map [B, C+D, H, W].
        """
        b, c, h, w = x.shape
        t = t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        return torch.cat([x, t], dim=1)
    
    def _get_num_groups(self, channels):
        """
        Determines the maximum number of groups that divide `channels` for GroupNorm.

        Args:
            channels (int): Number of feature channels.

        Returns:
            int: Optimal number of groups.
        """
        num_groups = 8  # Start with preferred group count
        while num_groups > 1:
            if channels % num_groups == 0:
                return num_groups
            num_groups -= 1
        return 1  # Fallback to GroupNorm(1,...) which is equivalent to LayerNorm
