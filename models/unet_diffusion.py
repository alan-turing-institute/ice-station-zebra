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

class Interpolate(nn.Module):
    """
    Interpolation module used in the U-Net decoder.

    Description:
        Lightweight wrapper around `torch.nn.functional.interpolate` for use in `nn.Sequential`.
        Enables spatial upsampling during the decoder stages of the U-Net.
        Commonly used to avoid artifacts from transposed convolutions.

    Args:
        scale_factor (float or tuple): Multiplier for spatial resolution (e.g., 2 for 2x upsampling).
        mode (str): Interpolation mode (e.g., 'nearest', 'bilinear').

    Example:
        upsample = Interpolate(scale_factor=2, mode='nearest')
        output = upsample(input_tensor)
    """
    
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
        
class UNetDiffusion(nn.Module):
    """
    U-Net architecture for conditional DDPM-based forecasting.
    Inputs include noisy predictions, time step embeddings, and conditioning inputs.
    Supports configurable depth, filter size, and number of forecast days/classes.
    """
    
    def __init__(self,
                 input_channels,
                 filter_size=3,
                 n_filters_factor=1,
                 n_forecast_days=7,
                 n_output_classes=1,
                 timesteps=1000,
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
            **kwargs: Additional arguments (ignored).
        """
        super(UNetDiffusion, self).__init__()

        self.input_channels = input_channels
        self.filter_size = filter_size
        self.n_filters_factor = n_filters_factor
        self.n_forecast_days = n_forecast_days
        self.n_output_classes = n_output_classes
        self.timesteps = timesteps
        
        # Time embedding
        self.time_embed_dim = 256
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim * 4, self.time_embed_dim),
        )
        
        # Channel calculations
        start_out_channels = 64
        reduced_channels = self._make_divisible(int(start_out_channels * n_filters_factor), 8)
        channels = {
            start_out_channels * 2**pow: self._make_divisible(reduced_channels * 2**pow, 8)
            for pow in range(4)
        }

        self.initial_conv_channels = (n_output_classes * n_forecast_days) + input_channels
        
        # Encoder
        self.conv1 = self.conv_block(self.initial_conv_channels, channels[64])
        self.conv2 = self.conv_block(channels[64], channels[128])
        self.conv3 = self.conv_block(channels[128], channels[256])
        self.conv4 = self.conv_block(channels[256], channels[256])

        # Bottleneck
        self.conv5 = self.bottleneck_block(channels[256], channels[512])

        # Decoder
        self.up6 = self.upconv_block(channels[512], channels[256])
        self.up7 = self.upconv_block(channels[256], channels[256])
        self.up8 = self.upconv_block(channels[256], channels[128])
        self.up9 = self.upconv_block(channels[128], channels[64])

        self.up6b = self.conv_block(channels[512] + self.time_embed_dim, channels[256])
        self.up7b = self.conv_block(channels[512] + self.time_embed_dim, channels[256])
        self.up8b = self.conv_block(channels[256] + self.time_embed_dim, channels[128])
        self.up9b = self.conv_block(channels[128] + self.time_embed_dim, channels[64], final=True)

        # Final layer
        self.final_layer = nn.Conv2d(channels[64], n_output_classes * n_forecast_days, kernel_size=1, padding="same")

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
        conv1 = F.max_pool2d(bn1, kernel_size=2)
        bn2 = self.conv2(conv1)
        conv2 = F.max_pool2d(bn2, kernel_size=2)
        bn3 = self.conv3(conv2)
        conv3 = F.max_pool2d(bn3, kernel_size=2)
        bn4 = self.conv4(conv3)
        conv4 = F.max_pool2d(bn4, kernel_size=2)

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
        
    def _make_divisible(self, v, divisor):
        """
        Ensures a value is divisible by a specified divisor.

        Args:
            v (int): Value to adjust.
            divisor (int): Value to divide by.

        Returns:
            int: Adjusted value divisible by divisor.
        """
        return max(divisor, (v // divisor) * divisor)

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
    
    def conv_block(self, in_channels, out_channels, final=False):
        """
        Standard convolutional block with GroupNorm and SiLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            final (bool): Whether to add an extra conv layer at the end.

        Returns:
            nn.Sequential: Conv block.
        """
        num_groups = self._get_num_groups(out_channels)
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=self.filter_size, padding="same"),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=self.filter_size, padding="same"),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
        ]
        if not final:
            return nn.Sequential(*layers)
        else:
            final_layers = [
                nn.Conv2d(out_channels, out_channels, kernel_size=self.filter_size, padding="same"),
                nn.GroupNorm(num_groups, out_channels),
                nn.SiLU(),
            ]
            return nn.Sequential(*(layers + final_layers))

    def bottleneck_block(self, in_channels, out_channels):
        """
        Bottleneck block at the center of the U-Net.

        Args:
            in_channels (int): Input channel size.
            out_channels (int): Output channel size.

        Returns:
            nn.Sequential: Bottleneck block.
        """
        num_groups = self._get_num_groups(out_channels)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.filter_size, padding="same"),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Dropout2d(0.1), 
            nn.Conv2d(out_channels, out_channels, kernel_size=self.filter_size, padding="same"),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
        )

    def upconv_block(self, in_channels, out_channels):
        """
        Upsampling block with interpolation and convolution.

        Args:
            in_channels (int): Input channel size.
            out_channels (int): Output channel size.

        Returns:
            nn.Sequential: Upsampling block.
        """
        num_groups = self._get_num_groups(out_channels)
        return nn.Sequential(
            Interpolate(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same"),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU()
        )