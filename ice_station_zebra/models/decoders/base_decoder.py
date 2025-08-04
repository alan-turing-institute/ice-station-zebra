from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseDecoder(nn.Module, ABC):
    """
    Decoder that takes data in a latent space and translates it to a larger output space

    Latent space:
        Tensor[NCHW] with (batch_size, latent_channels, latent_height, latent_width)

    Output space:
        Tensor[NTCHW] with (batch_size, n_forecast_steps, output_channels, output_height, output_width)
    """

    def __init__(self, *, n_forecast_steps: int) -> None:
        super().__init__()
        self.n_forecast_steps = n_forecast_steps

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Transformation summary

        Args:
            x: Tensor[NCHW] with (batch_size, latent_channels, latent_height, latent_width)

        Returns:
            Tensor[NTCHW] with (batch_size, n_forecast_steps, output_channels, output_height, output_width)
        """
