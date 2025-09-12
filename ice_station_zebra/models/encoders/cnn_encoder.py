from typing import Any

from torch import nn

from ice_station_zebra.models.common import ConvBlockDownsample, ResizingAveragePool2d
from ice_station_zebra.types import DataSpace, TensorNCHW

from .base_encoder import BaseEncoder


class CNNEncoder(BaseEncoder):
    """Encoder that uses a convolutional neural net (CNN) to translate data to a latent space.

    The layers are the reverse of those in the CNNDecoder.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_history_steps, latent_channels, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        input_space: DataSpace,
        latent_space: DataSpace,
        activation: str = "ReLU",
        kernel_size: int = 3,
        n_layers: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialise a CNNEncoder."""
        super().__init__(name=input_space.name, **kwargs)

        # Construct list of layers
        layers: list[nn.Module] = []

        # Start with an adaptive pooling layer that sets the initial spatial dimensions
        initial_conv_shape = [size * (2**n_layers) for size in latent_space.shape]
        layers.append(ResizingAveragePool2d(input_space.shape, initial_conv_shape))

        # Add n_layers size-reducing convolutional blocks
        n_channels = input_space.channels
        for _ in range(n_layers):
            layers.append(
                ConvBlockDownsample(
                    n_channels, activation=activation, kernel_size=kernel_size
                )
            )
            n_channels *= 2

        # Convolve to the desired number of latent channels
        layers.append(nn.Conv2d(n_channels, latent_space.channels, 1))

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def rollout(self, x: TensorNCHW) -> TensorNCHW:
        """Single rollout step: encode NCHW input into NCHW latent space.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return self.model(x)
