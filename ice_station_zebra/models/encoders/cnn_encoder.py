from typing import Any

from torch import nn

from ice_station_zebra.models.common import ConvBlockDownsample
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

        # Start by normalising the input across height and width separately for each channel
        n_channels = input_space.channels
        layers.append(nn.BatchNorm2d(n_channels))

        # Set the initial spatial dimensions (previously used adaptive pooling which is extremely slow)
        initial_conv_shape = (
            latent_space.shape[0] * (2**n_layers),
            latent_space.shape[1] * (2**n_layers),
        )
        layers.append(nn.Upsample(initial_conv_shape))

        # Add n_layers size-reducing convolutional blocks
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
