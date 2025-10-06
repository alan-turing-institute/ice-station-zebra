import logging
from typing import Any

from torch import nn

from ice_station_zebra.models.common import ConvBlockDownsample, ResizingInterpolation
from ice_station_zebra.types import TensorNCHW

from .base_encoder import BaseEncoder

logger = logging.getLogger(__name__)


class CNNEncoder(BaseEncoder):
    """Encoder that uses a convolutional neural net (CNN) to translate data to a latent space.

    - n_layers of size-reducing convolutional blocks
    - Resize with interpolation (if needed)

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_history_steps, latent_channels, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        activation: str = "ReLU",
        kernel_size: int = 3,
        n_layers: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialise a CNNEncoder."""
        super().__init__(**kwargs)

        # Construct list of layers
        layers: list[nn.Module] = []
        logger.debug("CNNEncoder (%s) with %d layers", self.name, n_layers)

        # Add n_layers size-reducing convolutional blocks
        n_channels = self.data_space_in.channels
        for _ in range(n_layers):
            layers.append(
                ConvBlockDownsample(
                    n_channels, activation=activation, kernel_size=kernel_size
                )
            )
            logger.debug(
                "- ConvBlockDownsample (%s, %s) with %d channels",
                activation,
                kernel_size,
                n_channels,
            )
            n_channels *= 2

        # If necessary, apply a convolutional resizing to get the correct output dimensions
        shape = (
            self.data_space_in.shape[0] // (2**n_layers),
            self.data_space_in.shape[1] // (2**n_layers),
        )
        if shape != self.data_space_out.shape:
            layers.append(ResizingInterpolation(self.data_space_out.shape))
            logger.debug(
                "- ResizingInterpolation from %s to %s",
                shape,
                self.data_space_out.shape,
            )

        # Set the number of output channels correctly
        self.data_space_out.channels = n_channels

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: encode input space into latent space with a CNN.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return self.model(x)
