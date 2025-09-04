from typing import Any

from torch import nn

from ice_station_zebra.models.common import ConvBlockUpsample, ResizingAveragePool2d
from ice_station_zebra.types import DataSpace, TensorNCHW

from .base_decoder import BaseDecoder


class CNNDecoder(BaseDecoder):
    """Decoder that uses a convolutional neural net (CNN) to translate latent space back to data space.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_history_steps, latent_channels, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        latent_space: DataSpace,
        output_space: DataSpace,
        n_layers: int = 4,
        activation: str = "ReLU",
        **kwargs: Any,
    ) -> None:
        """Initialise a CNNDecoder."""
        super().__init__(**kwargs)

        # Construct list of layers
        layers: list[nn.Module] = []

        # Add n_layers size-increasing convolutional blocks
        n_channels = self.n_latent_channels_total
        for _ in range(n_layers):
            layers.append(ConvBlockUpsample(n_channels, activation=activation))
            n_channels //= 2

        # Add an adaptive pooling layer that sets the final spatial dimensions
        final_conv_shape = [size * (2**n_layers) for size in latent_space.shape]
        layers.append(ResizingAveragePool2d(final_conv_shape, output_space.shape))

        # Convolve to the desired number of latent channels
        layers.append(nn.Conv2d(n_channels, output_space.channels, 1))

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def rollout(self, x: TensorNCHW) -> TensorNCHW:
        """Apply CNNDecoder to NCHW tensor.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return self.model(x)
