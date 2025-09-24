from typing import Any

from torch import nn

from ice_station_zebra.models.common import ConvBlockUpsample
from ice_station_zebra.types import DataSpace, TensorNCHW

from .base_decoder import BaseDecoder


class CNNDecoder(BaseDecoder):
    """Decoder that uses a convolutional neural net (CNN) to translate latent space back to data space.

    The layers are the reverse of those in the CNNEncoder.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_history_steps, latent_channels, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        output_space: DataSpace,
        activation: str = "ReLU",
        kernel_size: int = 3,
        n_layers: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialise a CNNDecoder."""
        super().__init__(**kwargs)

        # Construct list of layers
        layers: list[nn.Module] = []

        # Convolve to the necessary number of channels that will reduce down to the output channels
        n_channels = output_space.channels * (2**n_layers)
        layers.append(nn.Conv2d(self.n_latent_channels_total, n_channels, 1))

        # Add n_layers size-increasing convolutional blocks
        for _ in range(n_layers):
            layers.append(
                ConvBlockUpsample(
                    n_channels, activation=activation, kernel_size=kernel_size
                )
            )
            n_channels //= 2

        # Set the final spatial dimensions (previously used adaptive pooling which is extremely slow)
        layers.append(nn.Upsample(output_space.shape))

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def rollout(self, x: TensorNCHW) -> TensorNCHW:
        """Single rollout step: decode NCHW latent data into NCHW output.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return self.model(x)
