import logging
from typing import Any

from torch import nn

from ice_station_zebra.models.common import ConvBlockUpsample, ResizingInterpolation
from ice_station_zebra.types import TensorNCHW

from .base_decoder import BaseDecoder

logger = logging.getLogger(__name__)


class CNNDecoder(BaseDecoder):
    """Decoder that uses a convolutional neural net (CNN) to translate latent space back to data space.

    - Increase size with interpolation (if needed)
    - n_layers of size-increasing convolutional blocks
    - Decrease size with interpolation (if needed)
    - Convolve to number of output channels (if needed)

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
        """Initialise a CNNDecoder."""
        super().__init__(**kwargs)

        # Calculate the factor by which the scale changes after n_layers
        layer_factor = 2**n_layers

        # Ensure number of channels is divisible by the power of two implied by n_layers
        if self.data_space_in.channels % layer_factor:
            msg = (
                f"The number of input channels {self.data_space_in.channels} must be divisible by {layer_factor}. "
                f"Without this, it is not possible to apply {n_layers} convolutions."
            )
            raise ValueError(msg)

        # Calculate the minimal input shape to produce at least the desired output shape
        # N.B. dividing by a negative integer performs a ceiling division
        minimal_input_shape = (
            -(self.data_space_out.shape[0] // -layer_factor),
            -(self.data_space_out.shape[1] // -layer_factor),
        )

        # Construct list of layers
        layers: list[nn.Module] = []
        logger.debug("CNNDecoder (%s) with %d layers", self.name, n_layers)

        # If necessary, resize upwards until the post-convolution shape will be larger
        # than or equal to the desired output shape.
        shape = (
            max(minimal_input_shape[0], self.data_space_in.shape[0]),
            max(minimal_input_shape[1], self.data_space_in.shape[1]),
        )
        if shape != self.data_space_in.shape:
            layers.append(ResizingInterpolation(shape))
            logger.debug(
                "- ResizingInterpolation from %s to %s",
                self.data_space_in.shape,
                minimal_input_shape,
            )

        # Add n_layers size-increasing convolutional blocks
        n_channels = self.data_space_in.channels
        for _ in range(n_layers):
            layers.append(
                ConvBlockUpsample(
                    n_channels, activation=activation, kernel_size=kernel_size
                )
            )
            logger.debug(
                "- ConvBlockUpsample (%s, %s) with %d channels",
                activation,
                kernel_size,
                n_channels,
            )
            n_channels //= 2
            shape = (shape[0] * 2, shape[1] * 2)

        # If necessary, resize downwards to match the output shape
        if shape != self.data_space_out.shape:
            layers.append(ResizingInterpolation(self.data_space_out.shape))
            logger.debug(
                "- ResizingInterpolation from %s to %s",
                shape,
                self.data_space_out.shape,
            )

        # If necessary, convolve to the required number of output channels
        if n_channels != self.data_space_out.channels:
            layers.append(nn.Conv2d(n_channels, self.data_space_out.channels, 1))
            logger.debug(
                "- Channel convolution from %d to %d",
                n_channels,
                self.data_space_out.channels,
            )

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: decode latent space into output space with a CNN.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return self.model(x)
