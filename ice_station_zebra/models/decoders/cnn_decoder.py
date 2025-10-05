import logging
from typing import Any

from torch import nn

from ice_station_zebra.models.common import (
    ConvBlockUpsample,
    ResizingConvolution,
    ResizingInterpolation,
)
from ice_station_zebra.types import TensorNCHW

from .base_decoder import BaseDecoder

logger = logging.getLogger(__name__)


class CNNDecoder(BaseDecoder):
    """Decoder that uses a convolutional neural net (CNN) to translate latent space back to data space.

    The layers are (almost) the reverse of those in the CNNEncoder, but moving the
    channel reduction step to the end.

    - Increase size with interpolation (if needed)
    - n_layers of size-increasing convolutional blocks
    - Decrease size with convolution (if needed)

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

        # If necessary, increase the input size until the post-convolution size will be
        # larger than or equal to the desired output size.
        upscaled_initial_shape = (
            max(minimal_input_shape[0], self.data_space_in.shape[0]),
            max(minimal_input_shape[1], self.data_space_in.shape[1]),
        )
        if upscaled_initial_shape != self.data_space_in.shape:
            layers.append(ResizingInterpolation(upscaled_initial_shape))

        # Add n_layers size-increasing convolutional blocks
        n_channels = self.data_space_in.channels
        for _ in range(n_layers):
            layers.append(
                ConvBlockUpsample(
                    n_channels, activation=activation, kernel_size=kernel_size
                )
            )
            n_channels //= 2

        # If necessary, apply a convolutional resizing to get the correct output dimensions
        conv_output_shape = tuple(dim * layer_factor for dim in upscaled_initial_shape)
        if (conv_output_shape != self.data_space_out.shape) or (
            n_channels != self.data_space_out.channels
        ):
            layers.append(
                ResizingConvolution(
                    n_channels,
                    conv_output_shape,
                    self.data_space_out.channels,
                    self.data_space_out.shape,
                )
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
