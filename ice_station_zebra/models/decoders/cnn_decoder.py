from typing import Any

from torch import nn

from ice_station_zebra.models.common import ConvBlockUpsample, ResizingInterpolation
from ice_station_zebra.types import TensorNCHW

from .base_decoder import BaseDecoder


class CNNDecoder(BaseDecoder):
    """Decoder that uses a convolutional neural net (CNN) to translate latent space back to data space.

    The layers are (almost) the reverse of those in the CNNEncoder, but moving the
    channel reduction step to the end.

    - Resize (if needed)
    - Batch normalisation
    - n_layers of size-increasing convolutional blocks
    - Convolve to required number of channels
    - Resize to output size (if needed)

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
        n_channels = self.data_space_in.channels
        if n_channels % layer_factor:
            msg = (
                f"The number of input channels {n_channels} must be divisible by {layer_factor}. "
                f"Without this, it is not possible to apply {n_layers} convolutions."
            )
            raise ValueError(msg)

        # Construct list of layers
        layers: list[nn.Module] = []

        # Ensure that the spatial dimensions are large enough that the size after
        # convolution will be at least as large as the desired output size.
        # N.B. Dividing by negative layer factor allows us to round up.
        minimal_initial_shape = (
            int(-(self.data_space_out.shape[0] // -layer_factor)),
            int(-(self.data_space_out.shape[1] // -layer_factor)),
        )
        if any(
            self.data_space_in.shape[idx] < minimal_initial_shape[idx]
            for idx in range(2)
        ):
            layers.append(ResizingInterpolation(minimal_initial_shape))
            conv_input_shape = minimal_initial_shape
        else:
            conv_input_shape = self.data_space_in.shape

        # Normalise the input across height and width separately for each channel
        layers.append(nn.BatchNorm2d(n_channels))

        # Add n_layers size-increasing convolutional blocks
        for _ in range(n_layers):
            layers.append(
                ConvBlockUpsample(
                    n_channels, activation=activation, kernel_size=kernel_size
                )
            )
            n_channels //= 2

        # If necessary, apply a final size-reducing layer to get the exact output shape
        conv_output_shape = tuple(dim * layer_factor for dim in conv_input_shape)
        if conv_output_shape != self.data_space_out.shape:
            layers.append(ResizingInterpolation(self.data_space_out.shape))

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
