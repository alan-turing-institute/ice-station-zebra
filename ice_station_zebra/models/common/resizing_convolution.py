from collections.abc import Sequence

from torch import nn

from ice_station_zebra.types import TensorNCHW

from .resizing_interpolation import ResizingInterpolation


class ResizingConvolution(nn.Module):
    """Resize to an arbitrary output shape using convolution.

    The output shape must be specified at initialisation time.
    """

    def __init__(
        self,
        input_channels: int,
        input_shape: Sequence[int],
        output_channels: int,
        output_shape: Sequence[int],
    ) -> None:
        """Initialize the ResizingConvolution module.

        Args:
            input_channels: the number of input channels.
            input_shape: the input shape in H x W format.
            output_channels: the number of output channels
            output_shape: the output shape in H x W format.

        """
        super().__init__()

        strides = (
            max(input_shape[0] // output_shape[0], 1),
            max(input_shape[1] // output_shape[1], 1),
        )
        scales_ = (
            input_shape[0] - (output_shape[0] - 1) * strides[0],
            input_shape[1] - (output_shape[1] - 1) * strides[1],
        )
        kernel_sizes = (
            max(scales_[0], 1),
            max(scales_[1], 1),
        )
        padding = (
            max((kernel_sizes[0] - scales_[0]) // 2, 0),
            max((kernel_sizes[1] - scales_[1]) // 2, 0),
        )


        # Create the convolution layer
        layers = [
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_sizes,
                stride=strides,
                padding=padding,
            )
        ]

        # Check whether an additional resizing step is needed
        conv_output_shape = (
            int(((input_shape[0] + 2 * padding[0] - kernel_sizes[0]) / strides[0]) + 1),
            int(((input_shape[1] + 2 * padding[1] - kernel_sizes[1]) / strides[1]) + 1),
        )
        if conv_output_shape != output_shape:
            layers.append(ResizingInterpolation(output_shape))

        # Build the model
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
