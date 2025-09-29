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
        max_kernel_size: int = 20,
    ) -> None:
        """Initialize the ResizingConvolution module.

        Args:
            input_channels: the number of input channels.
            input_shape: the input shape in H x W format.
            output_channels: the number of output channels
            output_shape: the output shape in H x W format.
            max_kernel_size: the maximum kernel size to use. Larger kernels will give a more accurate resize but are slower.

        """
        super().__init__()

        # Construct list of layers
        layers: list[nn.Module] = []

        strides = (
            max(input_shape[0] // output_shape[0], 1),
            max(input_shape[1] // output_shape[1], 1),
        )
        out_to_in_scale = (
            input_shape[0] - (output_shape[0] - 1) * strides[0],
            input_shape[1] - (output_shape[1] - 1) * strides[1],
        )
        kernel_sizes = (
            min(max(out_to_in_scale[0], 1), max_kernel_size),
            min(max(out_to_in_scale[1], 1), max_kernel_size),
        )
        padding = (
            max((kernel_sizes[0] - out_to_in_scale[0]) // 2, 0),
            max((kernel_sizes[1] - out_to_in_scale[1]) // 2, 0),
        )

        # Create the convolution layer
        layers.append(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_sizes,
                stride=strides,
                padding=padding,
            )
        )

        # If necessary, apply an additional resizing step
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
