import math

from torch import nn

from ice_station_zebra.types import TensorNCHW


class ResizingAveragePool2d(nn.Module):
    """Resize to an arbitrary output shape using average pooling.

    This performs an upscaling and then downsamples using average pooling.
    The input and output sizes must be specified at initialisation time.
    """

    def __init__(
        self, input_size: tuple[int, int], output_size: tuple[int, int]
    ) -> None:
        """Initialize the ResizingAveragePool2d module.

        Args:
            input_size: the target input size in H x W format.
            output_size: the target output size in H x W format.

        """
        super().__init__()

        # Calculate an upsampling factor that will leave the input larger than the output
        upsample_scale = math.ceil(
            max(
                [
                    size_out / size_in
                    for size_in, size_out in zip(input_size, output_size, strict=True)
                ]
                + [1]
            )
        )

        # We may have different stride/kernel sizes in the H and W dimensions
        # These values will ensure that the output is the correct size
        strides = [
            input_size[idx] * upsample_scale // output_size[idx] for idx in range(2)
        ]
        kernel_sizes = [
            input_size[idx] * upsample_scale - (output_size[idx] - 1) * strides[idx]
            for idx in range(2)
        ]

        # We upsample and then pool down to the desired output size
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=upsample_scale),
            nn.AvgPool2d(kernel_size=kernel_sizes, stride=strides, padding=0),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
