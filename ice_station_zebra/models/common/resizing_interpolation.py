from collections.abc import Sequence

from torch import nn

from ice_station_zebra.types import TensorNCHW


class ResizingInterpolation(nn.Module):
    """Resize to an arbitrary output shape using interpolation.

    The output shape must be specified at initialisation time.
    """

    def __init__(self, output_shape: Sequence[int]) -> None:
        """Initialize the ResizingInterpolate module.

        Args:
            output_shape: the target output size in H x W format.

        """
        super().__init__()
        self.output_shape = (output_shape[0], output_shape[1])

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return nn.functional.interpolate(
            x,
            size=self.output_shape,
            mode="bilinear",
            align_corners=True,
            antialias=True,
        )
