import math
from collections.abc import Sequence

from torch import nn

from ice_station_zebra.types import TensorNCHW


class ResizingShuffle(nn.Module):
    """Resize to an arbitrary output shape by reshaping and padding.

    The input channels and shape and then output shape must be specified at initialisation time.
    """

    def __init__(
        self,
        input_channels: int,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
    ) -> None:
        """Initialize the ResizingShuffle module.

        Args:
            input_channels: the number of input channels.
            input_shape: the input shape in H x W format.
            output_shape: the output shape in H x W format.

        """
        super().__init__()
        input_size = input_channels * input_shape[0] * input_shape[1]
        self.output_h = output_shape[0]
        self.output_w = output_shape[1]
        self.output_channels = math.ceil(input_size / self.output_h / self.output_w)
        output_size = self.output_channels * self.output_h * self.output_w
        self.padding = output_size - input_size

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: pad and reshape input to the desired output shape."""
        # Flatten to N, C_in * H_in * W_in
        x = x.flatten(start_dim=1)
        # Pad to N, C_in * H_in * W_in + padding
        x = nn.functional.pad(x, (0, self.padding))
        # Reshape to N, C_out, H_out, W_out
        return x.reshape(-1, self.output_channels, self.output_h, self.output_w)
