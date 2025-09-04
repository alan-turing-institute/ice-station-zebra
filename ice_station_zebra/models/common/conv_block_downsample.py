from torch import nn

from ice_station_zebra.types import TensorNCHW

from .activations import ACTIVATION_FROM_NAME


class ConvBlockDownsample(nn.Module):
    """Convolutional block that halves the resolution and doubles the number of channels."""

    def __init__(
        self, n_input_channels: int, *, activation: str = "ReLU", kernel_size: int = 3
    ) -> None:
        """Initialize the ConvBlockDownsample module.

        Args:
            activation: the activation function to use.
            kernel_size: the size of the convolutional kernel.
            n_input_channels: the number of input channels.

        """
        super().__init__()
        activation_layer = ACTIVATION_FROM_NAME[activation]
        self.n_output_channels = 2 * n_input_channels
        self.model = nn.Sequential(
            nn.Conv2d(
                n_input_channels,
                self.n_output_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm2d(self.n_output_channels),
            activation_layer(inplace=True),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
