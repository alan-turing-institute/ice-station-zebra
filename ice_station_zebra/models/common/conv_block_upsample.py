from torch import nn

from ice_station_zebra.types import TensorNCHW

from .activations import ACTIVATION_FROM_NAME


class ConvBlockUpsample(nn.Module):
    """Convolutional block that doubles the resolution and halves the number of channels.

    This is the reverse of ConvBlockDownsample.
    """

    def __init__(
        self, n_input_channels: int, *, activation: str = "ReLU", kernel_size: int = 3
    ) -> None:
        """Initialize the ConvBlockUpsample module.

        Args:
            activation: the activation function to use.
            kernel_size: the size of the convolutional kernel.
            n_input_channels: the number of input channels.

        """
        super().__init__()
        activation_layer = ACTIVATION_FROM_NAME[activation]

        # Calculate convolutional parameters
        n_output_channels = n_input_channels // 2
        padding = (kernel_size - 1) // 2
        output_padding = kernel_size % 2

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                n_input_channels,
                n_output_channels,
                kernel_size=kernel_size,
                output_padding=output_padding,
                padding=padding,
                stride=2,
            ),
            nn.BatchNorm2d(n_output_channels),
            activation_layer(inplace=True),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
