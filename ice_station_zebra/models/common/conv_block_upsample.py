from torch import nn

from ice_station_zebra.types import TensorNCHW

from .activations import ACTIVATION_FROM_NAME


class ConvBlockUpsample(nn.Module):
    """Convolutional block that doubles the resolution.

    (ConvTranspose2d → Normalization → Activation) → (ConvTranspose2d → Normalization → Activation)

    This is the reverse of ConvBlockDownsample.
    """

    def __init__(
        self,
        n_input_channels: int,
        *,
        activation: str = "ReLU",
        kernel_size: int = 3,
        n_output_channels: int | None = None,
    ) -> None:
        """Initialize the ConvBlockUpsample module.

        Args:
            activation: the activation function to use.
            kernel_size: the size of the convolutional kernel (odd numbers are preferable!).
            n_input_channels: the number of input channels.
            n_output_channels: the number of output channels (if None, half of n_input_channels).

        """
        super().__init__()
        activation_layer = ACTIVATION_FROM_NAME[activation]

        # Calculate convolutional parameters
        n_output_channels = (
            n_input_channels // 2 if n_output_channels is None else n_output_channels
        )
        padding = (kernel_size - 1) // 2
        output_padding = kernel_size % 2

        self.model = nn.Sequential(
            # Size reducing convolution/normalisation/activation
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
            # Size preserving convolution/normalisation/activation
            # Since ConvTranspose2d does not yet support `padding=same`, even-sized
            # kernels cannot preserve size. We therefore adjust the kernel size and
            # padding accordingly.
            nn.ConvTranspose2d(
                n_output_channels,
                n_output_channels,
                kernel_size=kernel_size + 1 - output_padding,
                padding=padding + 1 - output_padding,
            ),
            nn.BatchNorm2d(n_output_channels),
            activation_layer(inplace=True),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
