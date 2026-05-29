from torch import nn

from icenet_mp.types import TensorNCHW

from .activations import ACTIVATION_FROM_NAME


class ConvBlockUpsample(nn.Module):
    """Convolutional block that doubles each spatial dimension.

    (ConvTranspose2d > Normalization > Activation) > (ConvTranspose2d > Normalization > Activation)

    Reverse of ConvBlockDownsample.
    """

    def __init__(
        self,
        n_input_channels: int,
        *,
        activation: str = "ReLU",
        kernel_size: int = 3,
        n_output_channels: int | None = None,
    ) -> None:
        """Initialize a ConvBlockUpsample module.

        Args:
            n_input_channels: the number of input channels.
            activation: the activation function to use.
            kernel_size: the base size of the convolutional kernel (must be odd).
            n_output_channels: the number of output channels (if None, half of n_input_channels).

        """
        super().__init__()
        activation_layer = ACTIVATION_FROM_NAME[activation]

        # Calculate convolutional parameters
        n_output_channels = (
            n_input_channels // 2 if n_output_channels is None else n_output_channels
        )
        # Since ConvTranspose2d does not support `padding=same`, even-sized kernels
        # cannot preserve size.
        if (kernel_size % 2) == 0:
            msg = "`kernel_size` must be odd to preserve spatial dimensions."
            raise ValueError(msg)

        self.model = nn.Sequential(
            # Size increasing convolution/normalisation/activation
            # We use an Upsample layer to avoid checkerboarding artifacts (see
            # https://discuss.pytorch.org/t/upsample-conv2d-vs-convtranspose2d/138081).
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ConvTranspose2d(
                n_input_channels,
                n_output_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm2d(n_output_channels),
            activation_layer(inplace=True),
            # Size preserving convolution/normalisation/activation
            nn.ConvTranspose2d(
                n_output_channels,
                n_output_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm2d(n_output_channels),
            activation_layer(inplace=True),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
