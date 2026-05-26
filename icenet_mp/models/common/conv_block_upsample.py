from torch import nn

from icenet_mp.types import TensorNCHW

from .activations import ACTIVATION_FROM_NAME


class ConvBlockUpsample(nn.Module):
    """Convolutional block that reduces channels and doubles each spatial dimension.

    (ConvTranspose2d > Normalization > Activation) > (ConvTranspose2d > Normalization > Activation)

    Reverse of ConvBlockDownsample.
    Preferred over ConvBlockUpsampleNaive for most use cases (see https://discuss.pytorch.org/t/upsample-conv2d-vs-convtranspose2d/138081).
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
            kernel_size: the base size of the convolutional kernel. The size-increasing
                convolution needs an even kernel and the size-preserving convolution
                needs an odd kernel, so one will use `kernel_size` and the other will
                use `kernel_size + 1`.
            n_output_channels: the number of output channels (if None, half of n_input_channels).

        """
        super().__init__()
        activation_layer = ACTIVATION_FROM_NAME[activation]

        # Calculate convolutional parameters
        n_output_channels = (
            n_input_channels // 2 if n_output_channels is None else n_output_channels
        )
        kernel_size_odd = kernel_size if kernel_size % 2 else kernel_size + 1
        kernel_size_even = kernel_size + 1 if kernel_size % 2 else kernel_size

        self.model = nn.Sequential(
            # Size increasing convolution/normalisation/activation
            # To avoid checkerboarding, kernel size must be a multiple of stride.
            # We therefore use an even kernel size with appropriate padding.
            nn.ConvTranspose2d(
                n_input_channels,
                n_output_channels,
                kernel_size=kernel_size_even,
                padding=(kernel_size_even - 2) // 2,
                stride=2,
            ),
            nn.BatchNorm2d(n_output_channels),
            activation_layer(inplace=True),
            # Size preserving convolution/normalisation/activation
            # Since ConvTranspose2d does not yet support `padding=same`, even-sized
            # kernels cannot preserve size.
            # We therefore use an odd kernel size with appropriate padding.
            nn.ConvTranspose2d(
                n_output_channels,
                n_output_channels,
                kernel_size=kernel_size_odd,
                padding=(kernel_size_odd - 1) // 2,
            ),
            nn.BatchNorm2d(n_output_channels),
            activation_layer(inplace=True),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
