from torch import nn

from icenet_mp.types import TensorNCHW

from .conv_norm_act import ConvNormAct


class ConvBlockDownsample(nn.Module):
    """Convolutional block that doubles channels and halves each spatial dimension.

    (Conv2d > Normalization > Activation) > (Conv2d > Normalization > Activation)

    Reverse of ConvBlockUpsample.
    """

    def __init__(
        self, n_input_channels: int, *, activation: str = "ReLU", kernel_size: int = 3
    ) -> None:
        """Initialize a ConvBlockDownsample module.

        Args:
            n_input_channels: the number of input channels.
            activation: the activation function to use.
            kernel_size: the size of the convolutional kernel.

        """
        super().__init__()

        # Calculate convolutional parameters
        n_output_channels = n_input_channels * 2
        padding = (kernel_size - 1) // 2

        self.model = nn.Sequential(
            # Size reducing convolution/normalisation/activation
            ConvNormAct(
                n_input_channels,
                n_output_channels,
                kernel_size=kernel_size,
                activation=activation,
                norm_type="batchnorm",
                padding=padding,
                stride=2,
            ),
            # Size preserving convolution/normalisation/activation
            ConvNormAct(
                n_output_channels,
                n_output_channels,
                kernel_size=kernel_size,
                activation=activation,
                norm_type="batchnorm",
            ),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
