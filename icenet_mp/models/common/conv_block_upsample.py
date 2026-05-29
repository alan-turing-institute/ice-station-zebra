from torch import nn

from icenet_mp.types import TensorNCHW

from .conv_norm_act import ConvNormAct


class ConvBlockUpsample(nn.Module):
    """Convolutional block that doubles each spatial dimension.

    If out_channels is not specified than this will halve the number of input channels.

    Upsample > (Conv2d > Normalization > Activation) > (Conv2d > Normalization > Activation)

    Reverse of ConvBlockDownsample, using upsampling to avoid checkerboarding.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        activation: str = "ReLU",
        kernel_size: int = 3,
        out_channels: int | None = None,
    ) -> None:
        """Initialize a ConvBlockUpsample module.

        Args:
            in_channels: the number of input channels.
            activation: the activation function to use.
            kernel_size: the size of the convolutional kernel.
            out_channels: the number of output channels (if None, half of in_channels).

        """
        super().__init__()

        out_channels = in_channels // 2 if out_channels is None else out_channels
        self.model = nn.Sequential(
            # Size increasing upsample
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # Size preserving convolution/normalisation/activation
            ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                activation=activation,
                norm_type="batchnorm",
            ),
            # Size preserving convolution/normalisation/activation
            ConvNormAct(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                activation=activation,
                norm_type="batchnorm",
            ),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.model(x)
