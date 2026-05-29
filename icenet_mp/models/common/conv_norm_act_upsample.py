from torch import Tensor, nn

from .conv_norm_act import ConvNormAct


class ConvNormActUpsample(nn.Module):
    """Convolutional block that doubles each spatial dimension.

    Upsample > (Conv2d > Normalization > Activation)

    Prefer ConvBlockUpsample for most use cases (see https://discuss.pytorch.org/t/upsample-conv2d-vs-convtranspose2d/138081).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: str = "ReLU",
        dropout_rate: float = 0.0,
        kernel_size: int,
        norm_type: str = "batchnorm",
    ) -> None:
        """Initialize a ConvNormActUpsample module.

        Args:
            in_channels: the number of input channels.
            out_channels: the number of output channels (if None, half of in_channels).
            kernel_size: the size of the convolutional kernel.
            norm_type: the type of normalization ("groupnorm", "batchnorm", or "none").
            activation: the activation function to use.
            dropout_rate: the dropout probability.

        """
        super().__init__()

        out_channels = in_channels // 2 if out_channels is None else out_channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvNormAct(
                in_channels,
                out_channels,
                activation=activation,
                dropout_rate=dropout_rate,
                kernel_size=kernel_size,
                norm_type=norm_type,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
