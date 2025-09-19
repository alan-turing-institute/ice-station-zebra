from torch import Tensor, nn

from .conv_norm_act import ConvNormAct


class ConvBlockUpsampleNaive(nn.Module):
    """Convolutional block that doubles the resolution.

    Upsample → ConvNormAct

    Note that using ConvTranspose2d as implemented in ConvBlockUpsample, is generally
    preferred (see e.g. https://discuss.pytorch.org/t/upsample-conv2d-vs-convtranspose2d/138081)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        norm_type: str = "batchnorm",
        activation: str = "ReLU",
        dropout_rate: float = 0.0,
    ) -> None:
        """Upsampling block with upsample and convolution.

        Args:
            in_channels (int): Input channel size.
            out_channels (int): Output channel size.
            kernel_size (int): Kernel size for the convolution after upsampling.
            norm_type (str): Type of normalization ("groupnorm", "batchnorm", or "none").
            activation (str): Name of activation function.
            dropout_rate (float): Dropout probability for ConvNormAct.

        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvNormAct(
                in_channels,
                out_channels,
                kernel_size,
                activation=activation,
                dropout_rate=dropout_rate,
                norm_type=norm_type,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
