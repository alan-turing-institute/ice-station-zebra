from torch import Tensor, nn

from .convnormact import ConvNormAct


class CommonConvBlock(nn.Module):
    """Full convolutional block: two ConvNormAct stacked, with optional final layer.

    Args:
        in_channels (int): Input channel size.
        out_channels (int): Output channel size.
        kernel_size (int): Kernel size for the convolutions.
        norm_type (str): Type of normalization ("groupnorm", "batchnorm", or "none").
        activation (str): Name of the activation function (from ACTIVATION_FROM_NAME).
        n_subblocks (int): n_subblocks (int): Number of ConvNormAct blocks to stack (default 2).
        dropout_rate (float): Dropout probability for each ConvNormAct block.

    """

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_type: str = "batchnorm",
        activation: str = "ReLU",
        *,
        n_subblocks: int = 2,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialise a CommonConvBlock."""
        super().__init__()

        # Create stacked ConvNormAct blocks
        layers = []
        for i in range(n_subblocks):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(
                ConvNormAct(
                    in_ch,
                    out_channels,
                    kernel_size,
                    norm_type,
                    activation,
                    dropout_rate,
                )
            )

        # Combine into Sequential
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Apply stacked ConvNormAct layers to input tensor."""
        return self.layers(x)


"""
# Usage examples:

# GroupNorm without dropout, no final layer (2 blocks)
block_gn = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="groupnorm",
    dropout_rate=0.0,
)

# BatchNorm without dropout, no final layer (2 blocks)
block_bn = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="batchnorm",
    dropout_rate=0.0,
)

# No normalization, no dropout
block_none = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="none",
    dropout_rate=0.0,
)

# GroupNorm with extra final layer (GroupNorm with 3 blocks)
block_final = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="groupnorm",
    dropout_rate=0.0,
    n_subblocks=3
)

# GroupNorm with dropout (10%) but no final layer
block_dropout = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="groupnorm",
    dropout_rate=0.1,
)

# GroupNorm with both final layer and 10% dropout (Bottleneck-style)
block_bottleneck = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="groupnorm",
    dropout_rate=0.1,
    n_subblocks=3
)
"""
