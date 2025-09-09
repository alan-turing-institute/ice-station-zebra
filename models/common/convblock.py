from torch import Tensor, nn

from .convnormact import ConvNormAct, get_num_groups


class CommonConvBlock(nn.Module):
    """Full convolutional block: two ConvNormAct stacked, with optional final layer.

    Args:
        in_channels (int): Input channel size.
        out_channels (int): Output channel size.
        kernel_size (int): Kernel size for the convolutions.
        norm_type (str): Type of normalization ("groupnorm", "batchnorm", or "none").
        activation (str): Name of the activation function (from ACTIVATION_FROM_NAME).
        final (bool): If True, adds an extra ConvNormAct layer at the end.
        dropout_rate (float): Dropout probability for each ConvNormAct block.

    Notes:
        - If `norm_type="groupnorm"`, `num_groups` is automatically chosen
          based on the number of output channels.
        - Dropout is applied after activation if `dropout_rate > 0.0`.

    """

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_type: str = "groupnorm",
        activation: str = "SiLU",
        *,
        final: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialise a CommonConvBlock."""
        super().__init__()
        self.final = final
        norm_type = norm_type.lower()

        # Determine num_groups only if using GroupNorm
        num_groups = get_num_groups(out_channels) if norm_type == "groupnorm" else None

        # Create stacked ConvNormAct blocks
        self.block1 = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size,
            norm_type,
            num_groups,
            activation,
            dropout_rate,
        )
        self.block2 = ConvNormAct(
            out_channels,
            out_channels,
            kernel_size,
            norm_type,
            num_groups,
            activation,
            dropout_rate,
        )

        self.final_layer = (
            ConvNormAct(
                out_channels,
                out_channels,
                kernel_size,
                norm_type,
                num_groups,
                activation,
                dropout_rate,
            )
            if final
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply stacked ConvNormAct layers to input tensor."""
        x = self.block1(x)
        x = self.block2(x)
        if self.final_layer is not None:
            x = self.final_layer(x)
        return x


"""
# Usage examples:

# GroupNorm without dropout, no final layer
block_gn = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="groupnorm",
    dropout_rate=0.0,
    final=False
)

# BatchNorm without dropout, no final layer
block_bn = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="batchnorm",
    dropout_rate=0.0,
    final=False
)

# No normalization, no dropout
block_none = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="none",
    dropout_rate=0.0,
    final=False
)

# GroupNorm with extra final layer
block_final = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="groupnorm",
    dropout_rate=0.0,
    final=True
)

# GroupNorm with dropout (10%) but no final layer
block_dropout = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="groupnorm",
    dropout_rate=0.1,
    final=False
)

# GroupNorm with both final layer and dropout (Bottleneck-style)
block_bottleneck = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="groupnorm",
    dropout_rate=0.1,
    final=True
)
"""