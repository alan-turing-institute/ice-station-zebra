from torch import Tensor, nn

from .activations import ACTIVATION_FROM_NAME


def get_num_groups(channels: int) -> int:
    """Determine the maximum number of groups that divide `channels` for GroupNorm.

    Args:
        channels (int): Number of feature channels.

    Returns:
        int: Optimal number of groups.

    """
    num_groups = 8  # Start with preferred group count
    while num_groups > 1:
        if channels % num_groups == 0:
            return num_groups
        num_groups -= 1
    return 1  # Fallback to GroupNorm(1,...), equivalent to LayerNorm


class ConvNormAct(nn.Module):
    """Mini block: Conv2d → Normalization → Activation, optional Dropout.

    Args:
        in_channels: Input channel size.
        out_channels: Output channel size.
        kernel_size: Kernel size for the convolution.
        norm_type: Type of normalization ("groupnorm", "batchnorm", or "none").
        num_groups: Number of groups for GroupNorm (required if norm_type="groupnorm").
        activation: Name of the activation function (from ACTIVATION_FROM_NAME).
        dropout_rate: Dropout probability. If 0.0, dropout is not applied.

    """

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_type: str = "groupnorm",
        num_groups: int | None = None,
        activation: str = "ReLU",
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialise a ConvNormAct block."""
        super().__init__()
        norm_type = norm_type.lower()
        if norm_type == "groupnorm":
            if num_groups is None:
                msg = "num_groups must be specified for GroupNorm"
                raise ValueError(msg)
            norm_layer = nn.GroupNorm(num_groups, out_channels)
        elif norm_type == "batchnorm":
            norm_layer = nn.BatchNorm2d(out_channels)
        elif norm_type == "none":
            norm_layer = nn.Identity()
        else:
            msg = (
                f"Unknown norm_type: {norm_type}. "
                "Choose 'groupnorm', 'batchnorm', or 'none'"
            )
            raise ValueError(msg)

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            norm_layer,
            ACTIVATION_FROM_NAME[activation](inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Apply ConvNormAct block to input tensor."""
        return self.block(x)
