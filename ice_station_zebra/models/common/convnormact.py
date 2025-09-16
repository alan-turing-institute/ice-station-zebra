from torch import Tensor, nn

from .activations import ACTIVATION_FROM_NAME


class ConvNormAct(nn.Module):
    """Mini block: Conv2d → Normalization → Activation, optional Dropout.

    Args:
        in_channels: Input channel size.
        out_channels: Output channel size.
        kernel_size: Kernel size for the convolution.
        norm_type: Type of normalization ("groupnorm", "batchnorm", or "none").
        activation: Name of the activation function (from ACTIVATION_FROM_NAME).
        dropout_rate: Dropout probability. If 0.0, dropout is not applied.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_type: str = "batchnorm",
        activation: str = "ReLU",
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialise a ConvNormAct block."""
        super().__init__()
        norm_type = norm_type.lower()
        norm_layer: nn.Module
        if norm_type == "groupnorm":
            norm_layer = nn.GroupNorm(self._get_num_groups(out_channels), out_channels)
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

        activation_layer = ACTIVATION_FROM_NAME[activation](inplace=True)
        dropout_layer = (
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            norm_layer,
            activation_layer,
            dropout_layer,
        )

    def _get_num_groups(self, channels: int) -> int:
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

    def forward(self, x: Tensor) -> Tensor:
        """Apply ConvNormAct block to input tensor."""
        return self.block(x)
