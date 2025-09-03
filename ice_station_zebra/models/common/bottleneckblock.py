from torch import Tensor, nn

from .activations import ACTIVATION_FROM_NAME


class BottleneckBlock(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        *,
        filter_size: int,
        activation: str = "ReLU",
        normalization: str = "batch",  # "batch" or "group"
        num_groups: int | None = None,  # Required for GroupNorm
        dropout_rate: float = 0.0,
        activation_after_norm: bool = False,
    ) -> None:
        """Initialise a flexible BottleneckBlock.

        Args:
            in_channels: Input channel size
            out_channels: Output channel size
            filter_size: Kernel size for convolutions
            activation: Activation function name
            normalization: Type of normalization ("batch" or "group")
            num_groups: Number of groups for GroupNorm (required if normalization="group")
            dropout_rate: Dropout rate (0.0 means no dropout)
            activation_after_norm: If True, apply activation after normalization
                                 If False, apply activation after convolution

        """
        super().__init__()

        if normalization == "group" and num_groups is None:
            msg = "num_groups must be specified when using GroupNorm"
            raise ValueError(msg)

        activation_layer = ACTIVATION_FROM_NAME[activation]

        # Build the sequential model based on configuration
        layers = []

        # First conv + norm + activation
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=filter_size, padding="same"
            )
        )

        if not activation_after_norm:
            layers.append(activation_layer(inplace=True))

        if normalization == "batch":
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        elif normalization == "group":
            layers.append(nn.GroupNorm(num_groups, out_channels))

        if activation_after_norm:
            layers.append(activation_layer(inplace=True))

        # Optional dropout after first block
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))

        # Second conv + norm + activation
        layers.append(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=filter_size, padding="same"
            )
        )

        if not activation_after_norm:
            layers.append(activation_layer(inplace=True))

        if normalization == "batch":
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        elif normalization == "group":
            layers.append(nn.GroupNorm(num_groups, out_channels))

        if activation_after_norm:
            layers.append(activation_layer(inplace=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
