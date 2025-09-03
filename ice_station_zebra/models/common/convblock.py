from torch import Tensor, nn

from .activations import ACTIVATION_FROM_NAME


class ConvBlock(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int,
        activation: str = "ReLU",
        *,
        normalization: str | None = "batch",  # None, "batch", or "group"
        num_groups: int | None = None,
        activation_after_norm: bool = False,
        final: bool = False,
    ) -> None:
        """Initialise a flexible ConvBlock.

        Args:
            in_channels: Input channel size
            out_channels: Output channel size
            filter_size: Kernel size for convolutions
            final: Whether to add an extra conv layer at the end
            activation: Activation function name
            normalization: Type of normalization (None, "batch", or "group")
            num_groups: Number of groups for GroupNorm (required if normalization="group")
            activation_after_norm: If True, apply activation after normalization

        """
        super().__init__()

        if normalization == "group" and num_groups is None:
            msg = "num_groups must be specified when using GroupNorm"
            raise ValueError(msg)

        activation_layer = ACTIVATION_FROM_NAME[activation]
        layers = []

        if activation_after_norm:
            # First conv block
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=filter_size,
                        padding="same",
                    ),
                    nn.GroupNorm(num_groups, out_channels)
                    if normalization == "group"
                    else nn.BatchNorm2d(out_channels),
                    activation_layer(inplace=True),
                ]
            )

            # Second conv block
            layers.extend(
                [
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=filter_size,
                        padding="same",
                    ),
                    nn.GroupNorm(num_groups, out_channels)
                    if normalization == "group"
                    else nn.BatchNorm2d(out_channels),
                    activation_layer(inplace=True),
                ]
            )

            # Final conv block if requested
            if final:
                layers.extend(
                    [
                        nn.Conv2d(
                            out_channels,
                            out_channels,
                            kernel_size=filter_size,
                            padding="same",
                        ),
                        nn.GroupNorm(num_groups, out_channels)
                        if normalization == "group"
                        else nn.BatchNorm2d(out_channels),
                        activation_layer(inplace=True),
                    ]
                )
        else:
            # conv → activation, then norm only at the end if not final
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=filter_size,
                        padding="same",
                    ),
                    activation_layer(inplace=True),
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=filter_size,
                        padding="same",
                    ),
                    activation_layer(inplace=True),
                ]
            )

            if final:
                # Final block: add extra conv → activation (no norm)
                layers.extend(
                    [
                        nn.Conv2d(
                            out_channels,
                            out_channels,
                            kernel_size=filter_size,
                            padding="same",
                        ),
                        activation_layer(inplace=True),
                    ]
                )
            # Non-final: add normalization at the end
            elif normalization == "batch":
                layers.append(nn.BatchNorm2d(num_features=out_channels))
            elif normalization == "group":
                layers.append(nn.GroupNorm(num_groups, out_channels))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
