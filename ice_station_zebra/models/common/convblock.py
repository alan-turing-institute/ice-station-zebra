from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        filter_size: int,
        final: bool = False,
    ) -> None:
        """Initialise a ConvBlock."""
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size=filter_size, padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=filter_size, padding="same"
            ),
            nn.ReLU(inplace=True),
        ]
        if final:
            layers += [
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=filter_size,
                    padding="same",
                ),
                nn.ReLU(inplace=True),
            ]

        else:
            layers.append(
                nn.BatchNorm2d(num_features=out_channels),
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
