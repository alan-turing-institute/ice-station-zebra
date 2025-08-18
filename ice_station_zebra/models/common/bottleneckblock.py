from torch import Tensor, nn


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        filter_size: int,
    ) -> None:
        """Initialise a BottleneckBlock."""
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=filter_size, padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=filter_size, padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
