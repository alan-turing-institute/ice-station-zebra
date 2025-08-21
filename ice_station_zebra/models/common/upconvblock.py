from torch import Tensor, nn

from .activations import ACTIVATION_FROM_NAME


class UpconvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "ReLU",
    ) -> None:
        """Initialise an UpconvBlock."""
        super().__init__()

        activation_layer = ACTIVATION_FROM_NAME[activation]

        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same"),
            activation_layer(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
