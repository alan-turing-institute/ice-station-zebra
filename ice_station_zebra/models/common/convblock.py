import torch.nn as nn
from torch import Tensor

from .activations import ACTIVATION_FROM_NAME


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        filter_size: int,
        final: bool = False,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()

        activation_layer = ACTIVATION_FROM_NAME[activation]

        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size=filter_size, padding="same"
            ),
            activation_layer(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=filter_size, padding="same"
            ),
            activation_layer(inplace=True),
        ]
        if final:
            layers += [
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=filter_size,
                    padding="same",
                ),
                activation_layer(inplace=True),
            ]

        else:
            layers.append(
                nn.BatchNorm2d(num_features=out_channels),
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
