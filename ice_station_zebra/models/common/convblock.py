import torch.nn as nn
from torch import Tensor
from .activations import get_activation


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

        def act():
            return get_activation(activation)


        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size=filter_size, padding="same"
            ),
            act(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=filter_size, padding="same"
            ),
            act(),
        ]
        if final:
            layers += [
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=filter_size,
                    padding="same",
                ),
                act(),
            ]

        else:
            layers.append(
                nn.BatchNorm2d(num_features=out_channels),
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
