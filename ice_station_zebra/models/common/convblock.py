from typing import Type

import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        filter_size: int,
        final: bool = False,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size=filter_size, padding="same"
            ),
            activation(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=filter_size, padding="same"
            ),
            activation(),
        ]
        if final:
            layers += [
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=filter_size,
                    padding="same",
                ),
                activation(),
            ]

        else:
            layers.append(
                nn.BatchNorm2d(num_features=out_channels),
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
