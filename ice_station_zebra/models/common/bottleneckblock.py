from typing import Type

import torch.nn as nn
from torch import Tensor
from .activations import get_activation


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        filter_size: int,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()

        act = lambda: get_activation(activation)

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=filter_size, padding="same"
            ),
            act(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=filter_size, padding="same"
            ),
            act(),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
