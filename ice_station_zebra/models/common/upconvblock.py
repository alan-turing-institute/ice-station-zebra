from typing import Type

import torch.nn as nn
from torch import Tensor
from .activations import get_activation


class UpconvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()

        act = lambda: get_activation(activation)

        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same"),
            act(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
