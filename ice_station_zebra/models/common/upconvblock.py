import torch.nn as nn
from torch import Tensor
from typing import Type

class UpconvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same"),
            activation(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
