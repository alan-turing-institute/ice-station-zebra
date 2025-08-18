import torch.nn as nn
from torch import Tensor

from .activations import get_activation


class TimeEmbed(nn.Module):
    def __init__(self, dim: int = 256, 
                 activation: str = "SiLU",) -> None:
        super().__init__()

        def act():
            return get_activation(activation)

        self.model = nn.Sequential(
            nn.Linear(dim, dim * 4),
            act(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
