import torch.nn as nn
from torch import Tensor


class TimeEmbed(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
