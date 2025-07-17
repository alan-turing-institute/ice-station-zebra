import torch.nn as nn
from torch import Tensor

from .base_processor import BaseProcessor


class NullProcessor(BaseProcessor):
    """Null model that simply returns input"""

    def __init__(self, n_latent_channels: int) -> None:
        super().__init__(n_latent_channels)
        self.model = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
