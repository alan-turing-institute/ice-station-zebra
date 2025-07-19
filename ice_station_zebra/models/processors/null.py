import torch.nn as nn
from torch import Tensor


class NullProcessor(nn.Module):
    """Null model that simply returns input"""

    def __init__(self, n_latent_channels: int) -> None:
        super().__init__()
        self.n_latent_channels = n_latent_channels
        self.model = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
