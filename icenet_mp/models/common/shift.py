import torch
from torch import nn

from icenet_mp.types import TensorNCHW


class Shift(nn.Module):
    def __init__(self, *, scale: bool, offset: bool) -> None:
        """Apply a scale and offset to the input tensor."""
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor(1), requires_grad=True) if scale else 1.0
        self.offset = (
            nn.Parameter(torch.Tensor(0), requires_grad=True) if offset else 0.0
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Apply a scale and offset to the input tensor."""
        return x * self.scale + self.offset
