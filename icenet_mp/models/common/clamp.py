from torch import nn

from icenet_mp.types import TensorNCHW


class Clamp(nn.Module):
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0) -> None:
        """Clamp the values of the input tensor to be within the given range."""
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Clamp the values of the input tensor to be within the given range."""
        return x.clamp(min=self.min_val, max=self.max_val)
