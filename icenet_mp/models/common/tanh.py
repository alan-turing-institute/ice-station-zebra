from torch import nn, tanh

from icenet_mp.types import TensorNCHW


class Tanh(nn.Module):
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0) -> None:
        """Apply tanh to the input tensor and scale it to be within the given range."""
        super().__init__()
        self.scale = max_val - min_val
        self.offset = min_val

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Apply tanh to the input tensor and scale it to be within the given range."""
        return self.offset + self.scale * (tanh(x) + 1) / 2
