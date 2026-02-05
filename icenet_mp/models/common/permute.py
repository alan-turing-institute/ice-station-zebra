from torch import nn

from icenet_mp.types import TensorNCHW


class Permute(nn.Module):
    def __init__(self, permutation: tuple[int, ...]) -> None:
        """Apply a permutation of the dimensions of the input tensor."""
        super().__init__()
        self.permutation = permutation

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Permute the dimensions of the input tensor."""
        return x.permute(self.permutation)
