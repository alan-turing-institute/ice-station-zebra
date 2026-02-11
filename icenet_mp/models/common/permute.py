from torch import nn

from icenet_mp.types import TensorNCHW


class Permute(nn.Module):
    def __init__(self, permutation: tuple[int, ...]) -> None:
        """Apply a permutation of the dimensions of the input tensor."""
        super().__init__()
        self.permutation = permutation

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Permute the dimensions of the input tensor.

        To avoid https://github.com/pytorch/pytorch/issues/142344 we reorder the output
        to be contiguous after permutation.
        """
        return x.permute(self.permutation).contiguous()
