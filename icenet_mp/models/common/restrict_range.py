from torch import nn, sigmoid, tanh

from icenet_mp.types import RangeRestriction, TensorNCHW


class RestrictRange(nn.Module):
    def __init__(
        self, method: RangeRestriction, *, min_val: float = 0.0, max_val: float = 1.0
    ) -> None:
        """Restrict the values of the input tensor to be within the given range.

        This can use torch.clamp, torch.sigmoid, or torch.tanh.
        """
        super().__init__()
        self.method = method
        if method == RangeRestriction.CLAMP:
            self.restrict_fn = lambda x: x.clamp(min=min_val, max=max_val)
        elif method == RangeRestriction.SIGMOID:
            self.restrict_fn = lambda x: min_val + (max_val - min_val) * sigmoid(x)
        elif method == RangeRestriction.TANH:
            self.restrict_fn = (
                lambda x: min_val + (max_val - min_val) * (tanh(x) + 1) / 2
            )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Apply the restriction function to the input tensor."""
        return self.restrict_fn(x)
