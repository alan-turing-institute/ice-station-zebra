"""SIEError metric adapted from the IceNet repository:
https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb

Original implementation by the IceNet authors (icenet-ai).
"""

import torch
from torchmetrics import Metric


class SIEError(Metric):
    """Sea Ice Extent error metric (in km^2) for use at multiple leadtimes."""

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = True

    def __init__(self, leadtimes_to_evaluate: list):
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate
        self.add_state("pred_sie", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("true_sie", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor
    ):
        preds = (preds > 0.15).long()
        target = (target > 0.15).long()
        self.pred_sie += preds[:, :, :, self.leadtimes_to_evaluate].sum()
        self.true_sie += target[:, :, :, self.leadtimes_to_evaluate].sum()

    def compute(self):
        return (self.pred_sie - self.true_sie) * 25**2  # each pixel is 25x25 km
