"""IceNetAccuracy metric adapted from the IceNet repository.

The repository is available at: https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb

Original implementation by the IceNet authors (icenet-ai). Edited by Maria Carolina Novitasari.
"""

import torch
from torchmetrics import Metric

THRESHOLD = 0.15  # Threshold for binarizing predictions and targets


class IceNetAccuracy(Metric):
    """Binary accuracy metric for use at multiple leadtimes."""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True

    def __init__(self, leadtimes_to_evaluate: list) -> None:
        """Initialize the IceNetAccuracy metric."""
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate
        self.add_state(
            "weighted_score", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "possible_score", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor
    ) -> None:
        """Update metric state with a new batch of predictions and targets."""
        preds = (preds > THRESHOLD).long()
        target = (target > THRESHOLD).long()
        sample_weight = sample_weight.squeeze(-1)
        base_score = (
            preds[:, :, :, self.leadtimes_to_evaluate]
            == target[:, :, :, self.leadtimes_to_evaluate]
        )
        self.weighted_score += torch.sum(
            base_score * sample_weight[:, :, :, self.leadtimes_to_evaluate]
        )  # type: ignore
        self.possible_score += torch.sum(
            sample_weight[:, :, :, self.leadtimes_to_evaluate]
        )  # type: ignore

    def compute(self) -> torch.Tensor:
        """Compute the final accuracy metric as a percentage."""
        return self.weighted_score.float() / self.possible_score * 100.0
