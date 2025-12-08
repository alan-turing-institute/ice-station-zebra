# mypy: ignore-errors
"""SIEError metric adapted from the IceNet repository.

The repository is available at:
https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb.

Original implementation by the IceNet authors (icenet-ai). Edited by: Maria Carolina Novitasari.
"""

import torch
from torchmetrics import Metric

SEA_ICE_THRESHOLD = 0.15  # Threshold for defining sea ice extent


class SIEError(Metric):
    """Sea Ice Extent error metric (in km^2) for use at multiple lead times."""

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = True

    def __init__(self, leadtimes_to_evaluate: list[int]) -> None:
        """Initialize the SIEError metric.

        Parameters
        ----------
        leadtimes_to_evaluate : List[int]
            Indices of lead times at which SIE should be evaluated.

        """
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate
        self.add_state("pred_sie", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("true_sie", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        _sample_weight: torch.Tensor | None = None,
    ) -> None:
        """Update the SIE accumulators.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground truth values.
        sample_weight : Optional[torch.Tensor]
            Ignored (present for API compatibility).

        """
        preds = (preds > SEA_ICE_THRESHOLD).long()
        target = (target > SEA_ICE_THRESHOLD).long()

        self.pred_sie += (
            preds[:, :, :, self.leadtimes_to_evaluate].sum().to(self.pred_sie.dtype)
        )
        self.true_sie += (
            target[:, :, :, self.leadtimes_to_evaluate].sum().to(self.true_sie.dtype)
        )

    def compute(self) -> torch.Tensor:
        """Compute the final Sea Ice Extent error in km²."""
        return (self.pred_sie - self.true_sie) * 25**2  # each pixel is 25x25 km
