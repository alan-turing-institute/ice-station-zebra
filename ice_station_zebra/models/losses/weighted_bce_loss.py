"""Weighted BCEWithLogitsLoss adapted from the IceNet repository:
https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb

Original implementation by the IceNet authors (icenet-ai).
"""

from torch import nn


class WeightedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, targets, sample_weights):
        """Weighted BCEWithLogitsLoss.

        Compute BCEWithLogitsLoss weighted by masking.
        Using BCEWithLogitsLoss instead of BCELoss, as it is more numerically stable:
        https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        """
        loss = super().forward(
            inputs.movedim(-2, 1), targets.movedim(-1, 1)
        ) * sample_weights.movedim(-1, 1)

        return loss.mean()
