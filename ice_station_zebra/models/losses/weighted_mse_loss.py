"""Weighted MSELoss adapted from the IceNet repository:
https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb

Original implementation by the IceNet authors (icenet-ai).
"""

from torch import nn


class WeightedMSELoss(nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, targets, sample_weights):
        """Weighted MSE loss.

        Compute MSE loss weighted by masking.
        """
        y_hat = inputs.squeeze()
        targets = targets.squeeze()
        sample_weights = sample_weights.squeeze()

        loss = super().forward(100 * y_hat, 100 * targets) * sample_weights
        return loss.mean()
