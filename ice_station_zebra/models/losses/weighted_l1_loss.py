"""Weighted L1Loss adapted from the IceNet repository:
https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb

Original implementation by the IceNet authors (icenet-ai).
"""

import torch
from torch import nn


class WeightedL1Loss(nn.L1Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, targets, sample_weights):
        """Weighted L1 loss.

        Compute L1 loss weighted by masking.
        """
        y_hat = torch.sigmoid(inputs)
        loss = super().forward(100 * y_hat, 100 * targets) * sample_weights
        return loss.mean()
