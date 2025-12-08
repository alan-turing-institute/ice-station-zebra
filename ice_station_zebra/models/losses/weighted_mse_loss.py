"""Weighted MSELoss adapted from the IceNet repository.

https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb

Original implementation by the IceNet authors (icenet-ai). Edited by: Maria Carolina Novitasari.
"""

from typing import Any

from torch import Tensor, nn


class WeightedMSELoss(nn.MSELoss):
    """Mean-squared error loss with per-element weighting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the WeightedMSELoss.

        Args:
            *args: Positional arguments passed to torch.nn.MSELoss.
            **kwargs: Keyword arguments passed to torch.nn.MSELoss.

        """
        super().__init__(*args, **kwargs)

    def forward(
        self,
        inputs: Tensor,
        targets: Tensor,
        sample_weights: Tensor,
    ) -> Tensor:
        """Compute weighted mean squared error loss.

        Args:
            inputs (Tensor): Predicted values.
            targets (Tensor): Ground-truth values.
            sample_weights (Tensor): Elementwise weighting tensor.

        Returns:
            Tensor: Scalar weighted loss value.

        """
        y_hat = inputs.squeeze()
        targets = targets.squeeze()
        sample_weights = sample_weights.squeeze()

        loss = super().forward(100 * y_hat, 100 * targets) * sample_weights
        return loss.mean()
