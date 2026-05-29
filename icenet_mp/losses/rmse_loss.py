import torch
from torch import nn


class RMSELoss(nn.Module):
    """Root Mean Squared Error loss."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps  # avoids sqrt(0) producing NaN gradients

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(prediction, target) + self.eps)
