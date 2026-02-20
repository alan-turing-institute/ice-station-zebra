"""Calculating RMSE, MAE by forecast step."""

import torch
from torchmetrics import Metric


class RMSEDaily(Metric):
    def __init__(self) -> None:
        """Initialize the RMSEDaily metric."""
        super().__init__()
        self.rmse_daily: torch.Tensor
        self.add_state("rmse_daily", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        _sample_weight: torch.Tensor | None = None,
    ) -> None:
        """Update the RMSEDaily accumulators.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground truth values.
        sample_weight : Optional[torch.Tensor]
            Ignored (present for API compatibility).

        """
        rmse = torch.sqrt(torch.mean((preds - target) ** 2, dim=(0, 2, 3, 4)))
        if self.rmse_daily.numel() == 0:
            self.rmse_daily = rmse.unsqueeze(1)  # Shape: (T,)
        else:
            self.rmse_daily = torch.cat(
                (self.rmse_daily, rmse.unsqueeze(1)), dim=1
            )  # Shape: (T, N)

    def compute(self) -> torch.Tensor:
        """Compute the final RMSEDaily values."""
        return torch.mean(self.rmse_daily, dim=1)


class MAEDaily(Metric):
    def __init__(self) -> None:
        """Initialize the MAEDaily metric."""
        super().__init__()
        self.mae_daily: torch.Tensor
        self.add_state("mae_daily", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        _sample_weight: torch.Tensor | None = None,
    ) -> None:
        """Update the MAEDaily accumulators.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground truth values.
        sample_weight : Optional[torch.Tensor]
            Ignored (present for API compatibility).

        """
        mae = torch.mean(torch.abs(preds - target), dim=(0, 2, 3, 4))
        if self.mae_daily.numel() == 0:
            self.mae_daily = mae.unsqueeze(1)  # Shape: (T,)
        else:
            self.mae_daily = torch.cat(
                (self.mae_daily, mae.unsqueeze(1)), dim=1
            )  # Shape: (T, N)

    def compute(self) -> torch.Tensor:
        """Compute the final MAEDaily values."""
        return torch.mean(self.mae_daily, dim=1)
