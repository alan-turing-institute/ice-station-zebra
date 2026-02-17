"""Calculating RMSE, MAE by forecast step"""

import torch
from torchmetrics import Metric

class RMSE_daily(Metric):

    def __init__(self) -> None:
        """Initialize the RMSE_daily metric.

        Parameters
        ----------
        pixel_size: int, optional
            Physical size of one pixel in kilometers (default is 25 km -> OSISAF).

        """
        super().__init__()
        self.rmse_daily: torch.Tensor
        self.add_state("rmse_daily", default=torch.tensor([]), dist_reduce_fx="cat")
                
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        _sample_weight: torch.Tensor | None = None,
    ) -> None:
        """Update the RMSE_daily accumulators.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground truth values.
        sample_weight : Optional[torch.Tensor]
            Ignored (present for API compatibility).

        """
        print("Updating RMSE_daily metric...")
        rmse = torch.sqrt(torch.mean((preds - target)**2, dim=(0, 2, 3, 4)))
        print("rmse shape:", rmse.shape)
        print("number of elements in rmse_daily before update:", self.rmse_daily.numel())
        if self.rmse_daily.numel() == 0:
            self.rmse_daily = rmse.unsqueeze(1)  # Shape: (T,)
        else:
            self.rmse_daily = torch.cat((self.rmse_daily, rmse.unsqueeze(1)), dim=1)  # Shape: (T, N)
        print("rmse_daily:", self.rmse_daily)
       
    def compute(self) -> dict[str, list[str] | list[float]]:
        """Compute the final RMSE_daily values."""
        print("Computing RMSE_daily metric...")
        print("compute:", self.rmse_daily)
        print("shape of rmse_daily:", self.rmse_daily.shape)
        values = torch.mean(self.rmse_daily, dim=1)
        print("RMSE_daily values:", values)
        return values
        
class MAE_daily(Metric):

    def __init__(self) -> None:
        """Initialize the MAE_daily metric.

        Parameters
        ----------
        pixel_size: int, optional
            Physical size of one pixel in kilometers (default is 25 km -> OSISAF).

        """
        super().__init__()
        self.mae_daily: torch.Tensor
        self.add_state("mae_daily", default=torch.tensor([]), dist_reduce_fx="cat")
                
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        _sample_weight: torch.Tensor | None = None,
    ) -> None:
        """Update the MAE_daily accumulators.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground truth values.
        sample_weight : Optional[torch.Tensor]
            Ignored (present for API compatibility).

        """
        print("Updating MAE_daily metric...")
        mae = torch.mean(torch.abs(preds - target), dim=(0, 2, 3, 4))
        print("mae shape:", mae.shape)
        if self.mae_daily.numel() == 0:
            self.mae_daily = mae.unsqueeze(1)  # Shape: (T,)
        else:
            self.mae_daily = torch.cat((self.mae_daily, mae.unsqueeze(1)), dim=1)  # Shape: (T, N)
        print("mae_daily:", self.mae_daily)
        
    def compute(self) -> dict[str, list[str] | list[float]]:
        """Compute the final MAE_daily values."""
        print("Computing MAE_daily metric...")
        print("compute:", self.mae_daily)
        print("shape of mae_daily:", self.mae_daily.shape)
        values = torch.mean(self.mae_daily, dim=1)
        print("MAE_daily values:", values)
        return values
        