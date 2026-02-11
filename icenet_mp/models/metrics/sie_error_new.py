"""SIEError metric."""

from typing import Any
import torch
from torchmetrics import Metric

SEA_ICE_THRESHOLD = 0.15  # Threshold for defining sea ice extent


class SIEErrorNew(Metric):
    """Sea Ice Extent error metric (in km^2) for use at multiple lead times."""

    def __init__(self, forecast_step: int, pixel_size: int = 25) -> None:
        """Initialize the SIEError metric.

        Parameters
        ----------
        forecast_step: int
            Index of the forecast step at which SIE should be evaluated.
        pixel_size: int, optional
            Physical size of one pixel in kilometers (default is 25 km -> OSISAF).

        """
        super().__init__()
        print("Initializing SIEError metric with forecast_step:", forecast_step)
        self.forecast_step = forecast_step
        self.add_state("sie_error", default=torch.Tensor([]), dist_reduce_fx="cat")
        self.pixel_size = pixel_size

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
        print("Updating SIEError metric...")
        # preds = (preds > SEA_ICE_THRESHOLD).long()
        # target = (target > SEA_ICE_THRESHOLD).long()
        preds = (preds > SEA_ICE_THRESHOLD)
        target = (target > SEA_ICE_THRESHOLD)
        print("preds shape:", preds.shape)
        print("target shape:", target.shape)
        
        pred_sie = torch.sum(preds[:, self.forecast_step])  # type: ignore[operator]
        print("pred_sie:", pred_sie)
        true_sie = torch.sum(target[:, self.forecast_step])  # type: ignore[operator]
        print("true_sie:", true_sie)
        error = torch.Tensor([pred_sie - true_sie]).to(self.device)
        self.sie_error = torch.cat((self.sie_error, error))
        print("sie_error:", self.sie_error)
        
    def compute(self) -> torch.Tensor:        
        """Compute the final Sea Ice Extent error in km²."""
        print("Computing SIEError metric...")
        print("compute:", self.sie_error)
        print("type of sie_error:", type(self.sie_error))
        return torch.mean(torch.abs(self.sie_error)) * self.pixel_size**2  # type: ignore[operator]
    
    # plot_lower_bound: Optional[float] = None
    # plot_upper_bound: Optional[float] = None

    # def plot(
    #     self, *args: Any, **kwargs: Any) -> Any:
    #     print("args:", args)
    #     print("kwargs:", kwargs)
    #     return self._plot(*args, **kwargs)
