"""SIEError metric."""

import torch
from torchmetrics import Metric

SEA_ICE_THRESHOLD = 0.15  # Threshold for defining sea ice extent


class SIEErrorNew(Metric):
    """Sea Ice Extent error metric (in km^2) for use at multiple lead times."""

    def __init__(self, pixel_size: int = 25) -> None:
        """Initialize the SIEError metric.

        Parameters
        ----------
        pixel_size: int, optional
            Physical size of one pixel in kilometers (default is 25 km -> OSISAF).

        """
        super().__init__()
        self.sie_error: torch.Tensor
        self.add_state("sie_error", default=torch.tensor([]), dist_reduce_fx="cat")

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
        preds = preds > SEA_ICE_THRESHOLD
        target = target > SEA_ICE_THRESHOLD
        print("preds shape:", preds.shape)
        print("target shape:", target.shape)

        # Calculate the SIE for each day of the forecast
        pred_sie = torch.sum(preds, dim=(0, 2, 3, 4))  # type: ignore[operator]
        print("pred_sie:", pred_sie)
        true_sie = torch.sum(target, dim=(0, 2, 3, 4))  # type: ignore[operator]
        print("true_sie:", true_sie)
        error = (pred_sie - true_sie).float().to(self.device)
        print("SIE error for each day:", error)
        # Reshape to (T, 1) to stack horizontally across epochs/batches
        if self.sie_error.numel() == 0:
            self.sie_error = error.unsqueeze(1)  # Shape: (T,)
        else:
            self.sie_error = torch.cat((self.sie_error, error.unsqueeze(1)), dim=1)  # Shape: (T, N)
        print("sie_error:", self.sie_error)

    def compute(self) -> torch.Tensor:
        """Compute the final Sea Ice Extent error in km²."""
        print("Computing SIEError metric...")
        print("compute:", self.sie_error)
        print("shape of sie_error:", self.sie_error.shape)
        print("abs value:"  , torch.abs(self.sie_error))
        print("mean abs value across batches:", torch.mean(torch.abs(self.sie_error), dim=1))
        return torch.mean(torch.abs(self.sie_error), dim=1) * self.pixel_size**2  # type: ignore[operator]
