"""Weighted BCEWithLogitsLoss adapted from the IceNet repository.

https://github.com/icenet-ai/icenet-notebooks/blob/main/pytorch/1_icenet_forecast_unet.ipynb

Original implementation by the IceNet authors (icenet-ai). Edited by: Maria Carolina Novitasari.
"""


class WeightedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """BCEWithLogits loss with elementwise weighting."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        inputs: Tensor,
        targets: Tensor,
        sample_weights: Tensor,
    ) -> Tensor:
        """Weighted BCEWithLogitsLoss.

        Compute BCEWithLogitsLoss weighted by masking.
        Using BCEWithLogitsLoss instead of BCELoss, as it is more numerically stable:
        https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        """
        loss = super().forward(
            inputs.movedim(-2, 1),
            targets.movedim(-1, 1),
        ) * sample_weights.movedim(-1, 1)

        return loss.mean()
