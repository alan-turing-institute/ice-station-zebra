from torch import Tensor, nn

from .activations import ACTIVATION_FROM_NAME


class TimeEmbed(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        activation: str = "SiLU",
    ) -> None:
        super().__init__()

        activation_layer = ACTIVATION_FROM_NAME[activation]

        self.model = nn.Sequential(
            nn.Linear(dim, dim * 4),
            activation_layer(inplace=True),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
