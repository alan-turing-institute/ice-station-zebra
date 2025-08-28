from torch import Tensor, nn

from .activations import ACTIVATION_FROM_NAME


class TimeEmbed(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        activation: str = "SiLU",
    ) -> None:
        """This module takes pre-computed sinusoidal time embeddings (from _timestep_embedding())
        and projects them through fully-connected layers to learn a more useful representation
        for the diffusion task. The MLP uses a dim→4*dim→dim expansion to provide additional
        flexibility for adapting the fixed sinusoidal embeddings.

        Args:
            dim (int, optional): Size of the input and output embedding dimension.
                Defaults to 256.
            activation (str, optional): Name of the activation function to use
                (e.g., "SiLU", "ReLU"). Defaults to "SiLU".

        """
        super().__init__()

        activation_layer = ACTIVATION_FROM_NAME[activation]

        self.model = nn.Sequential(
            nn.Linear(dim, dim * 4),
            activation_layer(inplace=True),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
