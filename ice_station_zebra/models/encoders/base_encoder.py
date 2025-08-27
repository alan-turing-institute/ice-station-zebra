from abc import ABC, abstractmethod

from torch import nn

from ice_station_zebra.types import TensorNTCHW


class BaseEncoder(nn.Module, ABC):
    """Encoder that takes data in an input space and translates it to a smaller latent space.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_history_steps, latent_channels, latent_height, latent_width)
    """

    def __init__(self, *, name: str, n_history_steps: int) -> None:
        """Initialise a BaseEncoder."""
        super().__init__()
        self.name = name
        self.n_history_steps = n_history_steps

    @abstractmethod
    def forward(self, x: TensorNTCHW) -> TensorNTCHW:
        """Forward step: encode input space into latent space.

        Args:
            x: TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

        Returns:
            TensorNTCHW with (batch_size, n_history_steps, latent_channels, latent_height, latent_width)

        """
