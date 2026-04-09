from collections.abc import Sequence
from typing import Any

from icenet_mp.types import TensorNCHW

from .base_encoder import BaseEncoder


class ReprojectionEncoder(BaseEncoder):
    """Encoder that reprojects data from a source projection to a target projection.

    Each cell in the target projection takes its value from the nearest neighbour in the
    source.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, latent_height, latent_width)
    """

    def __init__(
        self,
        source_latitudes: Sequence[float],
        source_longitudes: Sequence[float],
        target_latitudes: Sequence[float],
        target_longitudes: Sequence[float],
        **kwargs: Any,
    ) -> None:
        """Initialise a ReprojectionEncoder."""
        super().__init__(**kwargs)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: encode input space into latent space by splitting into patches.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return x
