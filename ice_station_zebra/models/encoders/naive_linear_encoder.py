from typing import Any

from torch import nn

from ice_station_zebra.types import TensorNCHW

from .base_encoder import BaseEncoder


class NaiveLinearEncoder(BaseEncoder):
    """Naive, linear encoder that takes data in an input space and translates it to a smaller latent space.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_history_steps, latent_channels, latent_height, latent_width)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise a NaiveLinearEncoder."""
        super().__init__(**kwargs)

        # Construct list of layers
        layers: list[nn.Module] = []

        # Start by normalising the input across height and width separately for each channel
        layers.append(nn.BatchNorm2d(self.data_space_in.channels))

        # Resample to the desired latent shape
        layers.append(nn.Upsample(self.data_space_out.shape))

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def rollout(self, x: TensorNCHW) -> TensorNCHW:
        """Single rollout step: encode NCHW input into NCHW latent space.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return self.model(x)
