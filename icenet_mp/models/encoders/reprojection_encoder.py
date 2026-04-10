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

    def __init__(self, project_to: str, **kwargs: Any) -> None:
        """Initialise a ReprojectionEncoder."""
        super().__init__(**kwargs)

        # Name of the output data space to project to
        self.project_to = project_to

        # In order to avoid input/output latitudes and longitudes being recorded as
        # model parameters, we set them later on
        self.input_latitudes: list[float] = []
        self.input_longitudes: list[float] = []
        self.output_latitudes: list[float] = []
        self.output_longitudes: list[float] = []

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: encode input space into latent space by splitting into patches.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return x

    def set_latlon(
        self, name: str, latitudes: list[float], longitudes: list[float]
    ) -> None:
        if name == self.name:
            self.input_latitudes = latitudes
            self.input_longitudes = longitudes
        if name == self.project_to:
            self.output_latitudes = latitudes
            self.output_longitudes = longitudes
