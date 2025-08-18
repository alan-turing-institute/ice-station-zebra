from torch import nn

from ice_station_zebra.types import TensorNCHW


class NullProcessor(nn.Module):
    """Null model that simply returns input.

    Operations all occur in latent space:
        TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)
    """

    def __init__(self, n_latent_channels: int) -> None:
        super().__init__()
        self.n_latent_channels = n_latent_channels
        self.model = nn.Identity()

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: process in latent space.

        Args:
            x: TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return self.model(x)
