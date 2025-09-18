from torch import nn, stack

from ice_station_zebra.types import TensorNCHW, TensorNTCHW


class BaseDecoder(nn.Module):
    """Decoder that takes data in a latent space and translates it to a larger output space.

    Latent space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)
    """

    def __init__(self, *, n_forecast_steps: int, n_latent_channels_total: int) -> None:
        """Initialise a BaseDecoder."""
        super().__init__()
        self.n_forecast_steps = n_forecast_steps
        self.n_latent_channels_total = n_latent_channels_total

    def forward(self, x: TensorNTCHW) -> TensorNTCHW:
        """Forward step: decode latent space into output space.

        The default implementation simply calls `self.rollout` independently on each
        time slice. These are then stacked together to produce the final output.

        Args:
            x: TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)

        """
        return stack(
            [
                # Apply rollout to each NCHW slice in the NTCHW input
                self.rollout(x[:, idx_t, :, :, :])
                for idx_t in range(self.n_forecast_steps)
            ],
            dim=1,
        )

    def rollout(self, x: TensorNCHW) -> TensorNCHW:
        """Single rollout step: decode NCHW latent data into NCHW output.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, output_channels, output_height, output_width)

        """
        msg = "If you are using the default forward method, you must implement rollout."
        raise NotImplementedError(msg)
