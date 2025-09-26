from torch import nn, stack

from ice_station_zebra.types import DataSpace, TensorNCHW, TensorNTCHW


class BaseEncoder(nn.Module):
    """Encoder that takes data in an input space and translates it to a smaller latent space.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_history_steps, latent_channels, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        data_space_in: DataSpace,
        data_space_out: DataSpace,
        n_history_steps: int,
    ) -> None:
        """Initialise a BaseEncoder."""
        super().__init__()
        self.data_space_in = data_space_in
        self.data_space_out = data_space_out
        self.name = data_space_in.name
        self.n_history_steps = n_history_steps

    def forward(self, x: TensorNTCHW) -> TensorNTCHW:
        """Forward step: encode input space into latent space.

        The default implementation simply calls `self.rollout` independently on each
        time slice. These are then stacked together to produce the final output.

        Args:
            x: TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

        Returns:
            TensorNTCHW with (batch_size, n_history_steps, latent_channels, latent_height, latent_width)

        """
        return stack(
            [
                # Apply rollout to each NCHW slice in the NTCHW input
                self.rollout(x[:, idx_t, :, :, :])
                for idx_t in range(self.n_history_steps)
            ],
            dim=1,
        )

    def rollout(self, x: TensorNCHW) -> TensorNCHW:
        """Single rollout step: encode NCHW input into NCHW latent space.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        msg = "If you are using the default forward method, you must implement rollout."
        raise NotImplementedError(msg)
