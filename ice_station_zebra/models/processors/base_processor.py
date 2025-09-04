from torch import nn, stack

from ice_station_zebra.types import TensorNCHW, TensorNTCHW


class BaseProcessor(nn.Module):
    """Processor that converts latent input into latent output.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        n_forecast_steps: int,
        n_history_steps: int,
        n_latent_channels_total: int,
    ) -> None:
        """Initialise a NullProcessor."""
        super().__init__()
        self.n_forecast_steps = n_forecast_steps
        self.n_history_steps = n_history_steps
        self.n_latent_channels_total = n_latent_channels_total

    def forward(self, x: TensorNTCHW) -> TensorNTCHW:
        """Forward step: process in latent space.

        The default implementation simply calls `self.rollout` on each time slice until
        a sufficient number of forecast steps have been produced. These are then stacked
        together to produce the final output.

        If you want to handle the NTCHW tensors directly, simply override this method in
        your child class.

        Args:
            x: TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)

        """
        # Cut the NTCHW input into NCHW slices
        nchw_slices = [x[:, idx_t, :, :, :] for idx_t in range(self.n_history_steps)]

        # Run the model over the input slices, producing an output for each one.
        # Also append the predictions to the list of input slices, so that we can still
        # predict when n_forecast_steps > n_history_steps.
        outputs: list[TensorNCHW] = []
        for _ in range(self.n_forecast_steps):
            outputs.append(self.rollout(nchw_slices.pop(0)))
            nchw_slices.append(outputs[-1])

        # Stack the outputs up as a new time dimension
        return stack(outputs, dim=1)

    def rollout(self, x: TensorNCHW) -> TensorNCHW:
        """Single rollout step: process in NCHW latent space.

        Args:
            x: TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)

        """
        msg = "If you are using the default forward method, you must implement rollout."
        raise NotImplementedError(msg)
