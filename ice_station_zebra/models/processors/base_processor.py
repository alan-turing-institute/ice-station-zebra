from abc import ABC, abstractmethod

from torch import nn, stack

from ice_station_zebra.types import TensorNCHW, TensorNTCHW


class BaseProcessor(nn.Module, ABC):
    """Processor that converts latent input into latent output.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels, latent_height, latent_width)
    """

    def __init__(
        self, *, n_forecast_steps: int, n_history_steps: int, n_latent_channels: int
    ) -> None:
        """Initialise a NullProcessor."""
        super().__init__()
        self.n_forecast_steps = n_forecast_steps
        self.n_history_steps = n_history_steps
        self.n_latent_channels = n_latent_channels

    def forward_nchw(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: process in NCHW latent space.

        Args:
            x: TensorNTCHW with (batch_size, n_history_steps, n_latent_channels, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels, latent_height, latent_width)

        """
        msg = "If you are using the default forward_ntchw, you must implement forward_nchw."
        raise NotImplementedError(msg)

    def forward_ntchw(self, x: TensorNTCHW) -> TensorNTCHW:
        """Forward step: process in NTCHW latent space.

        This is a default implementation of the forward step that assumes the child
        class has implemented forward_nchw. It can be overridden by children that want
        to implement a time-aware forward step.

        Args:
            x: TensorNTCHW with (batch_size, n_history_steps, n_latent_channels, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels, latent_height, latent_width)

        """
        # Cut the NTCHW input into NCHW slices
        nchw_slices = [x[:, idx_t, :, :, :] for idx_t in range(self.n_history_steps)]

        # Run the model over the input slices, producing an output for each one.
        # Also append the predictions to the list of input slices, so that we can still
        # predict when n_forecast_steps > n_history_steps.
        outputs: list[TensorNCHW] = []
        for _ in range(self.n_forecast_steps):
            outputs.append(self.forward_nchw(nchw_slices.pop(0)))
            nchw_slices.append(outputs[-1])

        # Stack the outputs up as a new time dimension
        return stack(outputs, dim=1)

    @abstractmethod
    def forward(self, x: TensorNTCHW) -> TensorNTCHW:
        """Forward step: process in latent space.

        To use the default timestep-by-timestep implementation, you must implement
        forward_nchw and then simply call `self.forward_ntchw(x)` in this method.

        Args:
            x: TensorNTCHW with (batch_size, n_history_steps, n_latent_channels, latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels, latent_height, latent_width)

        """
