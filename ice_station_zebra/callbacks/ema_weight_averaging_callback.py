from .weight_averaging import WeightAveraging


class EMAWeightAveragingCallback(WeightAveraging):
    """A callback that updates an averaged model for Exponential Moving Average (EMA) after each training step."""

    def __init__(
        self, *, every_n_epochs: int | None = None, every_n_steps: int | None = None
    ) -> None:
        """Summarise metrics during evaluation.

        Args:
            every_n_epochs: How many epochs to wait before updating.
            every_n_steps: How many steps to wait before updating.

        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.every_n_steps = every_n_steps

    def should_update(
        self, step_idx: int | None = None, epoch_idx: int | None = None
    ) -> bool:
        """Update if we are at the requested number of steps or epochs."""
        if self.every_n_epochs and epoch_idx:
            return epoch_idx % self.every_n_epochs == 0

        if self.every_n_steps and step_idx:
            return step_idx % self.every_n_steps == 0

        return False
