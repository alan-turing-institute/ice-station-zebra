from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


class UnconditionalCheckpoint(Callback):
    """A callback to summarise metrics during evaluation."""

    def __init__(self, on_train_end: bool = False) -> None:
        """Save a checkpoint unconditionally.

        Args:
            on_train_end: Whether to save a checkpoint at the end of training
        """
        super().__init__()
        self.impl = ModelCheckpoint()
        self._on_train_end = on_train_end

    @property
    def dirpath(self) -> str:
        """Return the directory path where checkpoints are saved."""
        return self.impl.dirpath

    @dirpath.setter
    def dirpath(self, value) -> None:
        """Set the directory path where checkpoints are saved."""
        self.impl.dirpath = value

    def on_train_end(self, trainer: Trainer, _: LightningModule) -> None:
        """Called when training ends."""
        if self._on_train_end:
            self.save_unconditionally(trainer)

    def save_unconditionally(self, trainer: Trainer) -> None:
        """Save a checkpoint unconditionally."""
        monitor_candidates = self.impl._monitor_candidates(trainer)
        self.impl._save_none_monitor_checkpoint(trainer, monitor_candidates)
