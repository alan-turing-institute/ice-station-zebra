from pathlib import Path

from omegaconf import DictConfig


class IPreprocessor:
    def __init__(self, config: DictConfig) -> None:
        """Initialise the IPreprocessor base class."""
        self.name = str(config.get("preprocessor", {}).get("type", "None"))

    def download(self, preprocessor_path: Path) -> None:
        """Download data to the specified preprocessor path."""


class NullPreprocessor(IPreprocessor):
    pass
