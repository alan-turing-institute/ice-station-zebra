from collections.abc import ABC, abstractmethod
from pathlib import Path

from omegaconf import DictConfig


class IPreprocessor(ABC):
    def __init__(self, config: DictConfig) -> None:
        """Initialise the IPreprocessor base class."""
        self.cls_name = str(config.get("preprocessor", {}).get("type", "None"))
        self.dataset_name = str(config.get("name", "None"))

    @abstractmethod
    def download(self, preprocessor_path: Path) -> None:
        """Download data to the specified preprocessor path."""


class NullPreprocessor(IPreprocessor):
    pass
