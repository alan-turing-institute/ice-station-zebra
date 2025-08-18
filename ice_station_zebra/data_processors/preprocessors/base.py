from pathlib import Path

from omegaconf import DictConfig


class IPreprocessor:
    def __init__(self, config: DictConfig) -> None:
        self.name = str(config.get("preprocessor", {}).get("type", "None"))

    def download(self, preprocessor_path: Path) -> None:
        pass


class NullPreprocessor(IPreprocessor):
    pass
