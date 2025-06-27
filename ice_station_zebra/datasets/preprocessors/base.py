from omegaconf import DictConfig


class IPreprocessor:
    def __init__(self, config: DictConfig):
        self.name = config.get("preprocessor", {}).get("type", "None")

    def download(self, data_path: str) -> None:
        pass


class NullPreprocessor(IPreprocessor):
    pass
