from omegaconf import DictConfig


class IPreprocessor:
    def __init__(self, config: DictConfig, data_path: str):
        pass

    def download(self) -> None:
        pass

    def outputs(self) -> dict[str, str]:
        return {}


class NullPreprocessor(IPreprocessor):
    pass
