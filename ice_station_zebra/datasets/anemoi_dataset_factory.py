from omegaconf import DictConfig

from .anemoi_dataset import AnemoiDataset
from .preprocessors import IceNetSICPreprocessor, NullPreprocessor


class AnemoiDatasetFactory:
    preprocessors = {
        "None": NullPreprocessor,
        "IceNetSIC": IceNetSICPreprocessor,
    }

    def __init__(self, config: DictConfig) -> None:
        self.datasets: list[AnemoiDataset] = []
        for dataset_name in config["datasets"]:
            cls_preprocessor = self.preprocessors[
                config["datasets"][dataset_name]
                .get("preprocessor", {})
                .get("type", "None")
            ]
            self.datasets.append(AnemoiDataset(dataset_name, config, cls_preprocessor))
