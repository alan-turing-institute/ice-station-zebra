from omegaconf import DictConfig

from .preprocessors import IceNetSICPreprocessor, NullPreprocessor
from .zebra_dataset import ZebraDataset


class ZebraDatasetFactory:
    preprocessors = {
        "None": NullPreprocessor,
        "IceNetSIC": IceNetSICPreprocessor,
    }

    def __init__(self, config: DictConfig) -> None:
        self.datasets: list[ZebraDataset] = []
        for dataset_name in config["datasets"]:
            cls_preprocessor = self.preprocessors[
                config["datasets"][dataset_name]
                .get("preprocessor", {})
                .get("type", "None")
            ]
            self.datasets.append(ZebraDataset(dataset_name, config, cls_preprocessor))
