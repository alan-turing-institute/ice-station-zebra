from types import MappingProxyType

from omegaconf import DictConfig

from .preprocessors import IceNetSICPreprocessor, NullPreprocessor
from .zebra_data_processor import ZebraDataProcessor


class ZebraDataProcessorFactory:
    preprocessors = MappingProxyType(
        {
            "None": NullPreprocessor,
            "IceNetSIC": IceNetSICPreprocessor,
        }
    )

    def __init__(self, config: DictConfig) -> None:
        self.datasets: list[ZebraDataProcessor] = []
        for dataset_name in config["datasets"]:
            cls_preprocessor = self.preprocessors[
                config["datasets"][dataset_name]
                .get("preprocessor", {})
                .get("type", "None")
            ]
            self.datasets.append(
                ZebraDataProcessor(dataset_name, config, cls_preprocessor)
            )
