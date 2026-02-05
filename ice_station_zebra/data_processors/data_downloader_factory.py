from types import MappingProxyType

from omegaconf import DictConfig

from .preprocessors import IceNetSICPreprocessor, NullPreprocessor
from .zebra_data_processor import ZebraDataProcessor


class DataDownloaderFactory:
    preprocessors = MappingProxyType(
        {
            "None": NullPreprocessor,
            "IceNetSIC": IceNetSICPreprocessor,
        }
    )

    def __init__(self, config: DictConfig) -> None:
        """Initialise a DataDownloaderFactory from a config."""
        self.downloaders: list[ZebraDataProcessor] = []
        for dataset_name in config["data"]["datasets"]:
            cls_preprocessor = self.preprocessors[
                config["data"]["datasets"][dataset_name]
                .get("preprocessor", {})
                .get("type", "None")
            ]
            self.downloaders.append(
                ZebraDataProcessor(dataset_name, config, cls_preprocessor)  # type: ignore[type-abstract]
            )
