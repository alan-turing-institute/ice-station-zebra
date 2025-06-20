import hydra
from omegaconf import DictConfig
from pathlib import Path
from ice_station_zebra.datasets import OSISAFDataset


@hydra.main(version_base=None, config_path="../config", config_name="zebra")
def train(cfg: DictConfig) -> None:
    ds_osisaf = OSISAFDataset(Path(cfg.train.osisaf_path))
