from pathlib import Path

from omegaconf import DictConfig

from ice_station_zebra.datasets import OSISAFDataset


def train(cfg: DictConfig) -> None:
    ds_osisaf = OSISAFDataset(Path(cfg.train.osisaf_path))
