from functools import cached_property

from anemoi.training.train.train import AnemoiTrainer
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from .anemoi_adaptors import TrainerConfigAdaptor


class ZebraTrainer(AnemoiTrainer):
    """An Anemoi trainer with no graph"""
    def __init__(self, config: DictConfig) -> None:
        """Initialize the Zebra trainer."""
        super().__init__(TrainerConfigAdaptor(config))

    @cached_property
    def graph_data(self) -> HeteroData:
        return HeteroData()
