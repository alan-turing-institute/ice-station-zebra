from functools import cached_property

import lightning as L
from anemoi.training.train.train import AnemoiTrainer
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from .anemoi_adaptors import TrainerConfigAdaptor


class ZebraTrainer(AnemoiTrainer):
    def __init__(self, config: DictConfig) -> None:
        """Initialize the Zebra trainer."""
        super().__init__(TrainerConfigAdaptor(config))

    def train(self) -> None:
        pass

    @cached_property
    def graph_data(self) -> HeteroData:
        return HeteroData()

    @cached_property
    def model(self) -> L.LightningModule:
        """Provide the model instance."""
        return L.LightningModule()
