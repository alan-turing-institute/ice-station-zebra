from abc import ABC, abstractmethod
from typing import Any

import hydra
import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig
from torch.optim import Optimizer

from ice_station_zebra.types import ZebraDataSpace

LightningBatch = list[torch.Tensor, torch.Tensor]


class ZebraModel(LightningModule, ABC):
    def __init__(
        self,
        *,
        name: str,
        input_spaces: list[ZebraDataSpace],
        output_space: ZebraDataSpace,
        optimizer: DictConfig,
    ) -> None:
        super().__init__()

        # Save model name
        self.name = name

        # Construct the input and output spaces
        self.input_spaces = input_spaces
        self.output_space = output_space

        # Initialise an empty module list
        self.model_list = nn.ModuleList()

        # Store the optimizer config
        self.optimizer_cfg = optimizer

    def register_hyperparameters(self, hyperparameters: dict[str, Any] = {}) -> None:
        """Save the hyper-parameters of the model

        All children of this class should call this method at the end of __init__
        """
        hparams = {
            "input_spaces": {
                "channels": [input_space.channels for input_space in self.input_spaces],
                "shape": [input_space.shape for input_space in self.input_spaces],
            },
            "model": {
                "components": [module._get_name() for module in self.model_list],
                "name": self.name,
            },
            "optimizer": {
                "type": self.optimizer_cfg.get("_target_", None),
                "lr": self.optimizer_cfg.get("lr", None),
            },
            "output_space": {
                "channels": self.output_space.channels,
                "shape": self.output_space.shape,
            },
        }
        hparams.update(hyperparameters)
        self.save_hyperparameters(hparams)

    @abstractmethod
    def forward(self, inputs: LightningBatch) -> torch.Tensor:
        """Forward step of the model"""

    def configure_optimizers(self) -> Optimizer:
        """Construct the optimizer from the config"""
        return hydra.utils.instantiate(
            dict(
                {"params": self.model_list.parameters()},
                **self.optimizer_cfg,
            ),
        )

    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.l1_loss(output, target)

    def training_step(self, batch: LightningBatch, batch_idx: int) -> torch.Tensor:
        """Run the training step

        A batch contains one tensor for each input dataset followed by one for the target
        The shape of each of these tensors is (batch_size; variables; ensembles; position)

        - Separate the batch into inputs and target
        - Run inputs through the model
        - Calculate the loss wrt. the target
        """
        inputs, target = batch[:-1], batch[-1]
        output = self(inputs)
        return self.loss(output, target)

    def validation_step(self, batch: LightningBatch, batch_idx: int) -> torch.Tensor:
        """Run the validation step

        A batch contains one tensor for each input dataset followed by one for the target
        The shape of each of these tensors is (batch_size; variables; ensembles; position)

        - Separate the batch into inputs and target
        - Run inputs through the model
        - Calculate the loss wrt. the target
        """
        inputs, target = batch[:-1], batch[-1]
        output = self(inputs)
        loss = self.loss(output, target)
        self.log("validation_loss", loss)
        return loss
