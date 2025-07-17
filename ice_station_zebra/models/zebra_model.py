from abc import ABC, abstractmethod

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
        input_spaces: list[ZebraDataSpace],
        output_space: ZebraDataSpace,
        latent_space: DictConfig,
        optimizer: DictConfig,
    ) -> None:
        super().__init__()

        # Construct the input and output spaces
        self.input_spaces = input_spaces
        self.output_space = output_space

        # Construct the latent space
        self.latent_space = ZebraDataSpace(
            channels=latent_space["channels"],
            shape=latent_space["shape"],
        )

        # Initialise an empty module list
        self.model_list = nn.ModuleList()

        # Store the optimizer config
        self.optimizer_cfg = optimizer

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
