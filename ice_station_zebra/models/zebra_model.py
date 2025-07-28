from abc import ABC, abstractmethod

import hydra
import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig
from torch.optim import Optimizer

from ice_station_zebra.types import DataSpace, LightningBatch


class ZebraModel(LightningModule, ABC):
    def __init__(
        self,
        *,
        name: str,
        input_spaces: list[DictConfig],
        output_space: DictConfig,
        optimizer: DictConfig,
    ) -> None:
        super().__init__()

        # Save model name
        self.name = name

        # Construct the input and output spaces
        self.input_spaces = [DataSpace.from_dict(space) for space in input_spaces]
        self.output_space = DataSpace.from_dict(output_space)

        # Initialise an empty module list
        self.model_list = nn.ModuleList()

        # Store the optimizer config
        self.optimizer_cfg = optimizer

        # Save all of the arguments to __init__ as hyperparameters
        # This will also save the parameters of whichever child class is used
        # Note that W&B will log all hyperparameters
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, inputs: LightningBatch) -> torch.Tensor:
        """Forward step of the model"""

    def configure_optimizers(self) -> Optimizer:
        """Construct the optimizer from the config"""
        return hydra.utils.instantiate(
            dict(**self.optimizer_cfg) | {"params": self.model_list.parameters()}
        )

    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.l1_loss(output, target)

    def test_step(
        self, batch: LightningBatch, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Run the test step, in PyTorch eval model (i.e. no gradients)

        A batch contains one tensor for each input dataset followed by one for the target
        The shape of each of these tensors is (batch_size; variables; ensembles; position)

        - Separate the batch into inputs and target
        - Run inputs through the model
        - Return the output, target and loss
        """
        inputs, target = batch[:-1], batch[-1]
        output = self(inputs)
        loss = self.loss(output, target)
        return {"output": output, "target": target, "loss": loss}

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
