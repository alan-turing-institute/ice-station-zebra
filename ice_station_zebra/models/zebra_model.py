import itertools
from abc import ABC, abstractmethod

import hydra
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig

from ice_station_zebra.types import DataSpace, ModelTestOutput, TensorNTCHW


class ZebraModel(LightningModule, ABC):
    """A base class for all models used in the Ice Station Zebra project."""

    def __init__(
        self,
        *,
        name: str,
        input_spaces: list[DictConfig],
        n_forecast_steps: int,
        n_history_steps: int,
        output_space: DictConfig,
        optimizer: DictConfig,
    ) -> None:
        super().__init__()

        # Save model name
        self.name = name

        # Save history and forecast steps
        self.n_forecast_steps = n_forecast_steps
        self.n_history_steps = n_history_steps

        # Construct the input and output spaces
        self.input_spaces = [DataSpace.from_dict(space) for space in input_spaces]
        self.output_space = DataSpace.from_dict(output_space)

        # Store the optimizer config
        self.optimizer_cfg = optimizer

        # Save all of the arguments to __init__ as hyperparameters
        # This will also save the parameters of whichever child class is used
        # Note that W&B will log all hyperparameters
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model

        - start with multiple [NTCHW] inputs each with shape [batch, n_history_steps, C_input_k, H_input_k, W_input_k]
        - return a single [NTCHW] output [batch, n_forecast_steps, C_output, H_output, W_output]
        """

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Construct the optimizer from the config"""
        return hydra.utils.instantiate(
            dict(**self.optimizer_cfg)
            | {
                "params": itertools.chain(
                    *[module.parameters() for module in self.children()]
                )
            }
        )

    def loss(self, prediction: TensorNTCHW, target: TensorNTCHW) -> torch.Tensor:
        """Calculate the loss given a prediction and target"""
        return torch.nn.functional.l1_loss(prediction, target)

    def test_step(
        self, batch: dict[str, TensorNTCHW], batch_idx: int
    ) -> ModelTestOutput:
        """Run the test step, in PyTorch eval model (i.e. no gradients)

        - Separate the batch into inputs and target
        - Run inputs through the model
        - Return the prediction, target and loss

        Args:
            batch: Dictionary mapping dataset name to its contents. There is one entry
                   for each input dataset and one for the target. Each of these is a
                   TensorNTCHW with (batch_size, n_history_steps, C, H, W).

        Returns:
            A ModelTestOutput containing the prediction, target and loss for the batch.
        """
        target = batch.pop("target")
        prediction = self(batch)
        loss = self.loss(prediction, target)
        return ModelTestOutput(prediction, target, loss)

    def training_step(
        self, batch: dict[str, TensorNTCHW], batch_idx: int
    ) -> torch.Tensor:
        """Run the training step

        - Separate the batch into inputs and target
        - Run inputs through the model
        - Calculate the loss wrt. the target

        Args:
            batch: Dictionary mapping dataset name to its contents. There is one entry
                   for each input dataset and one for the target. Each of these is a
                   TensorNTCHW with (batch_size, n_history_steps, C, H, W).

        Returns:
            A Tensor containing the loss for the batch.
        """
        target = batch.pop("target")
        prediction = self(batch)
        return self.loss(prediction, target)

    def validation_step(
        self, batch: dict[str, TensorNTCHW], batch_idx: int
    ) -> torch.Tensor:
        """Run the validation step

        A batch contains one tensor for each input dataset and one for the target
        These are [NTCHW] tensors with (batch_size, n_history_steps, C, H, W)

        - Separate the batch into inputs and target
        - Run inputs through the model
        - Calculate and log the loss wrt. the target

        Args:
            batch: Dictionary mapping dataset name to its contents. There is one entry
                   for each input dataset and one for the target. Each of these is a
                   TensorNTCHW with (batch_size, n_history_steps, C, H, W).

        Returns:
            A Tensor containing the loss for the batch.
        """
        target = batch.pop("target")
        prediction = self(batch)
        loss = self.loss(prediction, target)
        self.log("validation_loss", loss)
        return loss
