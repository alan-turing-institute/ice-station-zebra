from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor

from ice_station_zebra.types import CombinedTensorBatch, DataSpace
from ice_station_zebra.models.encoders import BaseEncoder

from .zebra_model import ZebraModel


class EncodeProcessDecode(ZebraModel):
    def __init__(
        self,
        *,
        encoder: DictConfig,
        processor: DictConfig,
        decoder: DictConfig,
        latent_space: DictConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Construct the latent space
        latent_space_ = DataSpace.from_dict(latent_space | {"name": "latent_space"})

        # Add one encoder per dataset
        # We store this as a list to ensure consistent ordering
        self.encoders: list[BaseEncoder] = [
            hydra.utils.instantiate(
                dict(**encoder)
                | {"input_space": input_space, "latent_space": latent_space_}
            )
            for input_space in self.input_spaces
        ]
        # We have to explicitly register each encoder as list[Module] will not be
        # automatically picked up by PyTorch
        for idx, module in enumerate(self.encoders):
            self.add_module(f"encoder_{idx}", module)

        # Add a processor
        self.processor = hydra.utils.instantiate(
            dict(**processor)
            | {"n_latent_channels": latent_space_.channels * len(self.encoders)}
        )

        # Add a decoder
        self.decoder = hydra.utils.instantiate(
            dict(**decoder)
            | {"latent_space": latent_space_, "output_space": self.output_space}
        )

    def forward(self, inputs: CombinedTensorBatch) -> torch.Tensor:
        """Forward step of the model

        - start with multiple inputs each with shape [N, C_input_k, H_input_k, W_input_k]
        - encode inputs to latent space [N, C_input, H_latent, W_latent]
        - concatenate inputs in latent space [N, C_total, H_latent, W_latent]
        - process in latent space [N, C_total, H_latent, W_latent]
        - decode back to output space [N, C_output, H_output, W_output]
        """
        # Encode inputs into latent space: list of tensors with (batch_size, variables, latent_height, latent_width)
        latent_inputs: list[Tensor] = [
            encoder(inputs[encoder.name]) for encoder in self.encoders
        ]

        # Combine in the variable dimension: tensor with (batch_size, all_variables, latent_height, latent_width)
        latent_input_combined = torch.cat(latent_inputs, dim=1)

        # Process in latent space: tensor with (batch_size, all_variables, latent_height, latent_width)
        latent_output: Tensor = self.processor(latent_input_combined)

        # Decode to output space: tensor with (batch_size, output_variables, output_height, output_width)
        output: Tensor = self.decoder(latent_output)

        # Return
        return output
