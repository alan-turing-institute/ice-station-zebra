from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from ice_station_zebra.models.encoders import BaseEncoder
from ice_station_zebra.types import DataSpace, TensorNCHW, TensorNTCHW

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
        latent_space["name"] = "latent_space"
        latent_space_ = DataSpace.from_dict(latent_space)

        # Add one encoder per dataset
        # We store this as a list to ensure consistent ordering
        self.encoders: list[BaseEncoder] = [
            hydra.utils.instantiate(
                dict(**encoder)
                | {
                    "input_space": input_space,
                    "latent_space": latent_space_,
                    "n_history_steps": self.n_history_steps,
                }
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
            | {
                "latent_space": latent_space_,
                "n_forecast_steps": self.n_forecast_steps,
                "output_space": self.output_space,
            }
        )

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Forward step of the model

        - start with multiple [NTCHW] inputs each with shape [batch, n_history_steps, C_input_k, H_input_k, W_input_k]
        - encode inputs to [NCHW] latent space [batch, C_input_kprime, H_latent, W_latent]
        - concatenate inputs in [NCHW] latent space [batch, C_total, H_latent, W_latent]
        - process in latent space [NCHW] [batch, C_total, H_latent, W_latent]
        - decode back to [NTCHW] output space [batch, n_forecast_steps, C_output, H_output, W_output]
        """
        # Encode inputs into latent space: list of tensors with (batch_size, variables, latent_height, latent_width)
        latent_inputs: list[TensorNCHW] = [
            encoder(inputs[encoder.name]) for encoder in self.encoders
        ]

        # Combine in the variable dimension: tensor with (batch_size, all_variables, latent_height, latent_width)
        latent_input_combined: TensorNCHW = torch.cat(latent_inputs, dim=1)

        # Process in latent space: tensor with (batch_size, all_variables, latent_height, latent_width)
        latent_output: TensorNCHW = self.processor(latent_input_combined)

        # Decode to output space: tensor with (batch_size, output_variables, output_height, output_width)
        output: TensorNTCHW = self.decoder(latent_output)

        # Return
        return output
