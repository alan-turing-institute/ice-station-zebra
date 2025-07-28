from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from ice_station_zebra.types import DataSpace

from .zebra_model import ZebraModel

LightningBatch = list[torch.Tensor, torch.Tensor]


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
        latent_space_ = DataSpace.from_dict(latent_space)

        # Add one encoder per dataset
        self.encoders = [
            hydra.utils.instantiate(
                dict(
                    {"input_space": input_space, "latent_space": latent_space_},
                    **encoder,
                ),
            )
            for input_space in self.input_spaces
        ]

        # Add a processor
        self.processor = hydra.utils.instantiate(
            dict(
                {"n_latent_channels": latent_space_.channels * len(self.encoders)},
                **processor,
            ),
        )

        # Add a decoder
        self.decoder = hydra.utils.instantiate(
            dict(
                {"latent_space": latent_space_, "output_space": self.output_space},
                **decoder,
            ),
        )

        # Register all modules that need to be trained
        self.model_list.extend(self.encoders)
        self.model_list.append(self.processor)
        self.model_list.append(self.decoder)

    def forward(self, inputs: LightningBatch) -> torch.Tensor:
        """Forward step of the model

        - start with multiple inputs each with shape [N, C_input_k, H_input_k, W_input_k]
        - encode inputs to latent space [N, C_input, H_latent, W_latent]
        - concatenate inputs in latent space [N, C_total, H_latent, W_latent]
        - process in latent space [N, C_total, H_latent, W_latent]
        - decode back to output space [N, C_output, H_output, W_output]
        """
        # Encode inputs into latent space: list of tensors with (batch_size, variables, latent_height, latent_width)
        latent_inputs = [
            encoder(input) for (input, encoder) in zip(inputs, self.encoders)
        ]

        # Combine in the variable dimension: tensor with (batch_size, all_variables, latent_height, latent_width)
        latent_input_combined = torch.cat(latent_inputs, dim=1)

        # Process in latent space: tensor with (batch_size, all_variables, latent_height, latent_width)
        latent_output = self.processor(latent_input_combined)

        # Decode to output space: tensor with (batch_size, output_variables, output_height, output_width)
        output = self.decoder(latent_output)

        # Return
        return output
