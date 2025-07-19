from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from .zebra_model import ZebraModel

LightningBatch = list[torch.Tensor, torch.Tensor]


class EncodeProcessDecode(ZebraModel):
    def __init__(
        self,
        *,
        encoder: DictConfig,
        processor: DictConfig,
        decoder: DictConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Add one encoder per dataset
        self.encoders = [
            hydra.utils.instantiate(
                dict(
                    {"input_space": input_space, "latent_space": self.latent_space},
                    **encoder,
                ),
                _convert_="object",
            )
            for input_space in self.input_spaces
        ]

        # Add a processor
        n_latent_channels = self.latent_space.channels * len(self.encoders)
        self.processor = hydra.utils.instantiate(
            dict({"n_latent_channels": n_latent_channels}, **processor),
            _convert_="object",
        )

        # Add a decoder
        self.decoder = hydra.utils.instantiate(
            dict(
                {"latent_space": self.latent_space, "output_space": self.output_space},
                **decoder,
            ),
            _convert_="object",
        )

        # Register all modules that need to be trained
        self.model_list.extend(self.encoders)
        self.model_list.append(self.processor)
        self.model_list.append(self.decoder)

    def forward(self, inputs: LightningBatch) -> torch.Tensor:
        """Forward step of the model

        - encode inputs to latent space
        - combine inputs
        - process in latent space
        - decode back to output space
        """
        # Encode inputs into latent space: list of tensors with (batch_size, variables, latent_size, latent_size)
        latent_inputs = [
            encoder(input) for (input, encoder) in zip(inputs, self.encoders)
        ]

        # Combine in the variable dimension: tensor with (batch_size, all_variables, latent_size, latent_size)
        latent_input_combined = torch.cat(latent_inputs, dim=1)

        # Process in latent space: tensor with (batch_size, all_variables, latent_size, latent_size)
        latent_output = self.processor(latent_input_combined)

        # Decode to output space: tensor with (batch_size, output_variables, output_height, output_width)
        output = self.decoder(latent_output)

        # Return
        return output
