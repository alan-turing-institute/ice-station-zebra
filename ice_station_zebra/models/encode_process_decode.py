import hydra
import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig

from ice_station_zebra.types import ZebraDataSpace

LightningBatch = list[torch.Tensor, torch.Tensor]


class EncodeProcessDecode(LightningModule):
    def __init__(
        self,
        *,
        input_spaces: list[ZebraDataSpace],
        output_space: ZebraDataSpace,
        latent_space: DictConfig,
        encoder: DictConfig,
        processor: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()

        # Construct the latent space
        latent_space_ = ZebraDataSpace(
            channels=latent_space["channels"],
            shape=latent_space["shape"],
        )

        # Add one encoder per dataset
        self.encoders = [
            hydra.utils.instantiate(
                dict(
                    {"input_space": input_space, "latent_space": latent_space_},
                    **encoder,
                ),
                _convert_="object",
            )
            for input_space in input_spaces
        ]

        # Add a processor
        n_latent_channels = latent_space_.channels * len(self.encoders)
        self.processor = hydra.utils.instantiate(
            dict({"n_latent_channels": n_latent_channels}, **processor),
            _convert_="object",
        )

        # Add a decoder
        self.decoder = hydra.utils.instantiate(
            dict(
                {"latent_space": latent_space_, "output_space": output_space}, **decoder
            ),
            _convert_="object",
        )

        # Construct a list of all modules
        self.model_list = nn.ModuleList(
            self.encoders + [self.processor] + [self.decoder]
        )

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
        print("latent_input_combined", latent_input_combined.shape)

        # Process in latent space: tensor with (batch_size, all_variables, latent_size, latent_size)
        latent_output = self.processor(latent_input_combined)
        print("latent_output", latent_output.shape)

        # Decode to output space: tensor with (batch_size, output_variables, output_height, output_width)
        output = self.decoder(latent_output)
        print("output", output.shape)

        # Return
        return output

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
        loss = torch.nn.functional.l1_loss(output, target)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model_list.parameters(), lr=0.1)
