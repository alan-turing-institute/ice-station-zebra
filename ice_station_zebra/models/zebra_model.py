import torch
import torch.nn as nn
from lightning import LightningModule

from ice_station_zebra.types import ZebraDataSpace

from .decoders import LatentSpaceDecoder
from .encoders import LatentSpaceEncoder
from .processors import NullProcessor

LightningBatch = list[torch.Tensor, torch.Tensor]


class ZebraModelEncProcDec(LightningModule):
    def __init__(
        self,
        *,
        input_spaces: list[ZebraDataSpace],
        latent_space: ZebraDataSpace,
        output_space: ZebraDataSpace,
    ) -> None:
        super().__init__()
        self.model = None
        self.encoders = [
            LatentSpaceEncoder(input_space=input_space, latent_space=latent_space)
            for input_space in input_spaces
        ]
        latent_channels = latent_space.channels * len(self.encoders)
        self.processor = NullProcessor(latent_channels)
        self.decoder = LatentSpaceDecoder(
            latent_space=latent_space, output_space=output_space
        )
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

        # Process in latent space: tensor with (batch_size, all_variables, latent_size, latent_size)
        latent_output = self.processor(latent_input_combined)

        # Decode to output space: tensor with (batch_size, output_variables, output_height, output_width)
        output = self.decoder(latent_output)

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
