from typing import Any

from torch import nn
from torch.nn.functional import sigmoid

from icenet_mp.models.common import CommonConvBlock, Permute
from icenet_mp.types import TensorNCHW

from .base_decoder import BaseDecoder


class PiecewiseDecoder(BaseDecoder):
    """Piecewise decoder that combines data patches from a latent space to build the output space.

    - 1 convolutional block to set the required number of channels
    - n_blocks of constant-size convolutional blocks
    - Combine patches into output of size output_height x output_width

    Latent space:
        TensorNTCHW with (batch_size, n_forecast_steps, latent_channels, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, output_channels, output_height, output_width)
    """

    def __init__(
        self, *, bounded: bool = False, n_blocks: int = 0, **kwargs: Any
    ) -> None:
        """Initialise a PiecewiseDecoder."""
        super().__init__(**kwargs)

        # specify whether the output is bounded between 0 and 1
        self.bounded = bounded

        # Calculate the number of patches required to cover the output space and the corresponding number of channels
        n_patches = (
            (
                self.data_space_out.shape[0]
                + 2 * self.data_space_in.shape[0]
                - 1 * (self.data_space_in.shape[0] - 1)
                - 1
            )
            // self.data_space_in.shape[0]
            + 1
        ) * (
            (
                self.data_space_out.shape[1]
                + 2 * self.data_space_in.shape[1]
                - 1 * (self.data_space_in.shape[1] - 1)
                - 1
            )
            // self.data_space_in.shape[1]
            + 1
        )
        input_channels_required = self.data_space_out.channels * n_patches

        # Construct list of layers
        layers: list[nn.Module] = []

        # Add a convolutional block to get the required number of channels
        layers.append(
            CommonConvBlock(
                self.data_space_in.channels,
                input_channels_required,
                kernel_size=3,
                activation="SiLU",
                n_subblocks=n_blocks + 1,
            ),
        )

        # Unflatten the channel dimension to extract the patches: [N, n_patches, C, patch_h, patch_w]
        layers.append(nn.Unflatten(1, (n_patches, -1)))

        # Flatten the patch dimensions: [N, n_patches, C * patch_area]
        layers.append(nn.Flatten(2, 4))

        # Permute dimensions: [N, C * patch_area, n_patches]
        layers.append(Permute((0, 2, 1)))

        # Fold patches into the output shape: [N, C, output_h, output_w]
        layers.append(
            nn.Fold(
                output_size=self.data_space_out.shape,
                kernel_size=self.data_space_in.shape,
                stride=self.data_space_in.shape,
                padding=self.data_space_in.shape,
            )
        )

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: decode latent space into output space by combining patches.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, output_channels, output_height, output_width)

        """
        if self.bounded:
            return sigmoid(self.model(x))
        return self.model(x)
