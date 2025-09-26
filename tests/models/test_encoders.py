import pytest
import torch

from ice_station_zebra.models.encoders import (
    BaseEncoder,
    CNNEncoder,
    NaiveLinearEncoder,
)
from ice_station_zebra.types import DataSpace


class TestEncoders:
    @pytest.mark.parametrize("test_batch_size", [1, 2, 5])
    @pytest.mark.parametrize("test_encoder_cls", ["CNNEncoder", "NaiveLinearEncoder"])
    @pytest.mark.parametrize("test_input_chw", [(4, 512, 512, 4), (1, 20, 200)])
    @pytest.mark.parametrize("test_latent_chw", [(128, 32, 32), (3, 40, 73)])
    @pytest.mark.parametrize("test_n_history_steps", [1, 3, 5])
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_encoder_cls: str,
        test_input_chw: tuple[int, int, int],
        test_latent_chw: tuple[int, int, int],
        test_n_history_steps: int,
    ) -> None:
        input_space = DataSpace(
            name="input", channels=test_input_chw[0], shape=test_input_chw[1:3]
        )
        latent_space = DataSpace(
            name="latent", channels=test_latent_chw[0], shape=test_latent_chw[1:3]
        )
        encoder: BaseEncoder = {
            "CNNEncoder": CNNEncoder(
                data_space_in=input_space,
                data_space_out=latent_space,
                n_history_steps=test_n_history_steps,
            ),
            "NaiveLinearEncoder": NaiveLinearEncoder(
                data_space_in=input_space,
                data_space_out=latent_space,
                n_history_steps=test_n_history_steps,
            ),
        }[test_encoder_cls]
        result: torch.Tensor = encoder(
            torch.randn(
                test_batch_size,
                test_n_history_steps,
                input_space.channels,
                *input_space.shape,
            )
        )
        assert result.shape == (
            test_batch_size,
            test_n_history_steps,
            encoder.n_output_channels,
            *latent_space.shape,
        )
