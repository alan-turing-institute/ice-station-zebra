import pytest
import torch

from ice_station_zebra.models.decoders import (
    BaseDecoder,
    CNNDecoder,
    NaiveLinearDecoder,
)
from ice_station_zebra.types import DataSpace


class TestDecoders:
    @pytest.mark.parametrize("test_batch_size", [1, 2, 5])
    @pytest.mark.parametrize("test_decoder_cls", ["CNNDecoder", "NaiveLinearDecoder"])
    @pytest.mark.parametrize("test_latent_chw", [(128, 32, 32), (3, 200, 100)])
    @pytest.mark.parametrize("test_n_forecast_steps", [1, 3, 5])
    @pytest.mark.parametrize("test_output_chw", [(4, 256, 256), (1, 100, 200)])
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_decoder_cls: str,
        test_latent_chw: tuple[int, int, int],
        test_output_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
    ) -> None:
        latent_space = DataSpace(
            name="latent", channels=test_latent_chw[0], shape=test_latent_chw[1:3]
        )
        output_space = DataSpace(
            name="output", channels=test_output_chw[0], shape=test_output_chw[1:3]
        )
        decoder: BaseDecoder = {
            "CNNDecoder": CNNDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_forecast_steps=test_n_forecast_steps,
                n_layers=1,
            ),
            "NaiveLinearDecoder": NaiveLinearDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_forecast_steps=test_n_forecast_steps,
            ),
        }[test_decoder_cls]
        result: torch.Tensor = decoder(
            torch.randn(
                test_batch_size,
                test_n_forecast_steps,
                latent_space.channels,
                *latent_space.shape,
            )
        )
        assert result.shape == (
            test_batch_size,
            test_n_forecast_steps,
            output_space.channels,
            *output_space.shape,
        )
