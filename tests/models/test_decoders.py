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
    @pytest.mark.parametrize("test_latent_shape", [(32, 32, 128), (200, 100, 3)])
    @pytest.mark.parametrize("test_n_forecast_steps", [1, 3, 5])
    @pytest.mark.parametrize("test_output_shape", [(256, 256, 4), (100, 200, 1)])
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_decoder_cls: str,
        test_latent_shape: tuple[int, int, int],
        test_output_shape: tuple[int, int, int],
        test_n_forecast_steps: int,
    ) -> None:
        latent_space = DataSpace(
            name="latent", shape=test_latent_shape[0:2], channels=test_latent_shape[2]
        )
        output_space = DataSpace(
            name="output", shape=test_output_shape[0:2], channels=test_output_shape[2]
        )
        decoder: BaseDecoder = {
            "CNNDecoder": CNNDecoder(
                n_forecast_steps=test_n_forecast_steps,
                n_latent_channels_total=latent_space.channels,
                n_layers=1,
                output_space=output_space,
            ),
            "NaiveLinearDecoder": NaiveLinearDecoder(
                n_forecast_steps=test_n_forecast_steps,
                n_latent_channels_total=latent_space.channels,
                output_space=output_space,
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
