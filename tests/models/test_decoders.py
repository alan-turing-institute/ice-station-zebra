import pytest
import torch

from ice_station_zebra.models.decoders import NaiveLatentSpaceDecoder
from ice_station_zebra.types import DataSpace


@pytest.mark.parametrize("test_latent_shape", [(32, 32, 128), (100, 200, 3)])
@pytest.mark.parametrize("test_output_shape", [(512, 512, 4), (1000, 200, 1)])
@pytest.mark.parametrize("test_batch_size", [1, 2, 5])
@pytest.mark.parametrize("test_n_forecast_steps", [1, 3, 5])
class TestNaiveLatentSpaceDecoder:
    def test_forward_shape(
        self,
        test_batch_size: int,
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
        decoder = NaiveLatentSpaceDecoder(
            latent_space=latent_space,
            n_forecast_steps=test_n_forecast_steps,
            output_space=output_space,
        )
        result = decoder(
            torch.randn(
                test_batch_size,
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
