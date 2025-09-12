import pytest
import torch

from ice_station_zebra.models.encoders import NaiveLatentSpaceEncoder
from ice_station_zebra.types import DataSpace


@pytest.mark.parametrize("test_input_shape", [(512, 512, 4), (100, 20, 1)])
@pytest.mark.parametrize("test_latent_shape", [(32, 32, 128), (100, 200, 3)])
@pytest.mark.parametrize("test_batch_size", [1, 2, 5])
@pytest.mark.parametrize("test_n_history_steps", [1, 3, 5])
class TestNaiveLatentSpaceEncoder:
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_input_shape: tuple[int, int, int],
        test_latent_shape: tuple[int, int, int],
        test_n_history_steps: int,
    ) -> None:
        input_space = DataSpace(
            name="input", shape=test_input_shape[0:2], channels=test_input_shape[2]
        )
        latent_space = DataSpace(
            name="latent", shape=test_latent_shape[0:2], channels=test_latent_shape[2]
        )
        encoder = NaiveLatentSpaceEncoder(
            input_space=input_space,
            latent_space=latent_space,
            n_history_steps=test_n_history_steps,
        )
        result = encoder(
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
            latent_space.channels,
            *latent_space.shape,
        )
