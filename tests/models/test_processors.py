import pytest
import torch

from ice_station_zebra.models.processors import NullProcessor
from ice_station_zebra.types import DataSpace


@pytest.mark.parametrize(
    "test_latent_shape", [(32, 32, 128), (100, 200, 3), (10, 10, 100)]
)
@pytest.mark.parametrize("test_batch_size", [0, 1, 2])
class TestNullProcessor:
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_latent_shape: tuple[int, int, int],
    ) -> None:
        latent_space = DataSpace(
            name="latent", shape=test_latent_shape[0:2], channels=test_latent_shape[2]
        )
        processor = NullProcessor(n_latent_channels=latent_space.channels)
        result: torch.Tensor = processor(
            torch.randn(
                test_batch_size,
                latent_space.channels,
                *latent_space.shape,
            )
        )
        assert result.shape == (
            test_batch_size,
            latent_space.channels,
            *latent_space.shape,
        )
