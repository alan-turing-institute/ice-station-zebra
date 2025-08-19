import pytest
import torch

from ice_station_zebra.models.processors import NullProcessor, UNetProcessor
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


@pytest.mark.parametrize(
    "test_latent_shape", [(128, 128, 20), (256, 512, 3), (64, 64, 100)]
)
@pytest.mark.parametrize("test_batch_size", [1, 2, 3])
@pytest.mark.parametrize("test_filter_size", [1, 3])
@pytest.mark.parametrize("test_start_out_channels", [32, 64])
class TestUNetProcessor:
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_filter_size: int,
        test_latent_shape: tuple[int, int, int],
        test_start_out_channels: int,
    ) -> None:
        latent_space = DataSpace(
            name="latent", shape=test_latent_shape[0:2], channels=test_latent_shape[2]
        )
        processor = UNetProcessor(
            filter_size=test_filter_size,
            n_latent_channels=test_latent_shape[2],
            start_out_channels=test_start_out_channels,
        )
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
