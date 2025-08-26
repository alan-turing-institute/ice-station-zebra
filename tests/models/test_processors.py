import pytest
import torch

from ice_station_zebra.models.processors import NullProcessor, UNetProcessor
from ice_station_zebra.types import DataSpace


@pytest.mark.parametrize("test_latent_shape", [(32, 32, 128), (100, 200, 3)])
@pytest.mark.parametrize("test_batch_size", [1, 2, 5])
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


@pytest.mark.parametrize("test_latent_shape", [(32, 32, 128), (100, 200, 3)])
@pytest.mark.parametrize("test_batch_size", [1, 2, 5])
@pytest.mark.parametrize("test_filter_size", [-1, 0, 1])
@pytest.mark.parametrize("test_start_out_channels", [-1, 7, 32])
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

        # Catch invalid filter size
        if test_filter_size <= 0:
            with pytest.raises(ValueError, match="Filter size must be greater than 0."):
                UNetProcessor(
                    filter_size=test_filter_size,
                    n_latent_channels=test_latent_shape[2],
                    start_out_channels=test_start_out_channels,
                )
            return

        # Catch invalid start out channels
        if test_start_out_channels <= 0:
            with pytest.raises(
                ValueError, match="Start out channels must be greater than 0."
            ):
                UNetProcessor(
                    filter_size=test_filter_size,
                    n_latent_channels=test_latent_shape[2],
                    start_out_channels=test_start_out_channels,
                )
            return

        processor = UNetProcessor(
            filter_size=test_filter_size,
            n_latent_channels=test_latent_shape[2],
            start_out_channels=test_start_out_channels,
        )

        # Create a tensor with the expected shape
        x = torch.randn(test_batch_size, latent_space.channels, *latent_space.shape)
        _, _, height, width = x.shape

        # We will either catch an error or see a successful run
        if height % 16 or width % 16:
            msg = f"Latent space height and width must be divisible by 16 with a factor more than 1, got {height} and {width}."
            with pytest.raises(ValueError, match=msg):
                processor(x)
        else:
            result: torch.Tensor = processor(x)
            assert result.shape == (
                test_batch_size,
                latent_space.channels,
                *latent_space.shape,
            )
