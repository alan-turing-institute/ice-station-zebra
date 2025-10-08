import pytest
import torch

from ice_station_zebra.models.processors import (
    BaseProcessor,
    NullProcessor,
    UNetProcessor,
)
from ice_station_zebra.types import DataSpace


@pytest.mark.parametrize("test_batch_size", [1, 2, 5])
@pytest.mark.parametrize("test_latent_chw", [(128, 32, 32), (3, 100, 200)])
@pytest.mark.parametrize("test_n_forecast_steps", [1, 2])
@pytest.mark.parametrize("test_n_history_steps", [1, 2])
class TestBaseProcessor:
    def test_rollout(
        self,
        test_batch_size: int,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        latent_space = DataSpace(
            name="latent", channels=test_latent_chw[0], shape=test_latent_chw[1:]
        )
        processor = BaseProcessor(
            data_space=latent_space,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
        )
        with pytest.raises(
            NotImplementedError,
            match="If you are using the default forward method, you must implement rollout.",
        ):
            processor.rollout(
                torch.randn(
                    test_batch_size,
                    test_n_history_steps,
                    *test_latent_chw,
                )
            )


@pytest.mark.parametrize("test_batch_size", [1, 2, 5])
@pytest.mark.parametrize("test_latent_chw", [(128, 32, 32), (3, 100, 200)])
@pytest.mark.parametrize("test_n_forecast_steps", [1, 2])
@pytest.mark.parametrize("test_n_history_steps", [1, 2])
class TestNullProcessor:
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        latent_space = DataSpace(
            name="latent", channels=test_latent_chw[0], shape=test_latent_chw[1:]
        )
        processor = NullProcessor(
            data_space=latent_space,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
        )
        result: torch.Tensor = processor.rollout(
            torch.randn(
                test_batch_size,
                test_n_history_steps,
                latent_space.channels,
                *latent_space.shape,
            )
        )
        assert result.shape == (
            test_batch_size,
            test_n_forecast_steps,
            latent_space.channels,
            *latent_space.shape,
        )


@pytest.mark.parametrize("test_batch_size", [1, 2, 5])
@pytest.mark.parametrize("test_kernel_size", [-1, 0, 1])
@pytest.mark.parametrize("test_latent_chw", [(128, 32, 32), (3, 100, 200)])
@pytest.mark.parametrize("test_n_forecast_steps", [1, 2])
@pytest.mark.parametrize("test_n_history_steps", [1, 2])
@pytest.mark.parametrize("test_start_out_channels", [-1, 7, 32])
class TestUNetProcessor:
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_kernel_size: int,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_start_out_channels: int,
    ) -> None:
        latent_space = DataSpace(
            name="latent", channels=test_latent_chw[0], shape=test_latent_chw[1:]
        )

        # Catch invalid filter size
        if test_kernel_size <= 0:
            with pytest.raises(ValueError, match="Kernel size must be greater than 0."):
                UNetProcessor(
                    data_space=latent_space,
                    kernel_size=test_kernel_size,
                    n_forecast_steps=test_n_forecast_steps,
                    n_history_steps=test_n_history_steps,
                    start_out_channels=test_start_out_channels,
                )
            return

        # Catch invalid start out channels
        if test_start_out_channels <= 0:
            with pytest.raises(
                ValueError, match="Start out channels must be greater than 0."
            ):
                UNetProcessor(
                    data_space=latent_space,
                    kernel_size=test_kernel_size,
                    n_forecast_steps=test_n_forecast_steps,
                    n_history_steps=test_n_history_steps,
                    start_out_channels=test_start_out_channels,
                )
            return

        processor = UNetProcessor(
            data_space=latent_space,
            kernel_size=test_kernel_size,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            start_out_channels=test_start_out_channels,
        )

        # Create a tensor with the expected shape
        x = torch.randn(
            test_batch_size,
            test_n_history_steps,
            latent_space.channels,
            *latent_space.shape,
        )
        _, _, _, height, width = x.shape

        # We will either catch an error or see a successful run
        if height % 16 or width % 16:
            msg = f"Latent space height and width must be divisible by 16 with a factor more than 1, got {height} and {width}."
            with pytest.raises(ValueError, match=msg):
                processor.rollout(x)
        else:
            result: torch.Tensor = processor.rollout(x)
            assert result.shape == (
                test_batch_size,
                test_n_forecast_steps,
                latent_space.channels,
                *latent_space.shape,
            )
