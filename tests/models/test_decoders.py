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
    @pytest.mark.parametrize("test_latent_chw", [(128, 32, 32), (2, 200, 100)])
    @pytest.mark.parametrize("test_n_forecast_steps", [1, 3, 5])
    @pytest.mark.parametrize("test_output_chw", [(4, 256, 256), (1, 100, 200)])
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_decoder_cls: str,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_output_chw: tuple[int, int, int],
    ) -> None:
        latent_space = DataSpace(
            name="latent", channels=test_latent_chw[0], shape=test_latent_chw[1:]
        )
        output_space = DataSpace(
            name="output", channels=test_output_chw[0], shape=test_output_chw[1:]
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


class TestCNNDecoder:
    def test_latent_shape_errors(self) -> None:
        test_n_forecast_steps = 1
        latent_space = DataSpace(name="latent", shape=(32, 32), channels=3)
        output_space = DataSpace(name="output", shape=(256, 256), channels=4)
        with pytest.raises(
            ValueError,
            match="The number of input channels 3 must be divisible by 2. Without this, it is not possible to apply 1 convolutions.",
        ):
            CNNDecoder(
                data_space_in=latent_space,
                data_space_out=output_space,
                n_forecast_steps=test_n_forecast_steps,
                n_layers=1,
            )
