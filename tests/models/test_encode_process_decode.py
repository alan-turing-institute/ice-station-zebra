import pytest
import torch
from omegaconf import DictConfig

from ice_station_zebra.models import EncodeProcessDecode


@pytest.mark.parametrize("test_n_forecast_steps", [1, 2, 5])
@pytest.mark.parametrize("test_n_history_steps", [1, 2, 5])
class TestEncodeProcessDecode:
    def test_init(
        self,
        cfg_decoder: DictConfig,
        cfg_encoder: DictConfig,
        cfg_processor: DictConfig,
        cfg_latent_space: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        model = EncodeProcessDecode(
            name="encode-null-decode",
            encoder=cfg_encoder,
            processor=cfg_processor,
            decoder=cfg_decoder,
            latent_space=cfg_latent_space,
            input_spaces=[cfg_input_space],
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
        )

        assert model.name == "encode-null-decode"
        assert model.input_spaces[0].channels == cfg_input_space["channels"]
        assert model.input_spaces[0].name == cfg_input_space["name"]
        assert model.input_spaces[0].shape == cfg_input_space["shape"]
        assert model.n_forecast_steps == test_n_forecast_steps
        assert model.n_history_steps == test_n_history_steps
        assert model.output_space.channels == cfg_output_space["channels"]
        assert model.output_space.name == cfg_output_space["name"]
        assert model.output_space.shape == cfg_output_space["shape"]

    @pytest.mark.parametrize("test_batch_size", [0, 1, 2])
    def test_forward(
        self,
        cfg_decoder: DictConfig,
        cfg_encoder: DictConfig,
        cfg_processor: DictConfig,
        cfg_latent_space: DictConfig,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        test_batch_size: int,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        model = EncodeProcessDecode(
            name="encode-null-decode",
            encoder=cfg_encoder,
            processor=cfg_processor,
            decoder=cfg_decoder,
            latent_space=cfg_latent_space,
            input_spaces=[cfg_input_space],
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
        )
        result: torch.Tensor = model(
            {
                cfg_input_space["name"]: torch.randn(
                    test_batch_size,
                    test_n_history_steps,
                    cfg_input_space["channels"],
                    cfg_input_space["shape"][0],
                    cfg_input_space["shape"][1],
                )
            }
        )
        assert result.shape == (
            test_batch_size,
            test_n_forecast_steps,
            cfg_output_space["channels"],
            cfg_output_space["shape"][0],
            cfg_output_space["shape"][1],
        )
