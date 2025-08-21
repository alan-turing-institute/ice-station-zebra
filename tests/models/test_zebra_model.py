from typing import Any

import pytest
import torch
from omegaconf import DictConfig

from ice_station_zebra.models import ZebraModel
from ice_station_zebra.types import TensorNTCHW


class DummyModel(ZebraModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise a dummy model for testing purposes."""
        super().__init__(*args, **kwargs)
        self.model = torch.nn.Linear(1, 1)

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Dummy forward method."""
        return self.model(next(iter(inputs.values())))


class TestZebraModel:
    @pytest.mark.parametrize("test_input_shape", [(512, 512, 4), (10, 20, 1)])
    @pytest.mark.parametrize("test_output_shape", [(432, 432, 1), (10, 20, 19)])
    @pytest.mark.parametrize("test_n_forecast_steps", [0, 1, 2, 5])
    @pytest.mark.parametrize("test_n_history_steps", [0, 1, 2, 5])
    def test_init(
        self,
        test_input_shape: tuple[int, int, int],
        test_output_shape: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        input_space = DictConfig(
            {
                "channels": test_input_shape[2],
                "name": "input",
                "shape": test_input_shape[0:2],
            }
        )
        output_space = DictConfig(
            {
                "channels": test_output_shape[2],
                "name": "target",
                "shape": test_output_shape[0:2],
            }
        )

        # Catch invalid n_forecast_steps
        if test_n_forecast_steps <= 0:
            with pytest.raises(
                ValueError, match="Number of forecast steps must be greater than 0."
            ):
                DummyModel(
                    name="dummy",
                    input_spaces=[input_space],
                    n_forecast_steps=test_n_forecast_steps,
                    n_history_steps=test_n_history_steps,
                    output_space=output_space,
                    optimizer=DictConfig({}),
                )
            return

        # Catch invalid n_history_steps
        if test_n_history_steps <= 0:
            with pytest.raises(
                ValueError, match="Number of history steps must be greater than 0."
            ):
                DummyModel(
                    name="dummy",
                    input_spaces=[input_space],
                    n_forecast_steps=test_n_forecast_steps,
                    n_history_steps=test_n_history_steps,
                    output_space=output_space,
                    optimizer=DictConfig({}),
                )
            return

        model = DummyModel(
            name="dummy",
            input_spaces=[input_space],
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=output_space,
            optimizer=DictConfig({}),
        )

        assert model.name == "dummy"
        assert model.input_spaces[0].channels == test_input_shape[2]
        assert model.input_spaces[0].name == "input"
        assert model.input_spaces[0].shape == test_input_shape[0:2]
        assert model.n_forecast_steps == test_n_forecast_steps
        assert model.n_history_steps == test_n_history_steps
        assert model.output_space.channels == test_output_shape[2]
        assert model.output_space.name == "target"
        assert model.output_space.shape == test_output_shape[0:2]

    def test_loss(
        self, cfg_input_space: DictConfig, cfg_output_space: DictConfig
    ) -> None:
        model = DummyModel(
            name="dummy",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
        )
        # Test loss
        prediction = torch.zeros(1, 1, 1, 1)
        target = torch.ones(1, 1, 1, 1)
        assert model.loss(prediction, target) == torch.tensor(1.0)

    def test_optimizer(
        self,
        cfg_input_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_output_space: DictConfig,
    ) -> None:
        model = DummyModel(
            name="dummy",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=cfg_optimizer,
        )
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 5e-4
