import pytest
from omegaconf import DictConfig

from ice_station_zebra.models import ZebraModel
from ice_station_zebra.types import TensorNTCHW


@pytest.mark.parametrize(
    "test_input_shape", [(512, 512, 4), (1000, 200, 1), (1, 1, 20)]
)
@pytest.mark.parametrize("test_output_shape", [(432, 432, 1), (10, 20, 19), (1, 1, 1)])
@pytest.mark.parametrize("test_n_forecast_steps", [0, 1, 2, 5])
@pytest.mark.parametrize("test_n_history_steps", [0, 1, 2, 5])
class TestZebraModel:
    def test_init(
        self,
        test_input_shape: tuple[int, int, int],
        test_output_shape: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        class DummyModel(ZebraModel):
            def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
                return next(iter(inputs.values()))

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
