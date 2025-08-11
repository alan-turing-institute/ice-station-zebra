import pytest
import numpy as np

from ice_station_zebra.data.lightning.zebra_dataset import ZebraDataset
from ice_station_zebra.types import DataSpace


class MockAnemoiDataset:
    def __init__(self):
        self.name = "mock_dataset"
        self.start_date = np.datetime64("2020-01-01")
        self.end_date = np.datetime64("2020-12-31")
        self.field_shape = (5, 5)
        self.shape = [1, 10]  # (time, channels)

    def __len__(self):
        return 10


class TestZebraDataset:
    def test_init(self) -> None:
        dataset = ZebraDataset(
            name="test_dataset",
            input_files=[],
        )
        assert dataset.name == "test_dataset"

    def test_dataset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ZebraDataset, "dataset", property(lambda _: MockAnemoiDataset())
        )
        dataset = ZebraDataset(
            name="mock_dataset",
            input_files=[],
        )
        assert dataset.end_date == np.datetime64("2020-12-31")
        assert dataset.name == "mock_dataset"
        assert isinstance(dataset.space, DataSpace)
        assert dataset.space.channels == 10
        assert dataset.space.shape == (5, 5)
        assert dataset.start_date == np.datetime64("2020-01-01")
        assert len(dataset) == 10
