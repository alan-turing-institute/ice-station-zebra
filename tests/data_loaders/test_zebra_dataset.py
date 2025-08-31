from pathlib import Path

import numpy as np
import pytest

from ice_station_zebra.data_loaders.zebra_dataset import ZebraDataset
from ice_station_zebra.types import DataSpace


class TestZebraDataset:
    dates_str = ("2020-01-01", "2020-01-02", "2020-01-03")
    dates_np = tuple(np.datetime64(s) for s in dates_str)

    def test_dataset_name(self) -> None:
        dataset = ZebraDataset(
            name="test_dataset",
            input_files=[],
        )
        assert dataset.name == "test_dataset"

    def test_dataset_dates(self, mock_dataset: Path) -> None:
        dataset = ZebraDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
        )
        # Test dates
        assert all(date in dataset.dataset.dates for date in self.dates_np)

    def test_dataset_end_date(self, mock_dataset: Path) -> None:
        dataset = ZebraDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
            end=self.dates_str[1],
        )
        assert dataset.start_date == self.dates_np[0]
        assert dataset.end_date == self.dates_np[1]

    def test_dataset_getitem(self, mock_dataset: Path) -> None:
        dataset = ZebraDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
        )
        # Check return type and shape
        data_array = dataset[0]
        assert isinstance(data_array, np.ndarray)
        assert data_array.shape == (1, 2, 2)
        # Check exception for out of range
        with pytest.raises(IndexError) as excinfo:
            dataset[5]
        assert "list index out of range" in str(excinfo.value)

    def test_dataset_get_tchw(self, mock_dataset: Path) -> None:
        dataset = ZebraDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
        )
        # Check return type and shape
        data_array = dataset.get_tchw(self.dates_np)
        assert isinstance(data_array, np.ndarray)
        assert data_array.shape == (3, 1, 2, 2)
        # Check exception for out of range
        with pytest.raises(
            ValueError, match="Date 1970-01-01T00:00:00 not found in the dataset"
        ):
            dataset.get_tchw([np.datetime64("1970-01-01"), np.datetime64("1970-01-02")])

    def test_dataset_index_from_date(self, mock_dataset: Path) -> None:
        dataset = ZebraDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
        )
        # Check type
        assert dataset.index_from_date(self.dates_np[0]) == 0
        # Check exception for out of range
        with pytest.raises(
            ValueError, match="Date 1970-01-01T00:00:00 not found in the dataset"
        ):
            dataset.index_from_date(np.datetime64("1970-01-01"))

    def test_dataset_len(self, mock_dataset: Path) -> None:
        dataset = ZebraDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
        )
        assert len(dataset) == 3

    def test_dataset_space(self, mock_dataset: Path) -> None:
        dataset = ZebraDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
        )
        # Test data space
        assert isinstance(dataset.space, DataSpace)
        assert dataset.space.channels == 1
        assert dataset.space.shape == (2, 2)

    def test_dataset_start_date(self, mock_dataset: Path) -> None:
        dataset = ZebraDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
            start=self.dates_str[1],
        )
        assert dataset.start_date == self.dates_np[1]
        assert dataset.end_date == self.dates_np[-1]
