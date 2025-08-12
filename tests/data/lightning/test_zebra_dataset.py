from pathlib import Path
import numpy as np

from ice_station_zebra.data.lightning.zebra_dataset import ZebraDataset


class TestZebraDataset:
    dates_str = ["2020-01-01", "2020-01-02", "2020-01-03"]
    dates_np = list(map(np.datetime64, dates_str))

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

    def test_dataset_start_date(self, mock_dataset: Path) -> None:
        dataset = ZebraDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
            start=self.dates_str[1],
        )
        assert dataset.start_date == self.dates_np[1]
        assert dataset.end_date == self.dates_np[-1]
