from pathlib import Path
from ice_station_zebra.data.lightning.zebra_dataset import ZebraDataset
import numpy as np


class TestZebraDataset:
    def test_init(self) -> None:
        dataset = ZebraDataset(
            name="test_dataset",
            input_files=[Path("dummy.nc")],
            start="1993-12-01",
            end="1993-12-31",
        )
        assert dataset.start_date == np.datetime64("1993-12-01")
        assert dataset.end_date == np.datetime64("1993-12-31")
        assert dataset.name == "test_dataset"
