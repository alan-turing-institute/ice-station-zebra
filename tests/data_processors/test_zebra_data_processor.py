from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest
from omegaconf import DictConfig, OmegaConf

from ice_station_zebra.data_processors.preprocessors.ipreprocessor import IPreprocessor
from ice_station_zebra.data_processors.zebra_data_processor import ZebraDataProcessor


class DummyPreprocessor(IPreprocessor):
    """A dummy preprocessor for testing."""

    def download(self, preprocessor_path: Path) -> None:  # pragma: no cover - unused
        preprocessor_path.mkdir(parents=True, exist_ok=True)


def _build_processor(tmp_path: Path, dataset_cfg: dict) -> ZebraDataProcessor:
    """Helper to create a ZebraDataProcessor with a dummy preprocessor."""
    full_cfg: DictConfig = OmegaConf.create(
        {
            "base_path": str(tmp_path),
            "datasets": {
                "test": {
                    "name": "test",
                    "preprocessor": {"type": "dummy"},
                    **dataset_cfg,
                }
            },
        }
    )
    return ZebraDataProcessor("test", full_cfg, DummyPreprocessor)


def test_compute_total_parts_defaults_when_missing_dates(tmp_path: Path) -> None:
    """Test that the default total parts is 1 when start/end are missing."""
    processor = _build_processor(
        tmp_path,
        {
            "group_by": "monthly",
        },
    )
    assert processor._compute_total_parts() == 1


@pytest.mark.parametrize(
    ("group_by", "start", "end", "expected"),
    [
        ("monthly", "2020-01-01", "2020-03-01", 3),
        ("month", "2020-01-01", "2020-03-01", 3),
        ("daily", "2020-01-01", "2020-03-01", 3),
        ("1d", "2020-01-01", "2020-03-01", 3),
        ("weekly", "2020-01-01", "2020-02-15", 1),
    ],
)
def test_compute_total_parts_grouping(
    tmp_path: Path,
    group_by: str,
    start: str,
    end: str,
    expected: int,
) -> None:
    """Test that the total parts is computed correctly when start/end are specified."""
    processor = _build_processor(
        tmp_path,
        {
            "start": start,
            "end": end,
            "group_by": group_by,
        },
    )
    assert processor._compute_total_parts() == expected


def test_compute_total_parts_timezone_strings(tmp_path: Path) -> None:
    """Test that the total parts is computed correctly when start/end are timezone strings."""
    processor = _build_processor(
        tmp_path,
        {
            "start": "2020-01-01T00:00:00Z",
            "end": "2020-03-01T00:00:00Z",
            "group_by": "monthly",
        },
    )
    assert processor._compute_total_parts() == 3
