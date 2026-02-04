from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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
            "data": {
                "datasets": {
                    "test": {
                        "name": "test",
                        "preprocessor": {"type": "dummy"},
                        **dataset_cfg,
                    }
                },
            },
        }
    )
    return ZebraDataProcessor("test", full_cfg, DummyPreprocessor)


@pytest.fixture
def processor_with_file_dataset(tmp_path: Path) -> ZebraDataProcessor:
    """Fixture that creates a processor with a file-based dataset path."""
    processor = _build_processor(
        tmp_path,
        {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "group_by": "monthly",
        },
    )
    processor.path_dataset = tmp_path / "test.zarr"
    # Create the file to ensure it exists
    processor.path_dataset.touch()
    return processor


@pytest.fixture
def processor_with_directory_dataset(tmp_path: Path) -> ZebraDataProcessor:
    """Fixture that creates a processor with a directory-based dataset path."""
    processor = _build_processor(
        tmp_path,
        {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "group_by": "monthly",
        },
    )
    processor.path_dataset = tmp_path / "test_dir.zarr"
    processor.path_dataset.mkdir(parents=True, exist_ok=True)
    return processor


def test_part_tracker_path_for_file(
    processor_with_file_dataset: ZebraDataProcessor,
) -> None:
    """When dataset path is a file, tracker should use a sibling JSON with custom suffix."""
    processor = processor_with_file_dataset
    tracker_path = processor._part_tracker_path()
    assert tracker_path == processor.path_dataset.with_suffix(".load_parts.json")
    assert processor.path_dataset.is_file()


def test_part_tracker_path_for_directory(
    processor_with_directory_dataset: ZebraDataProcessor,
) -> None:
    """When dataset path is a directory, tracker should live inside it as .load_parts.json."""
    processor = processor_with_directory_dataset
    tracker_path = processor._part_tracker_path()
    assert tracker_path == processor.path_dataset / ".load_parts.json"
    assert processor.path_dataset.is_dir()


def test_read_part_tracker_returns_default_when_missing(
    processor_with_file_dataset: ZebraDataProcessor,
) -> None:
    """_read_part_tracker should return an empty completed map when no file exists."""
    processor = processor_with_file_dataset
    tracker_path = processor._part_tracker_path()
    assert not tracker_path.exists()
    assert processor._read_part_tracker() == {"completed": {}}


def test_write_and_read_part_tracker_roundtrip(
    processor_with_file_dataset: ZebraDataProcessor,
) -> None:
    """_write_part_tracker should persist JSON that _read_part_tracker can load."""
    processor = processor_with_file_dataset
    data = {"completed": {"1/3": {"completed_at": "2020-01-05T00:00:00Z"}}}
    processor._write_part_tracker(data)

    tracker_path = processor._part_tracker_path()
    assert tracker_path.exists()

    # File content should be valid JSON with the same structure
    with tracker_path.open("r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk == data
    assert processor._read_part_tracker() == data


def test_write_part_tracker_fallback_on_atomic_write_failure(
    processor_with_file_dataset: ZebraDataProcessor,
) -> None:
    """_write_part_tracker should fallback to naive write when atomic write fails."""
    processor = processor_with_file_dataset
    data = {"completed": {"2/3": {"completed_at": "2020-01-10T00:00:00Z"}}}
    tracker_path = processor._part_tracker_path()

    # Mock Path.replace() to raise OSError, simulating atomic write failure
    with patch.object(
        Path, "replace", side_effect=OSError("Cross-device link not permitted")
    ):
        processor._write_part_tracker(data)

    # Verify the file was written via fallback path
    assert tracker_path.exists()
    with tracker_path.open("r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk == data
    assert processor._read_part_tracker() == data
