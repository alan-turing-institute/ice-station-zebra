import json
import shutil
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest
from filelock import Timeout
from omegaconf import DictConfig, OmegaConf

from ice_station_zebra.data_processors.data_downloader import DataDownloader
from ice_station_zebra.data_processors.preprocessors.ipreprocessor import IPreprocessor


class DummyPreprocessor(IPreprocessor):
    """A dummy preprocessor for testing."""

    def download(self, preprocessor_path: Path) -> None:  # pragma: no cover - unused
        preprocessor_path.mkdir(parents=True, exist_ok=True)


def _build_downloader(tmp_path: Path, dataset_cfg: dict) -> DataDownloader:
    """Helper to create a DataDownloader with a dummy preprocessor."""
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
    return DataDownloader("test", full_cfg, DummyPreprocessor)


@pytest.fixture
def downloader_with_file_dataset(tmp_path: Path) -> DataDownloader:
    """Fixture that creates a downloader with a file-based dataset path."""
    downloader = _build_downloader(
        tmp_path,
        {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "group_by": "monthly",
        },
    )
    downloader.path_dataset = tmp_path / "test.zarr"
    # Create the file to ensure it exists
    downloader.path_dataset.touch()
    return downloader


@pytest.fixture
def downloader_with_directory_dataset(tmp_path: Path) -> DataDownloader:
    """Fixture that creates a downloader with a directory-based dataset path."""
    downloader = _build_downloader(
        tmp_path,
        {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "group_by": "monthly",
        },
    )
    downloader.path_dataset = tmp_path / "test_dir.zarr"
    downloader.path_dataset.mkdir(parents=True, exist_ok=True)
    return downloader


def test_part_tracker_path_for_file(
    downloader_with_file_dataset: DataDownloader,
) -> None:
    """When dataset path is a file, tracker should use a sibling JSON with custom suffix."""
    downloader = downloader_with_file_dataset
    tracker_path = downloader._part_tracker_path()
    assert tracker_path == downloader.path_dataset.with_suffix(".load_parts.json")
    assert downloader.path_dataset.is_file()


def test_part_tracker_path_for_directory(
    downloader_with_directory_dataset: DataDownloader,
) -> None:
    """When dataset path is a directory, tracker should live inside it as .load_parts.json."""
    downloader = downloader_with_directory_dataset
    tracker_path = downloader._part_tracker_path()
    assert tracker_path == downloader.path_dataset / ".load_parts.json"
    assert downloader.path_dataset.is_dir()


def test_read_part_tracker_returns_default_when_missing(
    downloader_with_file_dataset: DataDownloader,
) -> None:
    """_read_part_tracker should return an empty completed map when no file exists."""
    downloader = downloader_with_file_dataset
    tracker_path = downloader._part_tracker_path()
    assert not tracker_path.exists()
    assert downloader._read_part_tracker() == {"completed": {}}


def test_write_and_read_part_tracker_roundtrip(
    downloader_with_file_dataset: DataDownloader,
) -> None:
    """_write_part_tracker should persist JSON that _read_part_tracker can load."""
    downloader = downloader_with_file_dataset
    data = {"completed": {"1/3": {"completed_at": "2020-01-05T00:00:00Z"}}}
    downloader._write_part_tracker(data)

    tracker_path = downloader._part_tracker_path()
    assert tracker_path.exists()

    # File content should be valid JSON with the same structure
    with tracker_path.open("r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk == data
    assert downloader._read_part_tracker() == data


def test_write_part_tracker_fallback_on_atomic_write_failure(
    downloader_with_file_dataset: DataDownloader,
) -> None:
    """_write_part_tracker should fallback to naive write when atomic write fails."""
    downloader = downloader_with_file_dataset
    data = {"completed": {"2/3": {"completed_at": "2020-01-10T00:00:00Z"}}}
    tracker_path = downloader._part_tracker_path()

    # Mock Path.replace() to raise OSError, simulating atomic write failure
    with patch.object(
        Path, "replace", side_effect=OSError("Cross-device link not permitted")
    ):
        downloader._write_part_tracker(data)

    # Verify the file was written via fallback path
    assert tracker_path.exists()
    with tracker_path.open("r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk == data
    assert downloader._read_part_tracker() == data


def _read_tracker(proc: DataDownloader) -> dict:
    p = proc._part_tracker_path()
    if not p.exists():
        return {"completed": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def test_load_in_parts_loads_all_parts_successfully(
    downloader_with_directory_dataset: DataDownloader,
) -> None:
    """load_in_parts should load all parts and track completion."""
    downloader = downloader_with_directory_dataset

    initial_tracker = _read_tracker(downloader)
    downloader._write_part_tracker(initial_tracker)

    with (
        patch.object(downloader, "load") as mock_load,
        patch.object(downloader, "inspect"),  # Mock inspect to assume dataset exists
    ):
        downloader.load_in_parts(continue_on_error=False, use_lock=False)
        # Should have called load 10 times (for parts 1/10 - 10/10)
        assert mock_load.call_count == 10
        mock_load.assert_any_call(parts="1/10")
        mock_load.assert_any_call(parts="5/10")
        mock_load.assert_any_call(parts="10/10")

    # Verify all parts are tracked as completed
    tracker = downloader._read_part_tracker()
    assert len(tracker["completed"]) == 10
    assert "1/10" in tracker["completed"]
    assert "5/10" in tracker["completed"]
    assert "10/10" in tracker["completed"]
    # Each should have a completed_at timestamp
    for part_spec in ["1/10", "5/10", "10/10"]:
        assert "completed_at" in tracker["completed"][part_spec]


def test_load_in_parts_resumes_skipping_completed_parts(
    downloader_with_directory_dataset: DataDownloader,
) -> None:
    """load_in_parts should skip parts that are already completed."""
    downloader = downloader_with_directory_dataset
    # Pre-mark part 5/10 as completed
    initial_tracker = {
        "completed": {
            "5/10": {"completed_at": "2020-01-15T00:00:00Z"},
        }
    }
    downloader._write_part_tracker(initial_tracker)

    with (
        patch.object(downloader, "load") as mock_load,
        patch.object(downloader, "inspect"),  # Mock inspect to assume dataset exists
    ):
        downloader.load_in_parts(continue_on_error=False, use_lock=False)

        # Should call load for all 10 parts except 5/10
        assert mock_load.call_count == 9
        mock_load.assert_any_call(parts="1/10")
        mock_load.assert_any_call(parts="10/10")
        # Should not have called 5/10
        called_parts = [
            call.kwargs.get("parts") or (call.args[0] if call.args else None)
            for call in mock_load.call_args_list
        ]
        assert "5/10" not in called_parts

    # Verify all parts are now completed
    tracker = downloader._read_part_tracker()
    assert len(tracker["completed"]) == 10
    assert "1/10" in tracker["completed"]
    assert "5/10" in tracker["completed"]  # Still there from before
    assert "10/10" in tracker["completed"]


def test_load_in_parts_force_reset_clears_tracker(
    downloader_with_directory_dataset: DataDownloader,
) -> None:
    """load_in_parts should clear tracker when force_reset=True."""
    downloader = downloader_with_directory_dataset
    # Pre-mark all parts as completed
    initial_tracker = {
        "completed": {
            "1/10": {"completed_at": "2020-01-05T00:00:00Z"},
            "5/10": {"completed_at": "2020-01-15T00:00:00Z"},
            "10/10": {"completed_at": "2020-01-25T00:00:00Z"},
        }
    }
    downloader._write_part_tracker(initial_tracker)

    with (
        patch.object(downloader, "load") as mock_load,
        patch.object(downloader, "inspect"),  # Mock inspect to assume dataset exists
    ):
        downloader.load_in_parts(
            continue_on_error=False, force_reset=True, use_lock=False
        )

        # Should call load for all 10 parts (not skipping any)
        assert mock_load.call_count == 10

    # Verify tracker was cleared and repopulated
    tracker = downloader._read_part_tracker()
    assert len(tracker["completed"]) == 10
    # Timestamps should be new (not the original ones)
    for part_spec in ["1/10", "5/10", "10/10"]:
        assert (
            tracker["completed"][part_spec]["completed_at"]
            != initial_tracker["completed"][part_spec]["completed_at"]
        )


def test_load_in_parts_continues_on_error_when_enabled(
    downloader_with_directory_dataset: DataDownloader,
) -> None:
    """load_in_parts should continue to next part when error occurs and continue_on_error=True."""
    downloader = downloader_with_directory_dataset

    initial_tracker = _read_tracker(downloader)
    downloader._write_part_tracker(initial_tracker)

    # Make load fail for part 5/10
    call_count = 0
    error_msg = "Simulated load failure"

    def mock_load_side_effect(parts: str) -> None:
        nonlocal call_count
        call_count += 1
        if parts == "5/10":
            raise RuntimeError(error_msg)

    with (
        patch.object(downloader, "load", side_effect=mock_load_side_effect),
        patch.object(downloader, "inspect"),  # Mock inspect to assume dataset exists
    ):
        downloader.load_in_parts(continue_on_error=True, use_lock=False)

        # Should have attempted all 10 parts
        assert call_count == 10

    # All 10 parts marked as completed except 5/10 marked as failed
    tracker = downloader._read_part_tracker()
    assert len(tracker["completed"]) == 9
    assert "1/10" in tracker["completed"]
    assert "5/10" not in tracker["completed"]
    assert "10/10" in tracker["completed"]


def test_load_in_parts_raises_on_error_when_continue_disabled(
    downloader_with_directory_dataset: DataDownloader,
) -> None:
    """load_in_parts should raise exception when error occurs and continue_on_error=False."""
    downloader = downloader_with_directory_dataset

    initial_tracker = _read_tracker(downloader)
    downloader._write_part_tracker(initial_tracker)

    # Make load fail for part 5/10
    error_msg = "Simulated load failure"

    def mock_load_side_effect(parts: str) -> None:
        if parts == "5/10":
            raise RuntimeError(error_msg)
        # First 4 parts should succeed

    with (
        patch.object(downloader, "load", side_effect=mock_load_side_effect),
        patch.object(downloader, "inspect"),  # Mock inspect to assume dataset exists
        pytest.raises(RuntimeError, match=error_msg),
    ):
        downloader.load_in_parts(continue_on_error=False, use_lock=False)

    # Only first 4 parts should be marked as completed (5/10 failed and raised)
    tracker = downloader._read_part_tracker()
    assert len(tracker["completed"]) == 4
    assert "1/10" in tracker["completed"]
    assert "4/10" in tracker["completed"]
    assert "5/10" not in tracker["completed"]
    assert "10/10" not in tracker["completed"]


def test_lock_timeout_skips_part(
    monkeypatch: pytest.MonkeyPatch,
    downloader_with_directory_dataset: DataDownloader,
) -> None:
    """Simulate FileLock timeout on acquisition by having FileLock.__enter__ raise Timeout.

    The implementation treats lock-timeout as a skip for that part (it will be retried on next run).
    """
    proc = downloader_with_directory_dataset

    # Patch FileLock to raise Timeout on entering context
    timeout_msg = "could not acquire"

    class BadLock:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> None:
            raise Timeout(timeout_msg)

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object,
        ) -> Literal[False]:
            return False

    monkeypatch.setattr(
        "ice_station_zebra.data_processors.data_downloader.FileLock", BadLock
    )

    calls = []

    def fake_load(parts: str) -> None:
        calls.append(parts)

    # run: since lock acquisition fails, load() should not be called for that part
    with (
        patch.object(proc, "load", side_effect=fake_load),
        patch.object(proc, "inspect"),  # Mock inspect to assume dataset exists
    ):
        proc.load_in_parts(
            continue_on_error=True,
            force_reset=False,
            use_lock=True,
            lock_timeout=0,
        )

    # Because lock acquisition failed for each part claim attempt, no loads should have been attempted
    assert calls == []


def test_total_parts_override(
    downloader_with_directory_dataset: DataDownloader,
) -> None:
    """Test that total_parts overrides the computed total parts."""
    downloader = downloader_with_directory_dataset

    initial_tracker = _read_tracker(downloader)
    downloader._write_part_tracker(initial_tracker)

    # Override to 5 parts instead of computed 10
    with (
        patch.object(downloader, "load") as mock_load,
        patch.object(downloader, "inspect"),  # Mock inspect to assume dataset exists
    ):
        downloader.load_in_parts(
            continue_on_error=False,
            use_lock=False,
            total_parts=5,
        )

        # Should have called load 5 times (for parts 1/5, 2/5, 3/5, 4/5, 5/5)
        assert mock_load.call_count == 5
        mock_load.assert_any_call(parts="1/5")
        mock_load.assert_any_call(parts="2/5")
        mock_load.assert_any_call(parts="3/5")
        mock_load.assert_any_call(parts="4/5")
        mock_load.assert_any_call(parts="5/5")

    # Verify all 5 parts are tracked as completed
    tracker = downloader._read_part_tracker()
    assert len(tracker["completed"]) == 5
    for part_spec in ["1/5", "2/5", "3/5", "4/5", "5/5"]:
        assert part_spec in tracker["completed"]


def test_overwrite_deletes_dataset_and_tracker(
    downloader_with_directory_dataset: DataDownloader,
) -> None:
    """Test that overwrite deletes dataset and tracker, then initializes."""
    downloader = downloader_with_directory_dataset
    # Create some initial state
    initial_tracker = {
        "completed": {
            "1/10": {"completed_at": "2020-01-05T00:00:00Z"},
        },
        "in_progress": {
            "5/10": {"started_at": "2020-01-10T00:00:00Z"},
        },
    }
    downloader._write_part_tracker(initial_tracker)
    # Create some dataset content
    if downloader.path_dataset.exists():
        if downloader.path_dataset.is_dir():
            shutil.rmtree(downloader.path_dataset)
        else:
            downloader.path_dataset.unlink()
    downloader.path_dataset.mkdir(parents=True, exist_ok=True)
    (downloader.path_dataset / "some_file").touch()

    with (
        patch.object(downloader, "init") as mock_init,
        patch.object(downloader, "load") as mock_load,
    ):
        downloader.load_in_parts(
            continue_on_error=False, use_lock=False, overwrite=True
        )

        # Should have called init to reinitialize the dataset
        mock_init.assert_called_once_with(overwrite=False)
        # Dataset should have been deleted (we can't easily verify this, but init was called)
        # Tracker should be cleared (all parts should be loaded)
        assert mock_load.call_count == 10

    # Verify tracker was cleared and repopulated
    tracker = downloader._read_part_tracker()
    assert len(tracker["completed"]) == 10
    # All parts should have new timestamps
    for part_spec in ["1/10", "5/10", "10/10"]:
        assert part_spec in tracker["completed"]
        assert tracker["completed"][part_spec]["completed_at"] != initial_tracker.get(
            "completed", {}
        ).get(part_spec, {}).get("completed_at", "")


def test_overwrite_skips_force_reset(
    downloader_with_directory_dataset: DataDownloader,
) -> None:
    """Test that force_reset is skipped when overwrite is used."""
    downloader = downloader_with_directory_dataset
    # Create initial tracker state
    initial_tracker = {
        "completed": {
            "1/10": {"completed_at": "2020-01-05T00:00:00Z"},
        },
        "in_progress": {
            "5/10": {"started_at": "2020-01-10T00:00:00Z"},
        },
    }
    downloader._write_part_tracker(initial_tracker)
    # Ensure dataset directory exists
    if downloader.path_dataset.exists():
        if downloader.path_dataset.is_dir():
            shutil.rmtree(downloader.path_dataset)
        else:
            downloader.path_dataset.unlink()
    downloader.path_dataset.mkdir(parents=True, exist_ok=True)

    with (
        patch.object(downloader, "init") as mock_init,
        patch.object(downloader, "load") as mock_load,
    ):
        # Both overwrite and force_reset are True
        downloader.load_in_parts(
            continue_on_error=False,
            use_lock=False,
            overwrite=True,
            force_reset=True,
        )

        # init should be called (by overwrite)
        mock_init.assert_called_once()
        # All parts should be loaded (overwrite clears everything)
        assert mock_load.call_count == 10


def test_automatic_initialization_when_dataset_missing(
    downloader_with_directory_dataset: DataDownloader,
) -> None:
    """Test that dataset is automatically initialized if it doesn't exist."""
    downloader = downloader_with_directory_dataset
    # Remove the dataset file/directory
    if downloader.path_dataset.exists():
        if downloader.path_dataset.is_dir():
            shutil.rmtree(downloader.path_dataset)
        else:
            downloader.path_dataset.unlink()

    with (
        patch.object(downloader, "init") as mock_init,
        patch.object(downloader, "inspect", side_effect=FileNotFoundError("not found")),
        patch.object(downloader, "load") as mock_load,
    ):
        downloader.load_in_parts(continue_on_error=False, use_lock=False)

        # Should have called init to initialize the missing dataset
        mock_init.assert_called_once_with(overwrite=False)
        # Should then proceed to load parts
        assert mock_load.call_count == 10


def test_force_reset_clears_in_progress(
    downloader_with_directory_dataset: DataDownloader,
) -> None:
    """Test that force_reset clears both completed and in_progress entries."""
    downloader = downloader_with_directory_dataset
    # Pre-mark some parts as completed and in_progress
    initial_tracker: dict[str, Any] = {
        "completed": {
            "1/10": {"completed_at": "2020-01-05T00:00:00Z"},
        },
        "in_progress": {
            "5/10": {
                "started_at": "2020-01-10T00:00:00Z",
                "pid": 12345,
                "host": "test",
            },
        },
    }
    downloader._write_part_tracker(initial_tracker)

    with (
        patch.object(downloader, "load") as mock_load,
        patch.object(downloader, "inspect"),  # Mock inspect to assume dataset exists
    ):
        downloader.load_in_parts(
            continue_on_error=False, force_reset=True, use_lock=False
        )

        # Should call load for all 10 parts (not skipping any due to force_reset)
        assert mock_load.call_count == 10

    # Verify tracker was cleared and repopulated (no in_progress entries)
    tracker = downloader._read_part_tracker()
    assert len(tracker["completed"]) == 10
    assert "in_progress" not in tracker or len(tracker.get("in_progress", {})) == 0
    # Timestamps should be new
    assert (
        tracker["completed"]["1/10"]["completed_at"]
        != initial_tracker["completed"]["1/10"]["completed_at"]
    )
