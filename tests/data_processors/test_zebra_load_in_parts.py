"""Tests for load_in_parts functionality.

This file contains all tests related to the load_in_parts method, including:
- Basic functionality (loading all parts, resume, force_reset, error handling)
- File locking behavior (timeouts, retries)
- Stale in_progress entry reclamation
- PID/host tracking
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import patch

import pytest
from filelock import Timeout
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from pathlib import Path

from ice_station_zebra.data_processors.preprocessors.ipreprocessor import IPreprocessor

# adjust import path to where your module is
from ice_station_zebra.data_processors.zebra_data_processor import (
    # DEFAULT_STALE_SECONDS,
    ZebraDataProcessor,
)


class DummyPreprocessor(IPreprocessor):
    def download(self, preprocessor_path: Path) -> None:  # pragma: no cover - trivial
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


def _read_tracker(proc: ZebraDataProcessor) -> dict:
    p = proc._part_tracker_path()
    if not p.exists():
        return {"completed": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def test_load_in_parts_loads_all_parts_successfully(
    processor_with_directory_dataset: ZebraDataProcessor,
) -> None:
    """load_in_parts should load all parts and track completion."""
    processor = processor_with_directory_dataset
    with (
        patch.object(processor, "load") as mock_load,
        patch.object(processor, "inspect"),  # Mock inspect to assume dataset exists
    ):
        processor.load_in_parts(resume=False, continue_on_error=False, use_lock=False)

        # Should have called load 10 times (for parts 1/10 - 10/10)
        assert mock_load.call_count == 10
        mock_load.assert_any_call(parts="1/10")
        mock_load.assert_any_call(parts="5/10")
        mock_load.assert_any_call(parts="10/10")

    # Verify all parts are tracked as completed
    tracker = processor._read_part_tracker()
    assert len(tracker["completed"]) == 10
    assert "1/10" in tracker["completed"]
    assert "5/10" in tracker["completed"]
    assert "10/10" in tracker["completed"]
    # Each should have a completed_at timestamp
    for part_spec in ["1/10", "5/10", "10/10"]:
        assert "completed_at" in tracker["completed"][part_spec]


def test_load_in_parts_resumes_skipping_completed_parts(
    processor_with_directory_dataset: ZebraDataProcessor,
) -> None:
    """load_in_parts should skip parts that are already completed when resume=True."""
    processor = processor_with_directory_dataset
    # Pre-mark part 5/10 as completed
    initial_tracker = {
        "completed": {
            "5/10": {"completed_at": "2020-01-15T00:00:00Z"},
        }
    }
    processor._write_part_tracker(initial_tracker)

    with (
        patch.object(processor, "load") as mock_load,
        patch.object(processor, "inspect"),  # Mock inspect to assume dataset exists
    ):
        processor.load_in_parts(resume=True, continue_on_error=False, use_lock=False)

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
    tracker = processor._read_part_tracker()
    assert len(tracker["completed"]) == 10
    assert "1/10" in tracker["completed"]
    assert "5/10" in tracker["completed"]  # Still there from before
    assert "10/10" in tracker["completed"]


def test_load_in_parts_force_reset_clears_tracker(
    processor_with_directory_dataset: ZebraDataProcessor,
) -> None:
    """load_in_parts should clear tracker when force_reset=True."""
    processor = processor_with_directory_dataset
    # Pre-mark all parts as completed
    initial_tracker = {
        "completed": {
            "1/10": {"completed_at": "2020-01-05T00:00:00Z"},
            "5/10": {"completed_at": "2020-01-15T00:00:00Z"},
            "10/10": {"completed_at": "2020-01-25T00:00:00Z"},
        }
    }
    processor._write_part_tracker(initial_tracker)

    with (
        patch.object(processor, "load") as mock_load,
        patch.object(processor, "inspect"),  # Mock inspect to assume dataset exists
    ):
        processor.load_in_parts(
            resume=True, continue_on_error=False, force_reset=True, use_lock=False
        )

        # Should call load for all 10 parts (not skipping any)
        assert mock_load.call_count == 10

    # Verify tracker was cleared and repopulated
    tracker = processor._read_part_tracker()
    assert len(tracker["completed"]) == 10
    # Timestamps should be new (not the original ones)
    for part_spec in ["1/10", "5/10", "10/10"]:
        assert (
            tracker["completed"][part_spec]["completed_at"]
            != initial_tracker["completed"][part_spec]["completed_at"]
        )


def test_load_in_parts_continues_on_error_when_enabled(
    processor_with_directory_dataset: ZebraDataProcessor,
) -> None:
    """load_in_parts should continue to next part when error occurs and continue_on_error=True."""
    processor = processor_with_directory_dataset

    # Make load fail for part 5/10
    call_count = 0
    error_msg = "Simulated load failure"

    def mock_load_side_effect(parts: str) -> None:
        nonlocal call_count
        call_count += 1
        if parts == "5/10":
            raise RuntimeError(error_msg)

    with (
        patch.object(processor, "load", side_effect=mock_load_side_effect),
        patch.object(processor, "inspect"),  # Mock inspect to assume dataset exists
    ):
        processor.load_in_parts(resume=False, continue_on_error=True, use_lock=False)

        # Should have attempted all 10 parts
        assert call_count == 10

    # All 10 parts marked as completed except 5/10 marked as failed
    tracker = processor._read_part_tracker()
    assert len(tracker["completed"]) == 9
    assert "1/10" in tracker["completed"]
    assert "5/10" not in tracker["completed"]
    assert "10/10" in tracker["completed"]


def test_load_in_parts_raises_on_error_when_continue_disabled(
    processor_with_directory_dataset: ZebraDataProcessor,
) -> None:
    """load_in_parts should raise exception when error occurs and continue_on_error=False."""
    processor = processor_with_directory_dataset

    # Make load fail for part 5/10
    error_msg = "Simulated load failure"

    def mock_load_side_effect(parts: str) -> None:
        if parts == "5/10":
            raise RuntimeError(error_msg)
        # First 4 parts should succeed

    with (
        patch.object(processor, "load", side_effect=mock_load_side_effect),
        patch.object(processor, "inspect"),  # Mock inspect to assume dataset exists
        pytest.raises(RuntimeError, match=error_msg),
    ):
        processor.load_in_parts(resume=False, continue_on_error=False, use_lock=False)

    # Only first 4 parts should be marked as completed (5/10 failed and raised)
    tracker = processor._read_part_tracker()
    assert len(tracker["completed"]) == 4
    assert "1/10" in tracker["completed"]
    assert "4/10" in tracker["completed"]
    assert "5/10" not in tracker["completed"]
    assert "10/10" not in tracker["completed"]


def test_lock_timeout_skips_part(
    monkeypatch: pytest.MonkeyPatch,
    processor_with_directory_dataset: ZebraDataProcessor,
) -> None:
    """Simulate FileLock timeout on acquisition by having FileLock.__enter__ raise Timeout.

    The implementation treats lock-timeout as a skip for that part (it will be retried on next run).
    """
    proc = processor_with_directory_dataset

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
        "ice_station_zebra.data_processors.zebra_data_processor.FileLock", BadLock
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
            resume=False,
            continue_on_error=True,
            force_reset=False,
            use_lock=True,
            lock_timeout=0,
        )

    # Because lock acquisition failed for each part claim attempt, no loads should have been attempted
    assert calls == []


# def test_reclaim_stale_in_progress(
#     processor_with_directory_dataset: ZebraDataProcessor,
# ) -> None:
#     """Test that stale in_progress entries are reclaimed and processed."""
#     proc = processor_with_directory_dataset

#     # write a stale in_progress entry older than DEFAULT_STALE_SECONDS
#     now = datetime.now(UTC)
#     started = (
#         (now - timedelta(seconds=DEFAULT_STALE_SECONDS + 10))
#         .isoformat()
#         .replace("+00:00", "Z")
#     )
#     stale = {
#         "in_progress": {
#             "1/10": {"started_at": started, "pid": 12345, "host": "oldhost"}
#         },
#         "completed": {},
#     }
#     proc._write_part_tracker(stale)

#     calls = []

#     def fake_load(parts: str) -> None:
#         calls.append(parts)

#     with (
#         patch.object(proc, "load", side_effect=fake_load),
#         patch.object(proc, "inspect"),  # Mock inspect to assume dataset exists
#     ):
#         # should reclaim the stale 1/10 and run it
#         proc.load_in_parts(
#             resume=True, continue_on_error=False, force_reset=False, use_lock=False
#         )

#     assert "1/10" in calls or "1/10" in _read_tracker(proc)["completed"]


def test_total_parts_override(
    processor_with_directory_dataset: ZebraDataProcessor,
) -> None:
    """Test that total_parts overrides the computed total parts."""
    processor = processor_with_directory_dataset
    # Override to 5 parts instead of computed 10
    with (
        patch.object(processor, "load") as mock_load,
        patch.object(processor, "inspect"),  # Mock inspect to assume dataset exists
    ):
        processor.load_in_parts(
            resume=False,
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
    tracker = processor._read_part_tracker()
    assert len(tracker["completed"]) == 5
    for part_spec in ["1/5", "2/5", "3/5", "4/5", "5/5"]:
        assert part_spec in tracker["completed"]


def test_overwrite_deletes_dataset_and_tracker(
    processor_with_directory_dataset: ZebraDataProcessor,
) -> None:
    """Test that overwrite deletes dataset and tracker, then initializes."""
    processor = processor_with_directory_dataset
    # Create some initial state
    initial_tracker = {
        "completed": {
            "1/10": {"completed_at": "2020-01-05T00:00:00Z"},
        },
        "in_progress": {
            "5/10": {"started_at": "2020-01-10T00:00:00Z"},
        },
    }
    processor._write_part_tracker(initial_tracker)
    # Create some dataset content
    if processor.path_dataset.exists():
        if processor.path_dataset.is_dir():
            shutil.rmtree(processor.path_dataset)
        else:
            processor.path_dataset.unlink()
    processor.path_dataset.mkdir(parents=True, exist_ok=True)
    (processor.path_dataset / "some_file").touch()

    with (
        patch.object(processor, "init") as mock_init,
        patch.object(processor, "load") as mock_load,
    ):
        processor.load_in_parts(
            resume=False, continue_on_error=False, use_lock=False, overwrite=True
        )

        # Should have called init to reinitialize the dataset
        mock_init.assert_called_once_with(overwrite=False)
        # Dataset should have been deleted (we can't easily verify this, but init was called)
        # Tracker should be cleared (all parts should be loaded)
        assert mock_load.call_count == 10

    # Verify tracker was cleared and repopulated
    tracker = processor._read_part_tracker()
    assert len(tracker["completed"]) == 10
    # All parts should have new timestamps
    for part_spec in ["1/10", "5/10", "10/10"]:
        assert part_spec in tracker["completed"]
        assert tracker["completed"][part_spec]["completed_at"] != initial_tracker.get(
            "completed", {}
        ).get(part_spec, {}).get("completed_at", "")


def test_overwrite_skips_force_reset(
    processor_with_directory_dataset: ZebraDataProcessor,
) -> None:
    """Test that force_reset is skipped when overwrite is used."""
    processor = processor_with_directory_dataset
    # Create initial tracker state
    initial_tracker = {
        "completed": {
            "1/10": {"completed_at": "2020-01-05T00:00:00Z"},
        },
        "in_progress": {
            "5/10": {"started_at": "2020-01-10T00:00:00Z"},
        },
    }
    processor._write_part_tracker(initial_tracker)
    # Ensure dataset directory exists
    if processor.path_dataset.exists():
        if processor.path_dataset.is_dir():
            shutil.rmtree(processor.path_dataset)
        else:
            processor.path_dataset.unlink()
    processor.path_dataset.mkdir(parents=True, exist_ok=True)

    with (
        patch.object(processor, "init") as mock_init,
        patch.object(processor, "load") as mock_load,
    ):
        # Both overwrite and force_reset are True
        processor.load_in_parts(
            resume=False,
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
    processor_with_directory_dataset: ZebraDataProcessor,
) -> None:
    """Test that dataset is automatically initialized if it doesn't exist."""
    processor = processor_with_directory_dataset
    # Remove the dataset file/directory
    if processor.path_dataset.exists():
        if processor.path_dataset.is_dir():
            shutil.rmtree(processor.path_dataset)
        else:
            processor.path_dataset.unlink()

    with (
        patch.object(processor, "init") as mock_init,
        patch.object(processor, "inspect", side_effect=FileNotFoundError("not found")),
        patch.object(processor, "load") as mock_load,
    ):
        processor.load_in_parts(resume=False, continue_on_error=False, use_lock=False)

        # Should have called init to initialize the missing dataset
        mock_init.assert_called_once_with(overwrite=False)
        # Should then proceed to load parts
        assert mock_load.call_count == 10


def test_force_reset_clears_in_progress(
    processor_with_directory_dataset: ZebraDataProcessor,
) -> None:
    """Test that force_reset clears both completed and in_progress entries."""
    processor = processor_with_directory_dataset
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
    processor._write_part_tracker(initial_tracker)

    with (
        patch.object(processor, "load") as mock_load,
        patch.object(processor, "inspect"),  # Mock inspect to assume dataset exists
    ):
        processor.load_in_parts(
            resume=True, continue_on_error=False, force_reset=True, use_lock=False
        )

        # Should call load for all 10 parts (not skipping any due to force_reset)
        assert mock_load.call_count == 10

    # Verify tracker was cleared and repopulated (no in_progress entries)
    tracker = processor._read_part_tracker()
    assert len(tracker["completed"]) == 10
    assert "in_progress" not in tracker or len(tracker.get("in_progress", {})) == 0
    # Timestamps should be new
    assert (
        tracker["completed"]["1/10"]["completed_at"]
        != initial_tracker["completed"]["1/10"]["completed_at"]
    )
