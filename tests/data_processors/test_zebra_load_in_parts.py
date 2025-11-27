"""Tests for load_in_parts functionality.

This file contains all tests related to the load_in_parts method, including:
- Basic functionality (loading all parts, resume, force_reset, error handling)
- File locking behavior (timeouts, retries)
- Stale in_progress entry reclamation
- PID/host tracking
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from filelock import Timeout
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from pathlib import Path

from ice_station_zebra.data_processors.preprocessors.ipreprocessor import IPreprocessor

# adjust import path to where your module is
from ice_station_zebra.data_processors.zebra_data_processor import (
    DEFAULT_STALE_SECONDS,
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
def processor_three_months(tmp_path: Path) -> ZebraDataProcessor:
    """Processor configured to create 3 parts (monthly)."""
    proc = _build_processor(
        tmp_path,
        {"start": "2020-01-01", "end": "2020-03-31", "group_by": "monthly"},
    )
    # make the dataset path exist as a file (file-case) to exercise .with_suffix() logic
    proc.path_dataset = tmp_path / "test.zarr"
    proc.path_dataset.touch()
    return proc


@pytest.fixture
def processor_with_three_parts(tmp_path: Path) -> ZebraDataProcessor:
    """Fixture that creates a processor configured for 3 parts."""
    processor = _build_processor(
        tmp_path,
        {
            "start": "2020-01-01",
            "end": "2020-03-31",
            "group_by": "monthly",
        },
    )
    processor.path_dataset = tmp_path / "test.zarr"
    processor.path_dataset.touch()
    return processor


def _read_tracker(proc: ZebraDataProcessor) -> dict:
    p = proc._part_tracker_path()
    if not p.exists():
        return {"completed": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def test_load_in_parts_loads_all_parts_successfully(
    processor_with_three_parts: ZebraDataProcessor,
) -> None:
    """load_in_parts should load all parts and track completion."""
    processor = processor_with_three_parts
    with patch.object(processor, "load") as mock_load:
        processor.load_in_parts(resume=False, continue_on_error=False, use_lock=False)

        # Should have called load 3 times (for parts 1/3, 2/3, 3/3)
        assert mock_load.call_count == 3
        mock_load.assert_any_call(parts="1/3")
        mock_load.assert_any_call(parts="2/3")
        mock_load.assert_any_call(parts="3/3")

    # Verify all parts are tracked as completed
    tracker = processor._read_part_tracker()
    assert len(tracker["completed"]) == 3
    assert "1/3" in tracker["completed"]
    assert "2/3" in tracker["completed"]
    assert "3/3" in tracker["completed"]
    # Each should have a completed_at timestamp
    for part_spec in ["1/3", "2/3", "3/3"]:
        assert "completed_at" in tracker["completed"][part_spec]


def test_load_in_parts_resumes_skipping_completed_parts(
    processor_with_three_parts: ZebraDataProcessor,
) -> None:
    """load_in_parts should skip parts that are already completed when resume=True."""
    processor = processor_with_three_parts
    # Pre-mark part 2/3 as completed
    initial_tracker = {
        "completed": {
            "2/3": {"completed_at": "2020-01-15T00:00:00Z"},
        }
    }
    processor._write_part_tracker(initial_tracker)

    with patch.object(processor, "load") as mock_load:
        processor.load_in_parts(resume=True, continue_on_error=False, use_lock=False)

        # Should only call load for parts 1/3 and 3/3 (skipping 2/3)
        assert mock_load.call_count == 2
        mock_load.assert_any_call(parts="1/3")
        mock_load.assert_any_call(parts="3/3")
        # Should not have called 2/3
        called_parts = [
            call.kwargs.get("parts") or (call.args[0] if call.args else None)
            for call in mock_load.call_args_list
        ]
        assert "2/3" not in called_parts

    # Verify all parts are now completed
    tracker = processor._read_part_tracker()
    assert len(tracker["completed"]) == 3
    assert "1/3" in tracker["completed"]
    assert "2/3" in tracker["completed"]  # Still there from before
    assert "3/3" in tracker["completed"]


def test_load_in_parts_force_reset_clears_tracker(
    processor_with_three_parts: ZebraDataProcessor,
) -> None:
    """load_in_parts should clear tracker when force_reset=True."""
    processor = processor_with_three_parts
    # Pre-mark all parts as completed
    initial_tracker = {
        "completed": {
            "1/3": {"completed_at": "2020-01-05T00:00:00Z"},
            "2/3": {"completed_at": "2020-01-15T00:00:00Z"},
            "3/3": {"completed_at": "2020-01-25T00:00:00Z"},
        }
    }
    processor._write_part_tracker(initial_tracker)

    with patch.object(processor, "load") as mock_load:
        processor.load_in_parts(
            resume=True, continue_on_error=False, force_reset=True, use_lock=False
        )

        # Should call load for all 3 parts (not skipping any)
        assert mock_load.call_count == 3

    # Verify tracker was cleared and repopulated
    tracker = processor._read_part_tracker()
    assert len(tracker["completed"]) == 3
    # Timestamps should be new (not the original ones)
    for part_spec in ["1/3", "2/3", "3/3"]:
        assert (
            tracker["completed"][part_spec]["completed_at"]
            != initial_tracker["completed"][part_spec]["completed_at"]
        )


def test_load_in_parts_continues_on_error_when_enabled(
    processor_with_three_parts: ZebraDataProcessor,
) -> None:
    """load_in_parts should continue to next part when error occurs and continue_on_error=True."""
    processor = processor_with_three_parts

    # Make load fail for part 2/3
    call_count = 0
    error_msg = "Simulated load failure"

    def mock_load_side_effect(parts: str) -> None:
        nonlocal call_count
        call_count += 1
        if parts == "2/3":
            raise RuntimeError(error_msg)

    with patch.object(processor, "load", side_effect=mock_load_side_effect):
        processor.load_in_parts(resume=False, continue_on_error=True, use_lock=False)

        # Should have attempted all 3 parts
        assert call_count == 3

    # Only parts 1/3 and 3/3 should be marked as completed (2/3 failed)
    tracker = processor._read_part_tracker()
    assert len(tracker["completed"]) == 2
    assert "1/3" in tracker["completed"]
    assert "2/3" not in tracker["completed"]
    assert "3/3" in tracker["completed"]


def test_load_in_parts_raises_on_error_when_continue_disabled(
    processor_with_three_parts: ZebraDataProcessor,
) -> None:
    """load_in_parts should raise exception when error occurs and continue_on_error=False."""
    processor = processor_with_three_parts

    # Make load fail for part 2/3
    error_msg = "Simulated load failure"

    def mock_load_side_effect(parts: str) -> None:
        if parts == "2/3":
            raise RuntimeError(error_msg)
        # Part 1/3 should succeed

    with (
        patch.object(processor, "load", side_effect=mock_load_side_effect),
        pytest.raises(RuntimeError, match=error_msg),
    ):
        processor.load_in_parts(resume=False, continue_on_error=False, use_lock=False)

    # Only part 1/3 should be marked as completed (2/3 failed and raised)
    tracker = processor._read_part_tracker()
    assert len(tracker["completed"]) == 1
    assert "1/3" in tracker["completed"]
    assert "2/3" not in tracker["completed"]
    assert "3/3" not in tracker["completed"]


def test_lock_timeout_skips_part(
    monkeypatch: pytest.MonkeyPatch, processor_three_months: ZebraDataProcessor
) -> None:
    """Simulate FileLock timeout on acquisition by having FileLock.__enter__ raise Timeout.

    The implementation treats lock-timeout as a skip for that part (it will be retried on next run).
    """
    proc = processor_three_months

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
        ) -> bool:
            return False

    monkeypatch.setattr(
        "ice_station_zebra.data_processors.zebra_data_processor.FileLock", BadLock
    )

    calls = []

    def fake_load(parts: str) -> None:
        calls.append(parts)

    # run: since lock acquisition fails, load() should not be called for that part
    with patch.object(proc, "load", side_effect=fake_load):
        proc.load_in_parts(
            resume=False,
            continue_on_error=True,
            force_reset=False,
            use_lock=True,
            lock_timeout=0,
        )

    # Because lock acquisition failed for each part claim attempt, no loads should have been attempted
    assert calls == []


def test_reclaim_stale_in_progress(
    processor_three_months: ZebraDataProcessor,
) -> None:
    """Test that stale in_progress entries are reclaimed and processed."""
    proc = processor_three_months

    # write a stale in_progress entry older than DEFAULT_STALE_SECONDS
    now = datetime.now(UTC)
    started = (
        (now - timedelta(seconds=DEFAULT_STALE_SECONDS + 10))
        .isoformat()
        .replace("+00:00", "Z")
    )
    stale = {
        "in_progress": {
            "1/3": {"started_at": started, "pid": 12345, "host": "oldhost"}
        },
        "completed": {},
    }
    proc._write_part_tracker(stale)

    calls = []

    def fake_load(parts: str) -> None:
        calls.append(parts)

    with patch.object(proc, "load", side_effect=fake_load):
        # should reclaim the stale 1/3 and run it
        proc.load_in_parts(
            resume=True, continue_on_error=False, force_reset=False, use_lock=False
        )

    assert "1/3" in calls or "1/3" in _read_tracker(proc)["completed"]
