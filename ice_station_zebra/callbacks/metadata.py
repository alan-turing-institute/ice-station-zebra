"""Metadata extraction and formatting for plot titles.

This module provides functions to extract training metadata from Hydra configs
and format them for display in plot titles.
"""

import contextlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ice_station_zebra.data_loaders import CombinedDataset

logger = logging.getLogger(__name__)


@dataclass
class Metadata:
    """Structured metadata extracted from training configuration.

    Attributes:
        model: Model name (if available).
        epochs: Maximum number of training epochs (if available).
        start: Training start date string (if available).
        end: Training end date string (if available).
        cadence: Training data cadence string (if available).
        n_points: Number of training points calculated from date range and cadence.
        vars_by_source: Dictionary mapping dataset source names to lists of variable names.

    """

    model: str | None = None
    epochs: int | None = None
    start: str | None = None
    end: str | None = None
    cadence: str | None = None
    n_points: int | None = None
    vars_by_source: dict[str, list[str]] | None = None


def extract_variables_by_source(config: dict[str, Any]) -> dict[str, list[str]]:  # noqa: C901
    """Extract variable names grouped by dataset source (group_as).

    Args:
        config: Configuration dictionary containing dataset definitions.

    Returns:
        Dictionary mapping dataset group names to lists of their variable names.
        Example: {"era5": ["10u", "10v", "2t"], "osisaf-south": ["sea ice"]}

    """
    vars_by_source: dict[str, list[str]] = {}

    def _extract_weather_params(ds: dict[str, Any]) -> list[str]:
        """Collect 'param' fields from nested input/join/mars blocks."""
        input_cfg = ds.get("input")
        if not isinstance(input_cfg, dict):
            return []

        join_items = input_cfg.get("join", [])
        if not isinstance(join_items, list):
            return []

        params: list[str] = []
        for item in join_items:
            if not isinstance(item, dict):
                continue
            # Accept both MARS and FORCINGS sources
            for source_type in ("mars", "forcings"):
                cfg = item.get(source_type)
                if isinstance(cfg, dict):
                    param_list = cfg.get("param", [])
                    if isinstance(param_list, list):
                        params.extend(str(p) for p in param_list if p)

        return sorted(set(params))

    try:
        datasets = config.get("datasets", {})
        if not isinstance(datasets, dict):
            return vars_by_source

        for ds in datasets.values():
            if not isinstance(ds, dict):
                continue

            ds_name = str(ds.get("name", "")).lower()
            group_name = ds.get("group_as")
            if not isinstance(group_name, str):
                continue

            # --- Infer variables based on dataset type ---
            variables: list[str] = []
            if "sicnorth" in ds_name or "sicsouth" in ds_name:
                variables = ["sea ice"]
            elif "weather" in ds_name:
                variables = _extract_weather_params(ds)

            # --- Add variables if any ---
            if not variables:
                continue

            group_vars = vars_by_source.setdefault(group_name, [])
            for v in variables:
                if v not in group_vars:
                    group_vars.append(v)

    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("Failed to extract variables from config: %s", exc, exc_info=True)

    # Sort variables alphabetically per source
    return {src: sorted(vs) for src, vs in vars_by_source.items()}


def calculate_training_points(
    start_str: str | None, end_str: str | None, cadence_str: str | None
) -> int | None:
    """Calculate number of training points from date range and cadence.

    Calculates the number of time points in an inclusive date range given a cadence.
    For example, Jan 1 to Jan 10 with 1d cadence = 10 points (inclusive endpoints).

    Args:
        start_str: Start date string (ISO format, with or without time).
            Time components are stripped before calculation.
        end_str: End date string (ISO format, with or without time).
            Time components are stripped before calculation.
        cadence_str: Cadence string (e.g., "1d", "3h", "daily", "24h").

    Returns:
        Number of points (at least 1) or None if calculation fails.
        Returns None if any input is None/empty or if cadence format is unrecognized.

    """
    if not start_str or not end_str or not cadence_str:
        return None

    try:
        delta_days = _inclusive_days(start_str, end_str)
        computed_points = _points_from_cadence(delta_days, cadence_str)
    except (ValueError, TypeError) as exc:
        logger.debug(
            "Failed to calculate training points from dates/cadence: %s",
            exc,
            exc_info=True,
        )
        return None
    else:
        return computed_points


def format_cadence_display(cadence_str: str | None) -> str | None:
    """Format cadence string for display (converts 1d/1h to daily/hourly).

    Args:
        cadence_str: Raw cadence string from config.

    Returns:
        Formatted cadence string (daily/hourly or original if not 1d/1h).

    """
    if not cadence_str:
        return None
    norm = cadence_str.strip().lower()
    if norm in {"1d", "1day", "1 day"}:
        return "daily"
    if norm in {"1h", "1hr", "1 hour"}:
        return "hourly"
    return cadence_str


def extract_cadence_from_config(config: dict[str, Any]) -> str | None:
    """Extract cadence (frequency) from dataset config for the prediction target group.

    Args:
        config: Configuration dictionary.

    Returns:
        Cadence string (e.g., "1d", "3h") or None if not found.

    """
    try:
        predict_group = config.get("predict", {}).get("dataset_group")
        datasets_cfg = config.get("datasets", {})
        if isinstance(predict_group, str) and isinstance(datasets_cfg, dict):
            for ds in datasets_cfg.values():
                if not isinstance(ds, dict):
                    continue
                if ds.get("group_as") == predict_group:
                    dates_section = ds.get("dates")
                    if isinstance(dates_section, dict):
                        freq_candidate = dates_section.get("frequency")
                        if isinstance(freq_candidate, str) and freq_candidate:
                            return freq_candidate
    except (AttributeError, TypeError) as exc:
        logger.debug("Failed to extract cadence from config: %s", exc, exc_info=True)
    return None


def extract_training_date_range(
    config: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Extract training date range from split config.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (start_date_str, end_date_str) or (None, None) if not found.

    """
    start_str: str | None = None
    end_str: str | None = None
    try:
        split_cfg = config.get("split", {})
        train_ranges = split_cfg.get("train") if isinstance(split_cfg, dict) else None
        if isinstance(train_ranges, list) and train_ranges:
            starts = [r.get("start") for r in train_ranges if isinstance(r, dict)]
            ends = [r.get("end") for r in train_ranges if isinstance(r, dict)]
            non_null_starts = [s for s in starts if isinstance(s, str) and s]
            non_null_ends = [e for e in ends if isinstance(e, str) and e]
            start_str = min(non_null_starts) if non_null_starts else None
            end_str = max(non_null_ends) if non_null_ends else None
    except (AttributeError, TypeError) as exc:
        logger.debug(
            "Failed to extract training date range from config: %s", exc, exc_info=True
        )
    return start_str, end_str


def build_metadata(
    config: dict[str, Any],
    model_name: str | None = None,
) -> Metadata:
    """Build structured metadata from configuration.

    Extracts training metadata from config and returns a Metadata dataclass.
    All fields are optional and will be None if the corresponding information
    is not available in the config.

    Args:
        config: Configuration dictionary containing training and dataset info.
        model_name: Optional model name (if not provided, will not be included).

    Returns:
        Metadata dataclass instance with extracted information.

    """
    # Extract training date range
    start_str, end_str = extract_training_date_range(config)

    # Extract cadence
    cadence_str = extract_cadence_from_config(config)
    training_points = calculate_training_points(start_str, end_str, cadence_str)

    # Get epochs
    trainer_cfg = config.get("train", {}).get("trainer", {})
    max_epochs = (
        trainer_cfg.get("max_epochs") if isinstance(trainer_cfg, dict) else None
    )

    # Get variables grouped by source
    vars_by_source = extract_variables_by_source(config)

    return Metadata(
        model=model_name if isinstance(model_name, str) and model_name else None,
        epochs=max_epochs if isinstance(max_epochs, int) else None,
        start=start_str,
        end=end_str,
        cadence=cadence_str,
        n_points=training_points,
        vars_by_source=vars_by_source if vars_by_source else None,
    )


def format_metadata_subtitle(metadata: Metadata) -> str | None:  # noqa: C901
    """Format metadata dataclass as a compact multi-line subtitle for plot titles.

    Lines:
      1) Model: <model>  Epoch: <num>  Training Dates: <start> — <end> (<cadence>) <num>pts
      2) Training Data: <source> (<vars>) <source> (<vars>)

    Args:
        metadata: Metadata dataclass instance to format.

    Returns:
        Formatted metadata string with newlines, or None if no metadata available.

    """
    lines: list[str] = []

    # Line 1: Model/Epoch/Dates
    info_parts: list[str] = []
    if metadata.model:
        info_parts.append(f"Model: {metadata.model}")
    if metadata.epochs is not None:
        info_parts.append(f"Epoch: {metadata.epochs}")

    if metadata.start or metadata.end:
        s_clean = (
            metadata.start.split("T")[0]
            if metadata.start and "T" in metadata.start
            else (metadata.start if metadata.start else "?")
        )
        e_clean = (
            metadata.end.split("T")[0]
            if metadata.end and "T" in metadata.end
            else (metadata.end if metadata.end else "?")
        )
        cadence_display = format_cadence_display(metadata.cadence)
        dates_part = f"Training Dates: {s_clean} — {e_clean}"
        if cadence_display:
            dates_part += f" ({cadence_display})"
        if metadata.n_points is not None:
            dates_part += f" {metadata.n_points} pts"
        info_parts.append(dates_part)

    if info_parts:
        lines.append("  ".join(info_parts))

    # Line 2: Training data sources and variables
    if metadata.vars_by_source:
        source_parts = []
        for source in sorted(metadata.vars_by_source.keys()):
            vars_list = metadata.vars_by_source[source]
            if vars_list:
                vars_str = ",".join(vars_list)
                source_parts.append(f"{source} ({vars_str})")
            else:
                source_parts.append(source)
        if source_parts:
            lines.append(f"Training Data: {' '.join(source_parts)}")

    return "\n".join(lines) if lines else None


def build_metadata_subtitle(
    config: dict[str, Any],
    model_name: str | None = None,
) -> str | None:
    """Build metadata subtitle for plot titles.

    Convenience function that combines build_metadata and format_metadata_subtitle.
    Maintains backward compatibility with existing code.

    Args:
        config: Configuration dictionary containing training and dataset info.
        model_name: Optional model name (if not provided, will not be included).

    Returns:
        Formatted metadata string with newlines, or None if no metadata available.

    """
    metadata = build_metadata(config, model_name=model_name)
    return format_metadata_subtitle(metadata)


def infer_hemisphere(dataset: CombinedDataset) -> str | None:  # noqa: C901, PLR0912
    """Infer hemisphere from dataset name or config as a fallback.

    Priority:
    1) CombinedDataset.target.name containing "north"/"south"
    2) Any input dataset name containing "north"/"south"
    3) Dataset-level name or config strings containing the keywords

    Args:
        dataset: CombinedDataset instance to infer hemisphere from.

    Returns:
        "north" or "south" (lowercase) when detected, otherwise None.

    """
    candidate_names: list[str] = []

    # 1) Target dataset name
    target = getattr(dataset, "target", None)
    target_name = getattr(target, "name", None)
    if isinstance(target_name, str) and target_name:
        candidate_names.append(target_name)

    # 2) Top-level dataset name
    ds_name = getattr(dataset, "name", None)
    if isinstance(ds_name, str) and ds_name:
        candidate_names.append(ds_name)

    # 3) Inputs: may be a Sequence of objects, mappings or plain strings
    inputs = getattr(dataset, "inputs", None)
    if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
        for item in inputs:
            # If the item is a mapping-like object (dict), try key access
            if isinstance(item, Mapping):
                name = item.get("name") or item.get("dataset_name") or None
            else:
                # Otherwise try attribute access, then try if item itself is a string
                name = (
                    getattr(item, "name", None) if not isinstance(item, str) else item
                )

            if isinstance(name, str) and name:
                candidate_names.append(name)

    # 4) Generic config-like hints: look for a config attribute (mapping) and make a string of a few keys
    config_like = getattr(dataset, "config", None) or getattr(
        dataset, "dataset_config", None
    )
    if isinstance(config_like, Mapping):
        # Check a few plausible keys
        for key in ("name", "dataset", "dataset_name", "target"):
            val = config_like.get(key)
            if isinstance(val, str) and val:
                candidate_names.append(val)
        # As a last resort, make the mapping (small) into a string and use as a candidate
        try:
            maybe_str = str(config_like)
            if maybe_str:
                candidate_names.append(maybe_str)
        except TypeError as exc:
            logger.debug(
                "Failed to extract config hint for hemisphere inference: %s",
                exc,
                exc_info=True,
            )

    # Normalise and search for hemisphere keywords.
    for cand in candidate_names:
        low = cand.lower()
        if "north" in low:
            logger.debug("Inferred hemisphere 'north' from dataset hint: %s", cand)
            return "north"
        if "south" in low:
            logger.debug("Inferred hemisphere 'south' from dataset hint: %s", cand)
            return "south"

    return None


# --- Internal helpers to reduce complexity/branching ---


def _clean_date_str(date_str: str) -> str:
    """Return date-only portion of an ISO string (strip any time part)."""
    return date_str.split("T")[0] if "T" in date_str else date_str


def _inclusive_days(start_str: str, end_str: str) -> int:
    """Return inclusive number of days between two ISO date strings."""
    start_dt = datetime.fromisoformat(_clean_date_str(start_str))
    end_dt = datetime.fromisoformat(_clean_date_str(end_str))
    return (end_dt - start_dt).days + 1


def _normalise_cadence(raw: str) -> str:
    """Normalise common cadence synonyms to canonical forms like '1d' or '1h'."""
    cad = raw.strip().lower()
    if cad in ("daily", "day"):
        return "1d"
    if cad in ("hourly", "hour"):
        return "1h"
    return cad


def _points_from_cadence(delta_days: int, cadence: str) -> int | None:
    """Compute point count from inclusive day span and normalized cadence.

    Supports day- and hour-based cadences (e.g., '1d', '2day', '3h', '12hr', '24hour').
    Returns None for unrecognized formats or non-positive periods.
    """
    cad = _normalise_cadence(cadence)

    # Day cadence
    if cad.endswith(("d", "day")):
        cleaned = cad[:-3] if cad.endswith("day") else cad[:-1]
        cleaned = cleaned.strip()
        num_days = 1
        if cleaned:
            with contextlib.suppress(ValueError):
                num_days = int(cleaned)
        if num_days <= 0:
            return None
        return max(1, delta_days // num_days)

    # Hour cadence
    if cad.endswith(("hour", "hr", "h")):
        if cad.endswith("hour"):
            cleaned = cad[:-4]
        elif cad.endswith("hr"):
            cleaned = cad[:-2]
        else:
            cleaned = cad[:-1]
        cleaned = cleaned.strip()
        num_hours = 1
        if cleaned:
            with contextlib.suppress(ValueError):
                num_hours = int(cleaned)
        if num_hours <= 0:
            return None
        delta_hours = delta_days * 24
        return max(1, delta_hours // num_hours)

    return None
