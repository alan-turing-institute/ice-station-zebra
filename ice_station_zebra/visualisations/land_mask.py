import logging
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


class LandMask:
    def __init__(
        self, base_path: Path | None, hemisphere: Literal["north", "south"]
    ) -> None:
        """A helper class to apply land masks to data arrays."""
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        if base_path:
            search = f"data/preprocessing/*/IceNetSIC/data/masks/{hemisphere}/masks/land_mask.npy"
            for mask_path in base_path.glob(search):
                try:
                    self.add_mask(np.load(mask_path))
                except (OSError, ValueError):
                    continue

    def add_mask(self, mask_array: np.ndarray) -> None:
        """Add a land mask to the cache, keyed by its shape."""
        self._cache[mask_array.shape] = mask_array.astype(bool)

    def apply_to(self, data_array: np.ndarray) -> np.ndarray:
        """Apply a land mask to an array."""
        hw = data_array.shape[-2:]
        if hw not in self._cache:
            logger.warning("No land mask available for shape %s.", hw)
            self._cache[hw] = np.empty(hw, dtype=bool)
        return np.where(self._cache[hw], np.nan, data_array)
