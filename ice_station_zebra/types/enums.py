from enum import IntEnum, StrEnum


class BetaSchedule(StrEnum):
    """Enum for diffusion beta schedule types."""

    LINEAR = "linear"
    COSINE = "cosine"


class TensorDimensions(IntEnum):
    """Enum for tensor dimensions."""

    THW = 3  # Time, Height, Width
    BTCHW = 5  # Batch, Time, Channels, Height, Width
