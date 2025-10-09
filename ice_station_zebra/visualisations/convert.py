import io
import tempfile
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.figure import Figure
from PIL import Image
from PIL.ImageFile import ImageFile

from ice_station_zebra.exceptions import VideoRenderError

DEFAULT_DPI = 200


def _image_from_figure(fig: Figure) -> ImageFile:
    """Convert a matplotlib figure to a PIL image file."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)


def _image_from_array(a: np.ndarray) -> ImageFile:
    """Convert a numpy array to a PIL image file."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(a)
    ax.axis("off")
    try:
        return _image_from_figure(fig)
    finally:
        plt.close(fig)


def _save_animation(
    anim: animation.FuncAnimation,
    *,
    fps: int | None = None,
    video_format: Literal["mp4", "gif"] = "gif",
    _fps: int | None = None,
    _video_format: Literal["mp4", "gif"] | None = None,
) -> io.BytesIO:
    """Save an animation to a temporary file and return BytesIO (with cleanup)."""
    # Accept both standard and underscored names for test compatibility
    fps_value: int = int(fps if fps is not None else (_fps if _fps is not None else 2))
    if _video_format is not None:
        video_format = _video_format  # prefer underscored override if provided
    suffix = ".gif" if video_format.lower() == "gif" else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        try:
            # Save video to tempfile
            writer: animation.AbstractMovieWriter = (
                animation.PillowWriter(fps=fps_value)
                if suffix == ".gif"
                else animation.FFMpegWriter(
                    fps=fps_value,
                    codec="libx264",
                    bitrate=1800,
                    # Ensure dimensions are compatible with yuv420p (even width/height)
                    # by applying a scale filter that truncates to the nearest even integers.
                    extra_args=[
                        "-pix_fmt",
                        "yuv420p",
                        "-vf",
                        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    ],
                )
            )
            anim.save(tmp.name, writer=writer, dpi=DEFAULT_DPI)
            # Load tempfile into a BytesIO buffer
            with Path(tmp.name).open("rb") as fh:
                buffer = io.BytesIO(fh.read())
        except (OSError, MemoryError) as err:
            msg = f"Video encoding failed: {err!s}"
            raise VideoRenderError(msg) from err
    buffer.seek(0)
    return buffer
