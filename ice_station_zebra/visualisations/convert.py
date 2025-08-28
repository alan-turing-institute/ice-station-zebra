import io
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.figure import Figure
from PIL import Image
from PIL.ImageFile import ImageFile

from .plotting_core import VideoRenderError

DEFAULT_DPI = 200


def _image_from_figure(fig: Figure) -> ImageFile:
    """Convert a matplotlib figure to a PIL image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)  # type: ignore[return-value]


def _image_from_array(a: np.ndarray) -> ImageFile:
    """(Optional spare) Convert a single array to an image — not used, kept for future."""
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
    fps: int,
    video_format: Literal["mp4", "gif"] = "gif",
) -> io.BytesIO:
    """Save an animation to a temporary file and return BytesIO (with cleanup)."""
    suffix = ".gif" if video_format.lower() == "gif" else ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        if video_format.lower() == "gif":
            writer = animation.PillowWriter(fps=fps)
            anim.save(tmp_path, writer=writer, dpi=DEFAULT_DPI)
        else:
            ffmpeg_writer = animation.FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=1800,
                extra_args=["-pix_fmt", "yuv420p"],
            )
            anim.save(tmp_path, writer=ffmpeg_writer, dpi=DEFAULT_DPI)

        with Path(tmp_path).open("rb") as fh:
            buffer = io.BytesIO(fh.read())
            buffer.seek(0)  # Reset to beginning
            return buffer
    except (OSError, MemoryError) as err:
        msg = f"Video encoding failed: {err!s}"
        raise VideoRenderError(msg) from err
    finally:
        if tmp_path and Path(tmp_path).exists():
            with suppress(OSError):
                Path(tmp_path).unlink()
