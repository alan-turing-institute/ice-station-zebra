import io
import os
import tempfile
from typing import Literal

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageFile

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
    format: Literal["mp4", "gif"] = "gif",
) -> bytes:
    """Save an animation to a temporary file and return bytes (with cleanup)."""
    suffix = ".gif" if format.lower() == "gif" else ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        if format.lower() == "gif":
            writer = animation.PillowWriter(fps=fps)
            anim.save(tmp_path, writer=writer, dpi=DEFAULT_DPI)
        else:
            writer = animation.FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=1800,
                extra_args=["-pix_fmt", "yuv420p"],
            )
            anim.save(tmp_path, writer=writer, dpi=DEFAULT_DPI)

        with open(tmp_path, "rb") as fh:
            return fh.read()
    except (OSError, MemoryError) as err:
        raise VideoRenderError(f"Video encoding failed: {err!s}") from err
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
