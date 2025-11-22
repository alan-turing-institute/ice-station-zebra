"""Animation helper module for managing matplotlib animation lifecycle.

Provides centralised cache management to prevent garbage collection of animation
objects during save operations, avoiding warnings and ensuring proper cleanup.
"""

import contextlib

from matplotlib import animation

# Centralized cache to hold strong references to animation objects
_ANIM_CACHE: list[animation.FuncAnimation] = []


def hold_anim(anim: animation.FuncAnimation) -> None:
    """Add an animation object to the cache to prevent garbage collection.

    Call this immediately after creating a FuncAnimation object to ensure
    it remains in memory during save operations.

    Args:
        anim: The FuncAnimation object to cache.

    """
    _ANIM_CACHE.append(anim)


def release_anim(anim: animation.FuncAnimation) -> None:
    """Remove an animation object from the cache after saving is complete.

    Call this in a finally block after saving the animation to allow
    proper cleanup and garbage collection.

    Args:
        anim: The FuncAnimation object to remove from cache.

    """
    with contextlib.suppress(ValueError):
        # Animation already removed or never added - ignore
        _ANIM_CACHE.remove(anim)
