from matplotlib.axes import Axes

__test__ = False  # prevent pytest from collecting anything in this helper module

EPSILON: float = 1e-6  # small tolerance for float noise

RECTANGLE = tuple[float, float, float, float]  # (Left, Bottom, Right, Top)


def axis_rectangle(ax: Axes) -> RECTANGLE:
    """Return (left, bottom, right, top) in figure-normalised coords [0, 1]."""
    bbox = ax.get_position()
    return (bbox.x0, bbox.y0, bbox.x1, bbox.y1)


def rectangles_overlap(
    rect_a: tuple[float, float, float, float],
    rect_b: tuple[float, float, float, float],
    *,
    epsilon: float = EPSILON,
) -> bool:
    """True if two axis-aligned rectangles overlap (with tolerance)."""
    la, ba, ra, ta = rect_a
    lb, bb, rb, tb = rect_b
    separated_h = ra <= lb + epsilon or rb <= la + epsilon
    separated_v = ta <= bb + epsilon or tb <= ba + epsilon
    return not (separated_h or separated_v)


def rectangles_have_minimum_gap(
    rect_a: tuple[float, float, float, float],
    rect_b: tuple[float, float, float, float],
    *,
    min_gap: float = 0.01,
) -> bool:
    """Require ≥ min_gap separation if rectangles are ordered horizontally or vertically."""
    la, ba, ra, ta = rect_a
    lb, bb, rb, tb = rect_b

    horizontally_ordered = ra <= lb or rb <= la
    vertically_ordered = ta <= bb or tb <= ba

    if horizontally_ordered:
        gap = min(abs(lb - ra), abs(la - rb))
        return gap >= min_gap
    if vertically_ordered:
        gap = min(abs(bb - ta), abs(ba - tb))
        return gap >= min_gap
    # Otherwise they overlap (caught by rectangles_overlap tests)
    return False
