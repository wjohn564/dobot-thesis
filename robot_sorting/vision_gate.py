from typing import Tuple

from robot_sorting import config as cfg


# Final checks that a detection is physically plausible.

def bbox_centre(
        x: float,
        y: float,
        w: float,
        h: float,
) -> Tuple[float, float]:
    """Calculate the bounding box center point."""
    u = x + w / 2.0
    v = y + h / 2.0
    return u, v


def passes_workspace_gate(
        u: float,
        v: float,
) -> Tuple[bool, str]:
    """Check whether the detection center is inside the valid workspace region."""

    if u < cfg.WORKSPACE_CENTER_X_MIN:
        return False, f"centre left of workspace: u={u:.1f}"

    if u > cfg.WORKSPACE_CENTER_X_MAX:
        return False, f"centre right of workspace: u={u:.1f}"

    if v < cfg.WORKSPACE_CENTER_Y_MIN:
        return False, f"centre above workspace: v={v:.1f}"

    if v > cfg.WORKSPACE_CENTER_Y_MAX:
        return False, f"centre below workspace: v={v:.1f}"

    return True, "ok"


def passes_opencv_gate(
        x: int,
        y: int,
        w: int,
        h: int,
        area: float,
) -> Tuple[bool, str]:
    """Check OpenCV contour size, shape and workspace position."""

    if area < cfg.MIN_AREA:
        return False, f"area too small: {area:.1f}"

    if area > cfg.MAX_AREA:
        return False, f"area too large: {area:.1f}"

    if w < cfg.MIN_BOX_WIDTH:
        return False, f"box too narrow: w={w}"

    if h < cfg.MIN_BOX_HEIGHT:
        return False, f"box too short: h={h}"

    if w > cfg.MAX_BOX_WIDTH:
        return False, f"box too wide: w={w}"

    if h > cfg.MAX_BOX_HEIGHT:
        return False, f"box too tall: h={h}"

    aspect_ratio = w / float(h)

    if aspect_ratio < cfg.MIN_ASPECT_RATIO:
        return False, f"aspect ratio too narrow: {aspect_ratio:.3f}"

    if aspect_ratio > cfg.MAX_ASPECT_RATIO:
        return False, f"aspect ratio too wide: {aspect_ratio:.3f}"

    u, v = bbox_centre(x, y, w, h)
    return passes_workspace_gate(u, v)
