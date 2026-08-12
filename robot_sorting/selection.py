from typing import List, Optional
from robot_sorting import config as cfg
from robot_sorting.geometry import is_robot_xy_safe
from robot_sorting.types import Detection


def assign_bins(detections: List[Detection]):
    """Assign each detection to its configured bin."""
    for det in detections:
        det.bin_name = cfg.COLOUR_TO_BIN.get(det.class_name)


def choose_next_target(detections: List[Detection]) -> Optional[Detection]:
    """Choose the next target to pick up."""

    # Collect valid pick targets.
    pickable = []

    # Filter invalid targets.
    for det in detections:
        # Robot coordinates are required.
        if det.robot_x is None or det.robot_y is None:
            continue

        # A target bin is required.
        if det.bin_name is None:
            continue

        # Pickup coordinates must be safe.
        if not is_robot_xy_safe(det.robot_x, det.robot_y):
            continue

        # Keep valid targets.
        pickable.append(det)

    # Nothing can be picked.
    if not pickable:
        return None

    order_index = {colour: i for i, colour in enumerate(cfg.PICK_COLOUR_ORDER)}

    pickable.sort(
        key=lambda d: (
            order_index.get(d.class_name, 999),
            -d.area,
        )
    )

    return pickable[0]


def selected_removed_after_pick(selected: Detection, post_detections: List[Detection]) -> bool:
    """
    Return True if the selected colour is no longer detected after the pickup.
    This is used as an automatic success proxy and assumes one block per colour.
    """
    for det in post_detections:
        if det.class_name == selected.class_name:
            return False

    return True
