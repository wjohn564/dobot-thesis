from typing import List, Optional
from robot_sorting import config as cfg
from robot_sorting.geometry import is_robot_xy_safe
from robot_sorting.types import Detection


def assign_bins(detections: List[Detection]):
    """Receives all currently detected blocks and assigns them to bins."""
    for det in detections:
        det.bin_name = cfg.COLOUR_TO_BIN.get(det.class_name)


def choose_next_target(detections: List[Detection]) -> Optional[Detection]:
    """Choose the next target to pick up."""

    # Store pickable targets
    pickable = []

    # filters
    for det in detections:
        # skip if we don't have a robot position
        if det.robot_x is None or det.robot_y is None:
            continue

        # skip if we don't have a bin
        if det.bin_name is None:
            continue

        # check if the coordinates are safely in the workspace
        if not is_robot_xy_safe(det.robot_x, det.robot_y):
            continue

        # if all the tests pass, add the detection to the list of pickable targets
        pickable.append(det)

    # if no targets are pickable, return None
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
