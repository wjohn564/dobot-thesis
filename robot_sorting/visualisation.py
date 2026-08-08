from pathlib import Path
from typing import List, Optional

import cv2

from robot_sorting import config as cfg
from robot_sorting.types import Detection


# This file is used to draw visualisations of the detections

def save_preview(
        image,
        detections: List[Detection],
        selected: Optional[Detection],
        preview_path: Path,
):
    """Save an annotated preview image showing the detected blocks and selected target."""

    # make sure the preview directory exists
    preview_path = Path(preview_path)
    preview_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview = image.copy()

    # get image dimensions
    img_h, img_w = preview.shape[:2]

    # Draw the masking ROI.
    cv2.rectangle(
        preview,
        (cfg.ROI_X_MIN, cfg.ROI_Y_MIN),
        (cfg.ROI_X_MAX, cfg.ROI_Y_MAX),
        (180, 180, 180),
        2,
    )

    # loop through detections and draw bounding boxes and annotations
    for detection in detections:
        # Get bounding box corners
        # NOTE: Drawing functions expect ints, so we round the coordinates
        raw_x1 = int(detection.bbox_x)
        raw_y1 = int(detection.bbox_y)
        raw_x2 = int(detection.bbox_x2)
        raw_y2 = int(detection.bbox_y2)

        # Pad the bounding box to help visualisation.
        x1 = max(
            0,
            raw_x1 - cfg.BOX_PADDING,
        )

        y1 = max(
            0,
            raw_y1 - cfg.BOX_PADDING,
        )

        x2 = min(
            img_w - 1,
            raw_x2 + cfg.BOX_PADDING,
        )

        y2 = min(
            img_h - 1,
            raw_y2 + cfg.BOX_PADDING,
        )

        u = int(round(detection.u))
        v = int(round(detection.v))

        # selected detection is highlighted
        thickness = 4 if detection is selected else 2

        # Draw bounding box
        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            thickness,
        )

        # Draw circle at centre of detection
        cv2.circle(
            preview,
            (u, v),
            8,
            (255, 255, 255),
            -1,
        )

        label = (
            f"{detection.class_name} "
            f"centre=({u},{v})"
        )

        # Add robot coordinates to the label when available.
        if (
                detection.robot_x is not None
                and detection.robot_y is not None
        ):
            label += (
                f" robot=({detection.robot_x:.1f},"
                f"{detection.robot_y:.1f})"
            )

        # Mark selected detection
        if detection is selected:
            label = "SELECTED " + label

        cv2.putText(
            preview,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    success = cv2.imwrite(
        str(preview_path),
        preview,
    )

    # Check if the image was successfully saved
    if not success:
        raise RuntimeError(
            f"Failed to save preview: {preview_path}"
        )

    print(f"Saved preview: {preview_path}")
