from typing import List

import cv2
import numpy as np

from robot_sorting import config as cfg
from robot_sorting.types import Detection


def load_homography():
    """Load the homography matrix from file."""
    if not cfg.H_PATH.exists():
        raise FileNotFoundError(f"Missing homography file: {cfg.H_PATH}")

    return np.load(cfg.H_PATH)


def pixel_to_robot(H, u, v):
    """Convert pixel coordinates to robot coordinates."""
    pt = np.array([[[u, v]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)
    x, y = mapped[0][0]
    return float(x), float(y)


def add_robot_coordinates(detections: List[Detection], H):
    """Add robot coordinates to detections."""
    for det in detections:
        x, y = pixel_to_robot(H, det.u, det.v)
        det.robot_x = x
        det.robot_y = y


def is_robot_xy_safe(x, y):
    """Check if a mapped pickup coordinate is within the safe workspace."""
    return cfg.X_MIN <= x <= cfg.X_MAX and cfg.Y_MIN <= y <= cfg.Y_MAX


def validate_robot_xy(x, y):
    """Stop execution if a mapped pickup coordinate is outside the safe workspace."""
    if not (cfg.X_MIN <= x <= cfg.X_MAX):
        raise RuntimeError(f"Mapped X={x:.3f} outside safe bounds [{cfg.X_MIN}, {cfg.X_MAX}]")

    if not (cfg.Y_MIN <= y <= cfg.Y_MAX):
        raise RuntimeError(f"Mapped Y={y:.3f} outside safe bounds [{cfg.Y_MIN}, {cfg.Y_MAX}]")
