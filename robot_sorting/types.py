from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    """One detected block passed through the sorting pipeline."""

    # Detector that produced this result.
    method: str
    # Block colour.
    class_name: str
    # Model confidence; OpenCV uses 1.0 as a placeholder.
    confidence: float

    # Bounding box in image coordinates: top-left x/y plus width/height.
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int

    # Detected block centre in pixels.
    u: float
    v: float

    # OpenCV stores contour area here.
    # For learned detectors this is bounding box area (bbox_w * bbox_h).
    # Area is also available for logging and target ordering.
    area: float = 0.0

    # Robot coordinates are filled after homography mapping.
    robot_x: Optional[float] = None
    robot_y: Optional[float] = None
    # selection.py assigns the target bin.
    bin_name: Optional[str] = None

    # Convenience properties for drawing previews.
    @property
    def bbox_x2(self):
        """Calculate the right side of the bounding box."""
        return self.bbox_x + self.bbox_w

    @property
    def bbox_y2(self):
        """Calculate the bottom side of the bounding box."""
        return self.bbox_y + self.bbox_h
