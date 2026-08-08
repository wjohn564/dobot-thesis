from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    """
     Class to define the standard object (One detected block) that carries one detected block through the pipeline.
     If the camera detects four blocks the detector will return four Detection objects.
     Each object contains all information currently known about the detected block.
    """

    # which detector produced this detection (eg Yolo, opencv, etc)
    method: str
    # block colour
    class_name: str
    # model confidence (Note: OpenCV currently 1.0 as a fixed placeholder, it doesn't produce a learned confidence score.)
    confidence: float

    # bounding box coordinates
    # Image coordinate system starts in the top left
    # bbox_x is the horizontal position of the top left corner of the bounding box
    # bbox_y is the vertical position of the top left corner of the bounding box
    # bbox_w is the width of the bounding box
    # bbox_h is the height of the bounding box
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int

    # Center point coordinate of the detected block
    u: float
    v: float

    # OpenCV stores contour area here.
    # For YOLO this will be bounding box area (bbox_w * bbox_h).
    # Area is also available for logging and target ordering.
    area: float = 0.0

    # Optional fields allow types to be none when not available. The detector only knows the camera coordinates initially.
    # After the Homography (Geometry.py) is applied the robot coordinates are available.
    robot_x: Optional[float] = None
    robot_y: Optional[float] = None
    # this class does not decide which bin (warm or cool) the block should be dropped into. selection.py assigns the bin
    # with the mapping in the config.py after the detector states the colour.
    bin_name: Optional[str] = None

    # Note: visualisation.py will use these properties below when drawing rectangles
    @property
    def bbox_x2(self):
        """Calculate the right side of the bounding box."""
        return self.bbox_x + self.bbox_w

    @property
    def bbox_y2(self):
        """Calculate the bottom side of the bounding box."""
        return self.bbox_y + self.bbox_h
