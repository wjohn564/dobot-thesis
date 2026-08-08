from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


# Config just stores values that other files use
# Frozen to prevent it from being changed
@dataclass(frozen=True)
class Pose:
    """This class is used to represent a single robot pose in 3D space."""

    # x, y and z are Cartesian robot coordinates in mm.
    # r is the rotation of the end effector.
    x: float
    y: float
    z: float
    r: float = 0.0


# The file paths for the raspberry pi project directory
PROJECT_ROOT = Path("/home/john/dobot-thesis")

RUN_DIR = PROJECT_ROOT / "runs"
IMAGE_DIR = RUN_DIR / "images"
PREVIEW_DIR = RUN_DIR / "previews"
LOG_DIR = RUN_DIR / "logs"

# Calibration paths
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
CALIBRATION_DATA_DIR = CALIBRATION_DIR / "data"
CALIBRATION_IMAGE_DIR = CALIBRATION_DIR / "images"
CALIBRATION_PREVIEW_DIR = CALIBRATION_DIR / "previews"
CALIBRATION_CAPTURE_LOG_DIR = CALIBRATION_DIR / "capture_logs"

CALIBRATION_POINTS_CSV = (
        CALIBRATION_DATA_DIR / "calibration_points.csv"
)

H_PATH = (
        CALIBRATION_DATA_DIR
        / "homography_pixel_to_robot.npy"
)

CALIBRATION_REPORT_PATH = (
        CALIBRATION_DATA_DIR
        / "homography_report.json"
)

HOMOGRAPHY_ARCHIVE_DIR = (
        CALIBRATION_DIR
        / "archive"
        / "homographies"
)

# The calibration jog settings
# Calibration jogging
CALIBRATION_DEFAULT_STEP_MM = 1.0
CALIBRATION_Z_STEP_MM = 1.0

CALIBRATION_JOG_X_MIN = 40.0
CALIBRATION_JOG_X_MAX = 360.0

CALIBRATION_JOG_Y_MIN = -260.0
CALIBRATION_JOG_Y_MAX = 260.0

CALIBRATION_JOG_Z_MIN = -80.0
CALIBRATION_JOG_Z_MAX = 160.0

CALIBRATION_SETTLE_SECONDS = 0.8
CALIBRATION_POSE_SAMPLES = 7
CALIBRATION_POSE_SAMPLE_DELAY = 0.08

# Create the runtime output directories if they don't already exist.
for path in [
    RUN_DIR,
    IMAGE_DIR,
    PREVIEW_DIR,
    LOG_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Camera
# resolution is 1920x1080
WIDTH = 1920
HEIGHT = 1080

# capture delay/time period before the still pic is taken.
CAMERA_TIMEOUT_MS = 1000

# Masking Region of Interest. This is deliberately slightly larger than the physical
# workspace so blocks near the boundary are not truncated before detection.
ROI_X_MIN = 60
ROI_Y_MIN = 85
ROI_X_MAX = 1860
ROI_Y_MAX = 1015

# Valid block center region in the original 1920x1080 camera image.
# This is separate from the masking ROI. A contour may extend outside the
# physical workspace, but its center must be inside this valid region.
# This was done to combat the false detections of the shadow cast at the top of the workspace.
# The top shadow had centers around v=124–136. Real top row block
# centers were around v=162, so 145 removes the false blob without excluding
# valid calibration positions.
WORKSPACE_CENTER_X_MIN = 60
WORKSPACE_CENTER_X_MAX = 1860
WORKSPACE_CENTER_Y_MIN = 145
WORKSPACE_CENTER_Y_MAX = 1015

# OpenCV colour detection

# remove contours that are noise
MIN_AREA = 800
MAX_AREA = 70000

# Set bounding box limits as another noise filter
MIN_BOX_WIDTH = 35
MIN_BOX_HEIGHT = 35
MAX_BOX_WIDTH = 350
MAX_BOX_HEIGHT = 350

# Another noise filter to reject things not likely to be blocks
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 2.20

# Add 8 pixels of padding when drawing the object bounding box.
# This is only for visualisation and does not change the stored bounding box or center point.
BOX_PADDING = 8

# Hue, Saturation, Value.
# Hue is what colour, Saturation is How colourful / intense, Value is how bright
# detectors.py creates both masks and combines them to get the complete mask.
HSV_RANGES = {
    "red": [
        # Weird thing with red where it wraps around the ends of OpenCV's hue scale.
        # So need to split it into two ranges because red occurs near both ends of OpenCV's hue range.
        ((0, 100, 50), (12, 255, 255)),
        ((165, 100, 50), (180, 255, 255)),
    ],
    "blue": [
        ((90, 100, 50), (135, 255, 255)),
    ],
    "yellow": [
        ((17, 90, 80), (45, 255, 255)),
    ],
    "green": [
        ((38, 70, 40), (95, 255, 255)),
    ],
}

# Dobot

# Serial port the Pi uses to talk to the Dobot.
DEFAULT_PORT = "/dev/ttyACM0"

# Vertical operating levels for the dobot
# This is a higher height used when moving sideways between different regions.
TRAVEL_Z = 70.0

# This is a local safe height above the block.
SAFE_Z = 20.0

# Height for grasping block
PICKUP_Z = -40.0

# I am not rotating the end effector, so this is 0.
# Using the suction cup as the end effector means pinpointing the middle of blocks
# I didn't see any value to rotating the end effector.
TARGET_R = 0.0

# Dobot timing settings in seconds

# Time it waits before lifting the make sure suction gets a successful grasp
SUCTION_GRAB_TIME = 0.8

# Time it waits before turning on suction whilst on the block
PRE_SUCTION_SETTLE_TIME = 0.2

# After dropping into the bin wait this amount of time before moving
POST_RELEASE_TIME = 0.3

# After completing one pick and returning to the camera-clear position, wait before capturing the next image.
AFTER_CYCLE_DELAY = 0.5

# Safety cap to stop the sorting run after 10 cycles.
MAX_CYCLES = 10

# Define what is acceptable mapped pickup positions. Prevents bad coordinates that could overextend the robot joints.
X_MIN, X_MAX = 80.0, 360.0
Y_MIN, Y_MAX = -200.0, 200.0

# Using Pose classes for all
# This is the position the arm moves to when you want the camera to see the workspace.
CAMERA_CLEAR_POSE: Optional[Pose] = Pose(
    x=197.0,
    y=0.0,
    z=46.0,
    r=0.0,
)

# Dictionary for warm bin pose and cool bin pose
DROP_BINS: Dict[str, Optional[Pose]] = {
    "warm_bin": Pose(
        x=277.840,
        y=214.206,
        z=-55.0,
        r=0.0,
    ),
    "cool_bin": Pose(
        x=291.838,
        y=-205.066,
        z=-43.0,
        r=0.0,
    ),
}

# Mapping what colour goes in what bin.
COLOUR_TO_BIN = {
    "red": "warm_bin",
    "yellow": "warm_bin",
    "blue": "cool_bin",
    "green": "cool_bin",
}

# The order in which blocks are grasped if there is a multiblock workspace.
PICK_COLOUR_ORDER = [
    "red",
    "yellow",
    "blue",
    "green",
]

# Flag that controls if the program keeps capturing images after a pickup to verify it was successful
VERIFY_AFTER_PICK = True


def ensure_runtime_config_ready():
    """Checks that essential robot positions have actually been configured before sorting starts."""
    if CAMERA_CLEAR_POSE is None:
        raise RuntimeError(
            "CAMERA_CLEAR_POSE is not configured in "
            "robot_sorting/config.py."
        )

    for bin_name, pose in DROP_BINS.items():
        if pose is None:
            raise RuntimeError(
                f"DROP_BINS['{bin_name}'] is not configured in "
                "robot_sorting/config.py."
            )
