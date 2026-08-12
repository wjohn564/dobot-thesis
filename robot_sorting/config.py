from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


# Shared project settings.
# Pose is frozen so it cannot be changed accidentally.
@dataclass(frozen=True)
class Pose:
    """Robot pose in Cartesian space."""

    # Cartesian coordinates are in mm; r is end-effector rotation.
    x: float
    y: float
    z: float
    r: float = 0.0


# Project root from robot_sorting/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

# Create runtime output directories.
for path in [
    RUN_DIR,
    IMAGE_DIR,
    PREVIEW_DIR,
    LOG_DIR,
]:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )

# Camera
# Camera resolution.
WIDTH = 1920
HEIGHT = 1080

# Delay before the still image is taken.
CAMERA_TIMEOUT_MS = 1000

# Masking ROI is slightly larger than the physical workspace.
ROI_X_MIN = 60
ROI_Y_MIN = 85
ROI_X_MAX = 1860
ROI_Y_MAX = 1015

# Valid block-centre region. The higher Y minimum rejects the top workspace shadow.
WORKSPACE_CENTER_X_MIN = 60
WORKSPACE_CENTER_X_MAX = 1860
WORKSPACE_CENTER_Y_MIN = 145
WORKSPACE_CENTER_Y_MAX = 1015

# OpenCV colour detection

# Contour area filter.
MIN_AREA = 800
MAX_AREA = 70000

# Bounding-box size filter.
MIN_BOX_WIDTH = 35
MIN_BOX_HEIGHT = 35
MAX_BOX_WIDTH = 350
MAX_BOX_HEIGHT = 350

# Bounding-box shape filter.
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 2.20

# Preview-only bounding-box padding.
BOX_PADDING = 8

# HSV colour ranges used by the OpenCV detector.
HSV_RANGES = {
    "red": [
        # Red wraps around the ends of OpenCV's hue scale.
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

# High travel position for sideways movement.
TRAVEL_Z = 70.0

# Local safe height above the block.
SAFE_Z = 20.0

# Block grasp height.
PICKUP_Z = -40.0

# Suction cup does not require end-effector rotation.
TARGET_R = 0.0

# Dobot timing settings in seconds

# Suction hold time before lifting.
SUCTION_GRAB_TIME = 0.8

# Settle time before suction starts.
PRE_SUCTION_SETTLE_TIME = 0.2

# Wait after releasing into a bin.
POST_RELEASE_TIME = 0.3

# Wait before the next image after a pick cycle.
AFTER_CYCLE_DELAY = 0.5

# Safety cap to stop the sorting run after 10 cycles.
MAX_CYCLES = 10

# Safe mapped pickup bounds.
X_MIN, X_MAX = 80.0, 360.0
Y_MIN, Y_MAX = -200.0, 200.0

# Camera-clear pose.
CAMERA_CLEAR_POSE: Optional[Pose] = Pose(
    x=197.0,
    y=0.0,
    z=46.0,
    r=0.0,
)

# Drop poses.
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

# Colour-to-bin mapping.
COLOUR_TO_BIN = {
    "red": "warm_bin",
    "yellow": "warm_bin",
    "blue": "cool_bin",
    "green": "cool_bin",
}

# Pick order for multi-block scenes.
PICK_COLOUR_ORDER = [
    "red",
    "yellow",
    "blue",
    "green",
]

# Enable post-pick verification.
VERIFY_AFTER_PICK = True


def ensure_runtime_config_ready():
    """Check that required robot poses are configured."""

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
