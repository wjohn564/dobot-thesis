import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from robot_sorting import config as cfg


def find_camera_command() -> str:
    """Check for the Raspberry Pi camera command and return it."""

    # I am trying two commands incase.
    for command in [
        "rpicam-still",
        "libcamera-still",
    ]:
        if shutil.which(command):
            return command

    raise FileNotFoundError(
        "Neither rpicam-still nor libcamera-still was found."
    )


def capture_image(
        # filename prefix
        prefix: str,
        # output directory, defaults to the runtime image directory.
        output_dir: Optional[Path] = None,
        # Control whether the text printed by the raspberry pi camera command is captured and saved to log file
        capture_log_dir: Optional[Path] = None,

        # Return image_path and OpenCV image
) -> Tuple[Path, np.ndarray]:
    if output_dir is None:
        # set image directory if none is set
        output_dir = cfg.IMAGE_DIR
    # output directory must be a Path object
    output_dir = Path(output_dir)
    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a timestamp
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    # Add the prefix and timestamp to the filename
    image_path = output_dir / f"{prefix}_{timestamp}.jpg"

    # Use the find camera command to find the raspberry pi camera command
    command = find_camera_command()

    # Build a command to capture an image using the raspberry pi camera command with the specified parameters.
    cmd = [
        command,
        "--nopreview",
        "-o",
        str(image_path),
        "--width",
        str(cfg.WIDTH),
        "--height",
        str(cfg.HEIGHT),
        "-t",
        str(cfg.CAMERA_TIMEOUT_MS),
    ]

    print(f"Capturing image: {image_path}")

    # Run the command and check for errors.
    try:
        # check if capture_log_dir is None, if it is None, run the command without capturing the log, otherwise capture the log
        if capture_log_dir is None:
            subprocess.run(
                cmd,
                # This will raise a CalledProcessError if the command fails.
                check=True,
            )
        else:
            # Create the capture log directory if it doesn't exist.
            capture_log_dir = Path(capture_log_dir)
            # create the capture log directory if it doesn't exist.
            capture_log_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # generate log filename
            log_path = (
                    capture_log_dir
                    / f"{image_path.stem}_capture.log"
            )

            # open log file
            with open(
                    log_path,
                    # write mode
                    "w",
                    encoding="utf-8",
            ) as log_file:
                subprocess.run(
                    cmd,
                    check=True,
                    # capture the log to the log file
                    stdout=log_file,
                    # put error message in the log file aswell
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            # Handle a failed camera command.

            print(f"Capture log: {log_path}")

    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Camera capture failed with exit code "
            f"{exc.returncode}."
        ) from exc

    # Check if image was created
    if not image_path.exists():
        raise FileNotFoundError(
            f"Camera did not create: {image_path}"
        )
    # Load the image using OpenCV
    image = cv2.imread(str(image_path))

    # Check if image was loaded
    if image is None:
        raise RuntimeError(
            f"OpenCV could not read: {image_path}"
        )

    return image_path, image
