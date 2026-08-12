import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from robot_sorting import config as cfg


def find_camera_command() -> str:
    """Return the available Raspberry Pi camera command."""

    # Try both Raspberry Pi camera command names.
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
        prefix: str,
        # Defaults to the runtime image directory.
        output_dir: Optional[Path] = None,
        # Optional camera command log directory.
        capture_log_dir: Optional[Path] = None,

) -> Tuple[Path, np.ndarray]:
    if output_dir is None:
        # Use the runtime image directory by default.
        output_dir = cfg.IMAGE_DIR
    # Normalise the output path.
    output_dir = Path(output_dir)
    # Create the output directory if needed.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp keeps filenames unique.
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    # Build the image filename.
    image_path = output_dir / f"{prefix}_{timestamp}.jpg"

    # Find the available camera command.
    command = find_camera_command()

    # Build the still-image command.
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

    # Run the camera command.
    try:
        # Capture camera output only when a log directory is supplied.
        if capture_log_dir is None:
            subprocess.run(
                cmd,
                check=True,
            )
        else:
            # Create the capture log directory.
            capture_log_dir = Path(capture_log_dir)
            capture_log_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Build the log filename.
            log_path = (
                    capture_log_dir
                    / f"{image_path.stem}_capture.log"
            )

            # Save camera output to the log file.
            with open(
                    log_path,
                    "w",
                    encoding="utf-8",
            ) as log_file:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )


            print(f"Capture log: {log_path}")

    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Camera capture failed with exit code "
            f"{exc.returncode}."
        ) from exc

    # Check that the image was created.
    if not image_path.exists():
        raise FileNotFoundError(
            f"Camera did not create: {image_path}"
        )
    # Load the captured image.
    image = cv2.imread(str(image_path))

    # Check that OpenCV loaded it.
    if image is None:
        raise RuntimeError(
            f"OpenCV could not read: {image_path}"
        )

    return image_path, image
