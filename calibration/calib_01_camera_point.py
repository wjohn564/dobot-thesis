import sys
from datetime import datetime
from pathlib import Path

# Add the project root so project modules can be imported
# when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.calibration_data import (
    next_point_id,
    read_rows,
    write_rows,
)
from robot_sorting import config as cfg
from robot_sorting.camera import capture_image
from robot_sorting.detectors import OpenCVColourDetector
from robot_sorting.visualisation import save_preview


def ensure_directories():
    """Make sure the calibration output directories exist."""
    for path in [
        cfg.CALIBRATION_IMAGE_DIR,
        cfg.CALIBRATION_PREVIEW_DIR,
        cfg.CALIBRATION_DATA_DIR,
        cfg.CALIBRATION_CAPTURE_LOG_DIR,
    ]:
        # Create this directory if it doesn't already exist
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def print_detection(detection):
    """Print the details of one detected calibration block."""
    print(
        f"{detection.class_name}: "
        f"centre=({detection.u:.1f}, "
        f"{detection.v:.1f}) "
        f"bbox=({detection.bbox_x}, "
        f"{detection.bbox_y}, "
        f"{detection.bbox_w}, "
        f"{detection.bbox_h}) "
        f"area={detection.area:.1f}"
    )


def main():
    ensure_directories()
    # get the point ID
    point_id = next_point_id()

    print()
    print("CALIBRATION CAMERA POINT")
    print(f"Point: {point_id}")
    print()

    # Capture the calibration image.
    image_path, image = capture_image(
        prefix=point_id,
        output_dir=cfg.CALIBRATION_IMAGE_DIR,
        capture_log_dir=cfg.CALIBRATION_CAPTURE_LOG_DIR,
    )

    # Detect the calibration block.
    detector = OpenCVColourDetector()

    detections, inference_time_ms = detector.detect(
        image
    )

    # Create the matching preview path.
    preview_path = (
            cfg.CALIBRATION_PREVIEW_DIR
            / f"{image_path.stem}_preview.jpg"
    )

    # Calibration requires exactly one valid block.
    # If zero or multiple detections survive, reject the point.
    if len(detections) != 1:
        save_preview(
            image=image,
            detections=detections,
            selected=None,
            preview_path=preview_path,
        )

        print()
        print("CALIBRATION POINT REJECTED")
        print(
            f"Expected exactly one valid block, "
            f"but found {len(detections)}."
        )

        for detection in detections:
            print_detection(detection)

        print()
        print("No calibration CSV row was created.")
        print(f"Preview: {preview_path}")
        return

    # There is exactly one detection, so use it
    # as the camera calibration point.
    selected = detections[0]

    # Save an annotated preview of the accepted point.
    save_preview(
        image=image,
        detections=detections,
        selected=selected,
        preview_path=preview_path,
    )

    # Read the existing calibration points.
    rows = read_rows()

    # Add the camera half of the new calibration point.
    # Robot coordinates are left blank until calib_02 is run.
    rows.append({
        "point_id": point_id,
        "u": f"{selected.u:.3f}",
        "v": f"{selected.v:.3f}",
        "robot_x": "",
        "robot_y": "",
        "robot_z": "",
        "colour": selected.class_name,
        "bbox_x": str(selected.bbox_x),
        "bbox_y": str(selected.bbox_y),
        "bbox_w": str(selected.bbox_w),
        "bbox_h": str(selected.bbox_h),
        "area": f"{selected.area:.3f}",
        "image_name": image_path.name,
        "preview_name": preview_path.name,
        "camera_time": datetime.now().isoformat(
            timespec="seconds"
        ),
        "robot_time": "",
        "notes": "",
    })

    # Save the updated calibration CSV.
    write_rows(rows)

    print()
    print("CAMERA POINT SAVED")
    print(f"point_id = {point_id}")
    print(f"colour   = {selected.class_name}")
    print(f"u        = {selected.u:.3f}")
    print(f"v        = {selected.v:.3f}")
    print(
        f"bbox     = "
        f"x={selected.bbox_x}, "
        f"y={selected.bbox_y}, "
        f"w={selected.bbox_w}, "
        f"h={selected.bbox_h}"
    )
    print(
        f"inference time = "
        f"{inference_time_ms:.3f} ms"
    )

    print()
    print(f"Image:   {image_path}")
    print(f"Preview: {preview_path}")
    print(f"CSV:     {cfg.CALIBRATION_POINTS_CSV}")

    print()
    print(
        "Now run calib_02_robot_point.py and "
        "jog the suction cup centre over the block."
    )


if __name__ == "__main__":
    main()
