import json
from datetime import datetime
import cv2
import numpy as np
from robot_sorting import config as cfg
from robot_sorting.geometry import pixel_to_robot


def extract_calibration_points(rows):
    """Extract complete pixel and robot coordinate pairs."""

    # create lists of pixel and robot coordinates and the rows used
    pixel_points = []
    robot_points = []
    rows_used = []

    # loop over the calibration points
    for row in rows:
        if not all([
            row.get("u"),
            row.get("v"),
            row.get("robot_x"),
            row.get("robot_y"),
        ]):
            continue
        # extract the pixel and robot coordinates
        u = float(row["u"])
        v = float(row["v"])
        x = float(row["robot_x"])
        y = float(row["robot_y"])

        # store the pixel and robot coordinates and the row used
        pixel_points.append([u, v])
        robot_points.append([x, y])
        rows_used.append(row)

    return (
        np.array(pixel_points, dtype=np.float32),
        np.array(robot_points, dtype=np.float32),
        rows_used,
    )


def fit_homography(
        pixel_points,
        robot_points,
):
    """Fit a homography from pixel coordinates to robot coordinates."""
    # opencv's findHomography() method returns the homography matrix
    H, _ = cv2.findHomography(
        pixel_points,
        robot_points,
        method=0,
    )
    # error handling
    if H is None:
        raise RuntimeError(
            "Homography calculation failed."
        )

    return H


def compute_fit_errors(
        H,
        pixel_points,
        robot_points,
        rows_used,
):
    """Calculate mapping error for each fitted calibration point."""
    # list for error record
    errors = []

    for i, row in enumerate(rows_used):
        # known camera calibration point
        u, v = pixel_points[i]
        # robot coordinate of known calibration point
        true_x, true_y = robot_points[i]
        # Use fitted homography to map pixel coordinates to robot coordinates
        pred_x, pred_y = pixel_to_robot(
            H,
            u,
            v,
        )
        # calculate the error in mm
        error_mm = float(
            np.sqrt(
                (pred_x - true_x) ** 2
                + (pred_y - true_y) ** 2
            )
        )
        # append the error record
        errors.append({
            "point_id": row["point_id"],
            "u": float(u),
            "v": float(v),
            "true_x": float(true_x),
            "true_y": float(true_y),
            "pred_x": float(pred_x),
            "pred_y": float(pred_y),
            "error_mm": error_mm,
            "robot_z": row.get("robot_z", ""),
            "colour": row.get("colour", ""),
            "image_name": row.get("image_name", ""),
            "notes": row.get("notes", ""),
        })

    return errors


def compute_leave_one_out_errors(
        pixel_points,
        robot_points,
        rows_used,
):
    """Calculate leave-one-out error for each calibration point."""
    loo_errors = []

    n = len(pixel_points)

    # Leaving one point out still requires at least
    # four points to fit the remaining homography.
    if n < 5:
        return loo_errors

    for i in range(n):
        # remove the point at index i from the calibration data
        train_pixels = np.delete(
            pixel_points,
            i,
            axis=0,
        )
        # same for the robot coordinates
        train_robot = np.delete(
            robot_points,
            i,
            axis=0,
        )
        # fit the homography to the remaining points
        H_i = fit_homography(
            train_pixels,
            train_robot,
        )
        # compute the error for the point at index i
        u, v = pixel_points[i]
        true_x, true_y = robot_points[i]

        pred_x, pred_y = pixel_to_robot(
            H_i,
            u,
            v,
        )
        # calculate the error in mm
        error_mm = float(
            np.sqrt(
                (pred_x - true_x) ** 2
                + (pred_y - true_y) ** 2
            )
        )
        # append the error record
        loo_errors.append({
            "point_id": rows_used[i]["point_id"],
            "error_mm": error_mm,
            "true_x": float(true_x),
            "true_y": float(true_y),
            "pred_x": float(pred_x),
            "pred_y": float(pred_y),
        })

    return loo_errors


def summarize_errors(errors):
    """Return summary statistics for a set of mapping errors."""
    values = [
        error["error_mm"]
        for error in errors
    ]

    return {
        "mean_error_mm": float(np.mean(values)),
        "median_error_mm": float(np.median(values)),
        "max_error_mm": float(np.max(values)),
        "std_error_mm": float(np.std(values)),
    }


def print_error_table(
        title,
        errors,
):
    """Print the error for each calibration point."""
    print()
    print(title)
    print("-" * len(title))

    if not errors:
        print("Not enough points.")
        return

    values = np.array(
        [
            error["error_mm"]
            for error in errors
        ],
        dtype=float,
    )

    mean = float(values.mean())
    std = float(values.std())

    for error in errors:
        flag = ""

        if error["error_mm"] > 7.0:
            flag = "  <-- redo/check"

        elif (
                std > 0
                and error["error_mm"] > mean + 2.0 * std
        ):
            flag = "  <-- suspicious"

        print(
            f"{error['point_id']}: "
            f"true=({error['true_x']:.2f}, "
            f"{error['true_y']:.2f}) "
            f"pred=({error['pred_x']:.2f}, "
            f"{error['pred_y']:.2f}) "
            f"error={error['error_mm']:.2f} mm"
            f"{flag}"
        )


def save_homography_outputs(
        H,
        report,
):
    """Save the active homography and report, plus archived copies."""
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    # make sure the archive directory exists
    cfg.HOMOGRAPHY_ARCHIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_h_path = (
            cfg.HOMOGRAPHY_ARCHIVE_DIR
            / f"homography_pixel_to_robot_{timestamp}.npy"
    )

    archive_report_path = (
            cfg.HOMOGRAPHY_ARCHIVE_DIR
            / f"homography_report_{timestamp}.json"
    )

    np.save(
        cfg.H_PATH,
        H,
    )

    np.save(
        archive_h_path,
        H,
    )

    with open(
            cfg.CALIBRATION_REPORT_PATH,
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    with open(
            archive_report_path,
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    return (
        archive_h_path,
        archive_report_path,
    )
