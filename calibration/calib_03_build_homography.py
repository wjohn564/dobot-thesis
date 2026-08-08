import sys
from datetime import datetime
from pathlib import Path

# Add project root so project modules can be imported
# when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.calibration_data import read_rows
from calibration.homography_tools import (
    compute_fit_errors,
    compute_leave_one_out_errors,
    extract_calibration_points,
    fit_homography,
    print_error_table,
    save_homography_outputs,
    summarize_errors,
)
from robot_sorting import config as cfg


def main():
    """Build and evaluate the pixel-to-robot homography."""

    # Make sure CSV for calibration points exists
    if not cfg.CALIBRATION_POINTS_CSV.exists():
        raise FileNotFoundError(
            f"Missing calibration CSV: "
            f"{cfg.CALIBRATION_POINTS_CSV}"
        )

    # Load calibration points from CSV
    rows = read_rows()

    (
        pixel_points,
        robot_points,
        rows_used,
    ) = extract_calibration_points(rows)

    print(
        f"Loaded calibration pairs: "
        f"{len(pixel_points)}"
    )
    # Ensure we have enough points to fit a homography
    if len(pixel_points) < 4:
        raise RuntimeError(
            "Need at least 4 calibration points. "
            "Use 9 minimum, 16 recommended."
        )
    # throw a warning if less than 9
    if len(pixel_points) < 9:
        print(
            "WARNING: fewer than 9 points. "
            "This can work, but it is not ideal."
        )

    # use the function from homography_tools.py to fit the homography
    H = fit_homography(
        pixel_points,
        robot_points,
    )
    # compute the errors for each fitted calibration point
    fit_errors = compute_fit_errors(
        H,
        pixel_points,
        robot_points,
        rows_used,
    )
    # compute the errors for each leave-one-out calibration point
    loo_errors = compute_leave_one_out_errors(
        pixel_points,
        robot_points,
        rows_used,
    )
    # summarize the errors
    fit_summary = summarize_errors(
        fit_errors
    )
    # summarize the leave-one-out errors
    loo_summary = (
        summarize_errors(loo_errors)
        if loo_errors
        else {}
    )
    # Print the error reports
    print_error_table(
        "FIT ERROR REPORT",
        fit_errors,
    )

    print_error_table(
        "LEAVE-ONE-OUT ERROR REPORT",
        loo_errors,
    )
    # Build the calibration report
    report = {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "source_csv": str(
            cfg.CALIBRATION_POINTS_CSV
        ),
        "homography_matrix": H.tolist(),
        "num_points": int(
            len(pixel_points)
        ),
        "fit_summary": fit_summary,
        "leave_one_out_summary": loo_summary,
        "fit_errors": fit_errors,
        "leave_one_out_errors": loo_errors,
        "notes": (
            "Homography maps image pixel centre "
            "(u, v) to Dobot Cartesian coordinates "
            "(x, y). R is fixed during movement and "
            "is not part of calibration."
        ),
    }

    ( # Save the active calibration outputs and archive copies
        archive_h_path,
        archive_report_path,
    ) = save_homography_outputs(
        H,
        report,
    )

    print()
    print("DONE")
    print(
        f"Saved active homography: "
        f"{cfg.H_PATH}"
    )
    print(
        f"Saved active report:     "
        f"{cfg.CALIBRATION_REPORT_PATH}"
    )
    print(
        f"Archived homography:     "
        f"{archive_h_path}"
    )
    print(
        f"Archived report:         "
        f"{archive_report_path}"
    )

    print()
    print("SUMMARY")
    print(
        f"Fit mean error:    "
        f"{fit_summary['mean_error_mm']:.2f} mm"
    )
    print(
        f"Fit median error:  "
        f"{fit_summary['median_error_mm']:.2f} mm"
    )
    print(
        f"Fit max error:     "
        f"{fit_summary['max_error_mm']:.2f} mm"
    )

    if loo_summary:
        print(
            f"LOO mean error:    "
            f"{loo_summary['mean_error_mm']:.2f} mm"
        )
        print(
            f"LOO median error:  "
            f"{loo_summary['median_error_mm']:.2f} mm"
        )
        print(
            f"LOO max error:     "
            f"{loo_summary['max_error_mm']:.2f} mm"
        )

    print()
    print("Homography matrix:")
    print(H)


if __name__ == "__main__":
    main()
