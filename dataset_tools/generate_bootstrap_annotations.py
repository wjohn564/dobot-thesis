import csv
import re
import shutil
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow imports from the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_sorting import config as cfg
from robot_sorting.detectors import OpenCVColourDetector


SOURCE_ROOT = PROJECT_ROOT / "dataset_raw"
OUTPUT_ROOT = PROJECT_ROOT / "dataset_annotated_bootstrap"
PREVIEW_ROOT = PROJECT_ROOT / "dataset_annotation_previews"

REPORT_PATH = OUTPUT_ROOT / "annotation_report.csv"
WARNINGS_PATH = OUTPUT_ROOT / "annotation_warnings.txt"

DATASET_GROUPS = [
    "01_single_blocks",
    "02_two_blocks",
    "03_three_blocks",
    "04_four_blocks",
    "05_empty_workspace",
]

CLASS_MAP = {
    "red": 0,
    "blue": 1,
    "yellow": 2,
    "green": 3,
}


def expected_colours_from_path(
        image_path: Path,
):
    """Read the expected block colours from the dataset path."""

    relative_text = str(
        image_path.relative_to(SOURCE_ROOT)
    ).lower()

    if "empty_workspace" in relative_text:
        return []

    expected = []

    for colour in [
        "red",
        "blue",
        "yellow",
        "green",
    ]:
        if re.search(
                rf"(^|[_\\/.\-])"
                rf"{colour}"
                rf"($|[_\\/.\-])",
                relative_text,
        ):
            expected.append(colour)

        elif f"{colour}_cube" in relative_text:
            expected.append(colour)

    # Remove duplicates without changing the order.
    result = []
    seen = set()

    for colour in expected:
        if colour not in seen:
            result.append(colour)
            seen.add(colour)

    return result


def pad_box(
        x: int,
        y: int,
        w: int,
        h: int,
        image_width: int,
        image_height: int,
):
    """Pad a box and clip it to the image."""

    x1 = max(
        0,
        x - cfg.BOX_PADDING,
    )

    y1 = max(
        0,
        y - cfg.BOX_PADDING,
    )

    x2 = min(
        image_width,
        x + w + cfg.BOX_PADDING,
    )

    y2 = min(
        image_height,
        y + h + cfg.BOX_PADDING,
    )

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )


def find_best_annotation_box(
        hsv,
        colour: str,
        image_width: int,
        image_height: int,
):
    """Find the largest valid box for one expected colour."""

    mask = OpenCVColourDetector._create_mask(
        hsv,
        colour,
    )

    mask = OpenCVColourDetector._apply_roi(
        mask
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:

        area = float(
            cv2.contourArea(contour)
        )

        if area < cfg.MIN_AREA:
            continue

        if area > cfg.MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < cfg.MIN_BOX_WIDTH:
            continue

        if h < cfg.MIN_BOX_HEIGHT:
            continue

        if w > cfg.MAX_BOX_WIDTH:
            continue

        if h > cfg.MAX_BOX_HEIGHT:
            continue

        aspect_ratio = w / float(h)

        if aspect_ratio < cfg.MIN_ASPECT_RATIO:
            continue

        if aspect_ratio > cfg.MAX_ASPECT_RATIO:
            continue

        # Padding is included in the saved annotation.
        x, y, w, h = pad_box(
            x=x,
            y=y,
            w=w,
            h=h,
            image_width=image_width,
            image_height=image_height,
        )

        candidates.append(
            (
                area,
                x,
                y,
                w,
                h,
            )
        )

    if not candidates:
        return None

    # The dataset path already tells us which colour to expect.
    candidates.sort(
        key=lambda candidate: candidate[0],
        reverse=True,
    )

    _, x, y, w, h = candidates[0]

    return x, y, w, h


def to_yolo(
        class_id: int,
        x: int,
        y: int,
        w: int,
        h: int,
        image_width: int,
        image_height: int,
):
    """Convert a pixel box to normalised YOLO format."""

    x_center = (
                       x + w / 2.0
               ) / image_width

    y_center = (
                       y + h / 2.0
               ) / image_height

    width = w / image_width
    height = h / image_height

    return (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{width:.6f} "
        f"{height:.6f}"
    )


def draw_preview_box(
        preview,
        x: int,
        y: int,
        w: int,
        h: int,
        colour: str,
):
    """Draw one generated annotation on a preview image."""

    cv2.rectangle(
        preview,
        (x, y),
        (x + w, y + h),
        (255, 255, 255),
        2,
    )

    u = int(
        round(x + w / 2.0)
    )

    v = int(
        round(y + h / 2.0)
    )

    cv2.circle(
        preview,
        (u, v),
        6,
        (255, 255, 255),
        -1,
    )

    cv2.putText(
        preview,
        f"{colour} centre=({u},{v})",
        (x, max(25, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def clear_outputs():
    """Clear generated annotations and previews."""

    if OUTPUT_ROOT.exists():
        shutil.rmtree(
            OUTPUT_ROOT
        )

    if PREVIEW_ROOT.exists():
        shutil.rmtree(
            PREVIEW_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREVIEW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


def find_images():
    """Find images from the configured dataset groups."""

    image_paths = []

    for group_name in DATASET_GROUPS:
        group_path = (
                SOURCE_ROOT
                / group_name
        )

        if not group_path.exists():
            print(
                f"WARNING: missing dataset group: "
                f"{group_path}"
            )
            continue

        for extension in [
            "*.jpg",
            "*.jpeg",
            "*.png",
        ]:
            image_paths.extend(
                group_path.rglob(
                    extension
                )
            )

    return sorted(
        set(image_paths)
    )


def process_image(
        image_path: Path,
        warnings,
):
    """Generate annotations and QA output for one image."""

    relative_path = (
        image_path.relative_to(
            SOURCE_ROOT
        )
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        warning = (
            f"FAILED TO READ: "
            f"{relative_path}"
        )

        print(warning)
        warnings.append(warning)

        return {
            "image": str(relative_path),
            "expected_count": 0,
            "detected_count": 0,
            "expected_colours": "",
            "detected_colours": "",
            "status": "FAILED_TO_READ",
        }

    image_height, image_width = (
        image.shape[:2]
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    expected_colours = (
        expected_colours_from_path(
            image_path
        )
    )

    detected_colours = []
    label_lines = []

    # Draw previews on a copy of the source image.
    preview = image.copy()

    for colour in expected_colours:
        box = find_best_annotation_box(
            hsv=hsv,
            colour=colour,
            image_width=image_width,
            image_height=image_height,
        )

        if box is None:
            warning = (
                f"NO VALID {colour.upper()} BOX: "
                f"{relative_path}"
            )

            print(warning)
            warnings.append(warning)

            continue

        x, y, w, h = box

        class_id = CLASS_MAP[
            colour
        ]

        label_lines.append(
            to_yolo(
                class_id=class_id,
                x=x,
                y=y,
                w=w,
                h=h,
                image_width=image_width,
                image_height=image_height,
            )
        )

        detected_colours.append(
            colour
        )

        draw_preview_box(
            preview=preview,
            x=x,
            y=y,
            w=w,
            h=h,
            colour=colour,
        )

    output_image_path = (
            OUTPUT_ROOT
            / relative_path
    )

    output_label_path = (
        output_image_path.with_suffix(
            ".txt"
        )
    )

    preview_path = (
            PREVIEW_ROOT
            / relative_path
    )

    output_image_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        image_path,
        output_image_path,
    )

    # Empty-workspace images deliberately get an empty label file.
    with open(
            output_label_path,
            "w",
            encoding="utf-8",
    ) as label_file:
        for line in label_lines:
            label_file.write(
                line + "\n"
            )

    if not cv2.imwrite(
            str(preview_path),
            preview,
    ):
        warning = (
            f"FAILED TO SAVE PREVIEW: "
            f"{relative_path}"
        )

        print(warning)
        warnings.append(warning)

    expected_count = len(
        expected_colours
    )

    detected_count = len(
        detected_colours
    )

    if (
            expected_count == detected_count
            and set(expected_colours)
            == set(detected_colours)
    ):
        status = "OK"

    else:
        status = "REVIEW"

    return {
        "image": str(relative_path),
        "expected_count": expected_count,
        "detected_count": detected_count,
        "expected_colours": "|".join(
            expected_colours
        ),
        "detected_colours": "|".join(
            detected_colours
        ),
        "status": status,
    }


def write_report(rows):
    """Write the annotation QA report."""

    fields = [
        "image",
        "expected_count",
        "detected_count",
        "expected_colours",
        "detected_colours",
        "status",
    ]

    with open(
            REPORT_PATH,
            "w",
            newline="",
            encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def write_warnings(warnings):
    """Write annotation warnings."""

    with open(
            WARNINGS_PATH,
            "w",
            encoding="utf-8",
    ) as warning_file:

        if warnings:
            for warning in warnings:
                warning_file.write(
                    warning + "\n"
                )

        else:
            warning_file.write(
                "No warnings.\n"
            )


def main():
    """Generate bootstrap annotations and QA outputs."""

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: "
            f"{SOURCE_ROOT}"
        )

    clear_outputs()

    image_paths = find_images()

    print()
    print("GENERATING BOOTSTRAP ANNOTATIONS")
    print(
        f"Source images: "
        f"{SOURCE_ROOT}"
    )
    print(
        f"Output dataset: "
        f"{OUTPUT_ROOT}"
    )
    print(
        f"Preview folder: "
        f"{PREVIEW_ROOT}"
    )
    print()
    print(
        f"Images found: "
        f"{len(image_paths)}"
    )
    print()

    warnings = []
    report_rows = []

    total_expected = 0
    total_detected = 0

    for index, image_path in enumerate(
            image_paths,
            start=1,
    ):
        relative_path = (
            image_path.relative_to(
                SOURCE_ROOT
            )
        )

        print(
            f"[{index}/{len(image_paths)}] "
            f"{relative_path}"
        )

        result = process_image(
            image_path=image_path,
            warnings=warnings,
        )

        report_rows.append(
            result
        )

        total_expected += (
            result["expected_count"]
        )

        total_detected += (
            result["detected_count"]
        )

    write_report(
        report_rows
    )

    write_warnings(
        warnings
    )

    review_count = sum(
        row["status"] != "OK"
        for row in report_rows
    )

    print()
    print("DONE")
    print(
        f"Images processed:       "
        f"{len(report_rows)}"
    )
    print(
        f"Expected boxes:         "
        f"{total_expected}"
    )
    print(
        f"Detected boxes:         "
        f"{total_detected}"
    )
    print(
        f"Images needing review:  "
        f"{review_count}"
    )
    print(
        f"Warnings:               "
        f"{len(warnings)}"
    )
    print()
    print(
        f"Dataset:  "
        f"{OUTPUT_ROOT}"
    )
    print(
        f"Previews: "
        f"{PREVIEW_ROOT}"
    )
    print(
        f"Report:   "
        f"{REPORT_PATH}"
    )
    print(
        f"Warnings: "
        f"{WARNINGS_PATH}"
    )
    print()
    print(
        "dataset_raw was not modified."
    )
    print(
        "test_shots was not included."
    )


if __name__ == "__main__":
    main()
