import csv
import re
import shutil
import sys
from pathlib import Path

import cv2

# This script automatically creates initial bounding box annotations for the raw dataset.
# It reuses the existing OpenCV colour segmentation logic, saves the generated annotations,
# creates preview images for manual checking, and records any annotation problems.


# This finds the project root.
# It works on both the Raspberry Pi and Windows.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add the project root so project modules can be imported.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the existing OpenCV colour detector and its configured thresholds
# rather than copying the detection logic into this script.
from robot_sorting import config as cfg
from robot_sorting.detectors import OpenCVColourDetector

# Original captured dataset.
# This folder is treated as read only by this script.
SOURCE_ROOT = PROJECT_ROOT / "dataset_raw"

# Generated copy of the dataset containing the automatic bounding box annotations.
OUTPUT_ROOT = PROJECT_ROOT / "dataset_annotated_bootstrap"

# Images with the generated boxes drawn on them so the annotations can be checked visually.
PREVIEW_ROOT = PROJECT_ROOT / "dataset_annotation_previews"

# Quality control files.
REPORT_PATH = OUTPUT_ROOT / "annotation_report.csv"
WARNINGS_PATH = OUTPUT_ROOT / "annotation_warnings.txt"

# These are the dataset folders that should be automatically annotated.
# Any other folders inside dataset_raw are ignored.
DATASET_GROUPS = [
    "01_single_blocks",
    "02_two_blocks",
    "03_three_blocks",
    "04_four_blocks",
    "05_empty_workspace",
]

# Map each block colour to the numeric class ID stored in the annotation file.
# This mapping should not be changed once model training begins.
CLASS_MAP = {
    "red": 0,
    "blue": 1,
    "yellow": 2,
    "green": 3,
}


def expected_colours_from_path(
        image_path: Path,
):
    """
    Return the block colours that should appear in an image.

    The dataset folder names and filenames record which colours were
    present when each image was captured. This lets the script check
    whether the expected blocks were actually found.
    """

    # Get only the part of the path inside dataset_raw.
    # Convert it to lowercase so colour matching is consistent.
    relative_text = str(
        image_path.relative_to(SOURCE_ROOT)
    ).lower()

    # Empty workspace images contain no blocks.
    if "empty_workspace" in relative_text:
        return []

    expected = []

    # Check each possible block colour.
    for colour in [
        "red",
        "blue",
        "yellow",
        "green",
    ]:
        # Check whether the colour appears as a separate part
        # of the folder name or filename.
        #
        # Examples:
        # blue_cube
        # red_blue
        # red_blue_yellow_green
        if re.search(
                rf"(^|[_\\/.\-])"
                rf"{colour}"
                rf"($|[_\\/.\-])",
                relative_text,
        ):
            expected.append(colour)

        elif f"{colour}_cube" in relative_text:
            expected.append(colour)

    # Remove duplicate colours while keeping the original order.
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
    """
    Add a small margin around a detected bounding box.

    Colour segmentation can stop slightly inside the visible edge of
    the block, so the box is made slightly larger. The box is clipped
    so it cannot extend outside the image.

    In the runtime OpenCV detector BOX_PADDING is only used when drawing
    previews. In this annotation script the padding is deliberately added
    to the saved annotation box.
    """

    # Move the left side of the box outwards.
    # max() stops it going outside the left side of the image.
    x1 = max(
        0,
        x - cfg.BOX_PADDING,
    )

    # Move the top side of the box outwards.
    y1 = max(
        0,
        y - cfg.BOX_PADDING,
    )

    # Move the right side of the box outwards.
    # min() stops it going outside the right side of the image.
    x2 = min(
        image_width,
        x + w + cfg.BOX_PADDING,
    )

    # Move the bottom side of the box outwards.
    y2 = min(
        image_height,
        y + h + cfg.BOX_PADDING,
    )

    # Return the padded box as x, y, width, height.
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
    """
    Find the best bounding box for one expected block colour.

    The existing OpenCV detector is reused to create the colour mask
    and apply the camera region of interest. Colour regions that are
    too large, too small, or the wrong shape are rejected. The largest
    remaining valid region is used as the block.
    """

    # Reuse the existing OpenCV detector to isolate pixels
    # belonging to this colour.
    mask = OpenCVColourDetector._create_mask(
        hsv,
        colour,
    )

    # Ignore colour regions outside the configured camera ROI.
    mask = OpenCVColourDetector._apply_roi(
        mask
    )

    # Find separate connected colour regions in the binary mask.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Store all regions that could realistically be a block.
    candidates = []

    for contour in contours:

        # Calculate the pixel area of this colour region.
        area = float(
            cv2.contourArea(contour)
        )

        # Reject regions that are too small to be a block.
        if area < cfg.MIN_AREA:
            continue

        # Reject regions that are too large to be a block.
        if area > cfg.MAX_AREA:
            continue

        # Create a rectangle around the colour region.
        x, y, w, h = cv2.boundingRect(
            contour
        )

        # Reject boxes that are too narrow.
        if w < cfg.MIN_BOX_WIDTH:
            continue

        # Reject boxes that are too short.
        if h < cfg.MIN_BOX_HEIGHT:
            continue

        # Reject boxes that are too wide.
        if w > cfg.MAX_BOX_WIDTH:
            continue

        # Reject boxes that are too tall.
        if h > cfg.MAX_BOX_HEIGHT:
            continue

        # Check whether the box shape is realistic for a block.
        aspect_ratio = w / float(h)

        if aspect_ratio < cfg.MIN_ASPECT_RATIO:
            continue

        if aspect_ratio > cfg.MAX_ASPECT_RATIO:
            continue

        # Add a small margin around the detected colour region.
        #
        # Unlike the runtime detector, this padding becomes part
        # of the saved annotation box.
        x, y, w, h = pad_box(
            x=x,
            y=y,
            w=w,
            h=h,
            image_width=image_width,
            image_height=image_height,
        )

        # Store the valid box together with its original contour area.
        candidates.append(
            (
                area,
                x,
                y,
                w,
                h,
            )
        )

    # No suitable colour region was found.
    if not candidates:
        return None

    # We already know which colour should be present from the dataset structure.
    # The largest realistic region of that colour is therefore taken as the block.
    candidates.sort(
        key=lambda candidate: candidate[0],
        reverse=True,
    )

    # Take the largest candidate.
    # The contour area is no longer needed, so _ is used for it.
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
    """
    Convert a pixel bounding box into normalised YOLO annotation format.

    The top-left x/y position and the box width and height are converted
    into normalised centre coordinates, width, and height.
    """

    # Calculate the horizontal centre of the box and normalise it.
    x_center = (
                       x + w / 2.0
               ) / image_width

    # Calculate the vertical centre of the box and normalise it.
    y_center = (
                       y + h / 2.0
               ) / image_height

    # Normalise the box size relative to the full image.
    width = w / image_width
    height = h / image_height

    # Return one annotation line in:
    # class_id x_center y_center width height
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
    """
    Draw one generated annotation on a preview image.

    The bounding box, colour name, and centre point are drawn so the
    automatic annotation can be checked visually.
    """

    # Draw the generated bounding box.
    cv2.rectangle(
        preview,
        (x, y),
        (x + w, y + h),
        (255, 255, 255),
        2,
    )

    # Calculate the centre point of the generated box.
    u = int(
        round(x + w / 2.0)
    )

    v = int(
        round(y + h / 2.0)
    )

    # Draw the centre point on the preview.
    cv2.circle(
        preview,
        (u, v),
        6,
        (255, 255, 255),
        -1,
    )

    # Write the block colour and centre coordinates above the box.
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
    """
    Clear outputs from the previous annotation run.

    Only generated annotation and preview folders are deleted and
    recreated. The original dataset_raw folder is never modified.
    """

    # Remove the old generated annotated dataset if it exists.
    if OUTPUT_ROOT.exists():
        shutil.rmtree(
            OUTPUT_ROOT
        )

    # Remove the old preview folder if it exists.
    if PREVIEW_ROOT.exists():
        shutil.rmtree(
            PREVIEW_ROOT
        )

    # Create a clean output folder for this run.
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create a clean preview folder for this run.
    PREVIEW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


def find_images():
    """
    Find all images that belong to the configured dataset groups.

    The search includes nested folders. Folders that are not listed
    in DATASET_GROUPS are ignored.
    """

    image_paths = []

    # Check each configured dataset group.
    for group_name in DATASET_GROUPS:
        group_path = (
                SOURCE_ROOT
                / group_name
        )

        # Warn about a missing dataset folder but continue running.
        if not group_path.exists():
            print(
                f"WARNING: missing dataset group: "
                f"{group_path}"
            )
            continue

        # Search for the image formats used by the dataset.
        for extension in [
            "*.jpg",
            "*.jpeg",
            "*.png",
        ]:
            # rglob searches through all subfolders as well.
            image_paths.extend(
                group_path.rglob(
                    extension
                )
            )

    # Remove duplicate paths and return them in a fixed order.
    return sorted(
        set(image_paths)
    )


def process_image(
        image_path: Path,
        warnings,
):
    """
    Generate annotations and quality-control outputs for one image.

    The expected block colours are read from the dataset path. Each
    expected block is then located using the OpenCV annotation method.
    The image and its annotation are saved, a preview is created, and
    a quality-control result is returned.
    """

    # Keep the path relative to dataset_raw.
    # This lets the same folder structure be recreated in the output.
    relative_path = (
        image_path.relative_to(
            SOURCE_ROOT
        )
    )

    # Load the original image.
    image = cv2.imread(
        str(image_path)
    )

    # Record a failed image read without stopping the rest of the dataset.
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

    # Get the image dimensions.
    image_height, image_width = (
        image.shape[:2]
    )

    # Convert the image to HSV because the colour detector uses HSV.
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    # Use the dataset naming structure to determine which blocks
    # should be present in this image.
    expected_colours = (
        expected_colours_from_path(
            image_path
        )
    )

    # Store which expected blocks were successfully found.
    detected_colours = []

    # Store the annotation lines that will be written to the .txt file.
    label_lines = []

    # Draw on a copy so the original image is not changed.
    preview = image.copy()

    # Try to create one annotation for each block that should be present.
    for colour in expected_colours:
        box = find_best_annotation_box(
            hsv=hsv,
            colour=colour,
            image_width=image_width,
            image_height=image_height,
        )

        # Flag the image if an expected block could not be found.
        if box is None:
            warning = (
                f"NO VALID {colour.upper()} BOX: "
                f"{relative_path}"
            )

            print(warning)
            warnings.append(warning)

            continue

        # Unpack the detected bounding box.
        x, y, w, h = box

        # Convert the colour name into its numeric class ID.
        class_id = CLASS_MAP[
            colour
        ]

        # Convert the box into the annotation format and store it.
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

        # Record that this expected colour was successfully found.
        detected_colours.append(
            colour
        )

        # Draw the generated annotation on the preview image.
        draw_preview_box(
            preview=preview,
            x=x,
            y=y,
            w=w,
            h=h,
            colour=colour,
        )

    # Recreate the original dataset folder structure inside the generated output.
    output_image_path = (
            OUTPUT_ROOT
            / relative_path
    )

    # Give the annotation the same filename as the image, but with .txt.
    output_label_path = (
        output_image_path.with_suffix(
            ".txt"
        )
    )

    # Create the matching preview path.
    preview_path = (
            PREVIEW_ROOT
            / relative_path
    )

    # Create any missing folders before writing the output files.
    output_image_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Copy the original image into the generated annotated dataset.
    shutil.copy2(
        image_path,
        output_image_path,
    )

    # Write one annotation line for each detected block.
    #
    # Empty-workspace images have no blocks, so their label file
    # is deliberately created with no annotation lines.
    with open(
            output_label_path,
            "w",
            encoding="utf-8",
    ) as label_file:
        for line in label_lines:
            label_file.write(
                line + "\n"
            )

    # Save the preview image used for manual checking.
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

    # Count how many blocks should be present and how many were found.
    expected_count = len(
        expected_colours
    )

    detected_count = len(
        detected_colours
    )

    # Mark the image as OK only when the expected number and colours
    # exactly match the generated annotations.
    if (
            expected_count == detected_count
            and set(expected_colours)
            == set(detected_colours)
    ):
        status = "OK"

    else:
        status = "REVIEW"

    # Return the information that will be written to the QA report.
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
    """
    Write the annotation quality-control report to CSV.

    Each row records which blocks were expected, which were detected,
    their counts, and whether the image passed the automatic check.
    """

    # Columns stored for every processed image.
    fields = [
        "image",
        "expected_count",
        "detected_count",
        "expected_colours",
        "detected_colours",
        "status",
    ]

    # Create the CSV report.
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
    """
    Write annotation warnings to a text file.

    If there were no warnings, the file records that the run
    completed without warnings.
    """

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
    """
    Run the complete automatic annotation process.

    The previous generated outputs are cleared, all configured dataset
    images are processed, quality-control files are written, and a
    summary of the run is printed.
    """

    # Stop if the original dataset cannot be found.
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: "
            f"{SOURCE_ROOT}"
        )

    # Start with clean output directories.
    clear_outputs()

    # Find all images belonging to the configured dataset groups.
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

    # Store warnings and the result for each processed image.
    warnings = []
    report_rows = []

    # Track how many boxes should exist and how many were successfully generated.
    total_expected = 0
    total_detected = 0

    # Process each image in the dataset.
    for index, image_path in enumerate(
            image_paths,
            start=1,
    ):
        relative_path = (
            image_path.relative_to(
                SOURCE_ROOT
            )
        )

        # Print progress through the dataset.
        print(
            f"[{index}/{len(image_paths)}] "
            f"{relative_path}"
        )

        # Generate the annotations and QA result for this image.
        result = process_image(
            image_path=image_path,
            warnings=warnings,
        )

        # Store the result for the final CSV report.
        report_rows.append(
            result
        )

        # Update the total expected and detected box counts.
        total_expected += (
            result["expected_count"]
        )

        total_detected += (
            result["detected_count"]
        )

    # Save the quality-control results after every image has been processed.
    write_report(
        report_rows
    )

    write_warnings(
        warnings
    )

    # Count images that did not pass the automatic annotation check.
    review_count = sum(
        row["status"] != "OK"
        for row in report_rows
    )

    # Print a final summary of the annotation run.
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


# Run the annotation process only when this file is executed directly.
if __name__ == "__main__":
    main()
