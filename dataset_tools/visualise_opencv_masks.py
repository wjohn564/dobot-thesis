# This file is just for testing purposes to get images for my paper

import sys
from pathlib import Path

import cv2
import numpy as np

# This script creates a small number of OpenCV mask visualisations.
# It reuses the existing colour detector so the masks match those
# used by the robotic sorting system.


# Find the project root.
# This works on both the Raspberry Pi and Windows.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add the project root so project modules can be imported.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the existing OpenCV colour detector rather than copying
# the HSV thresholding and ROI logic into this script.
from robot_sorting import config as cfg
from robot_sorting.detectors import OpenCVColourDetector

# Original captured dataset.
SOURCE_ROOT = PROJECT_ROOT / "dataset_raw"

# Generated mask images.
OUTPUT_ROOT = (
        PROJECT_ROOT
        / "generated_visuals"
        / "opencv_masks"
)

# Representative examples to find in the dataset.
#
# The first matching image for each example is used.
EXAMPLES = {
    "single_normal": {
        "group": "01_single_blocks",
        "keyword": "normal",
    },
    "single_edge": {
        "group": "01_single_blocks",
        "keyword": "edge",
    },
    "multi_block": {
        "group": "03_three_blocks",
        "keyword": None,
    },
    "empty_workspace": {
        "group": "05_empty_workspace",
        "keyword": None,
    },
}


def find_example_image(
        group_name: str,
        keyword=None,
):
    """
    Find one representative image from a dataset group.

    If a keyword is given, the image path must also contain that
    word. The first matching image is returned.
    """

    group_path = (
            SOURCE_ROOT
            / group_name
    )

    if not group_path.exists():
        return None

    # Search through the supported image formats.
    image_paths = []

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

    # Keep the order fixed so the same example is chosen each time.
    image_paths = sorted(
        image_paths
    )

    for image_path in image_paths:

        # If no keyword is required, use the first image found.
        if keyword is None:
            return image_path

        # Check both the folder names and filename for the keyword.
        relative_text = str(
            image_path.relative_to(
                SOURCE_ROOT
            )
        ).lower()

        if keyword.lower() in relative_text:
            return image_path

    return None


def create_combined_mask(
        image,
):
    """
    Create one combined OpenCV colour mask for an image.

    A mask is created for each configured block colour using the
    existing OpenCV detector. The masks are combined so all detected
    block colours appear together in one binary image.
    """

    # Convert the image to HSV because the colour detector uses HSV.
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    # Start with a completely black mask.
    combined_mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    # Create the same colour masks used by the runtime detector.
    for colour in cfg.HSV_RANGES:
        mask = OpenCVColourDetector._create_mask(
            hsv,
            colour,
        )

        # Apply the same camera ROI used by the runtime detector.
        mask = OpenCVColourDetector._apply_roi(
            mask
        )

        # Add this colour to the combined mask.
        combined_mask = cv2.bitwise_or(
            combined_mask,
            mask,
        )

    return combined_mask


def create_comparison(
        image,
        mask,
):
    """
    Place the original image and OpenCV mask beside each other.

    The mask is converted to three channels so it can be joined
    directly with the original colour image.
    """

    mask_bgr = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR,
    )

    return np.hstack(
        (
            image,
            mask_bgr,
        )
    )


def main():
    """
    Generate OpenCV mask visualisations for representative dataset images.

    A small number of raw images are selected automatically. The existing
    detector is used to create a combined colour mask for each image, and
    both the mask and an original-versus-mask comparison are saved.
    """

    # Stop if the raw dataset cannot be found.
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: "
            f"{SOURCE_ROOT}"
        )

    # Create the output directory if it does not already exist.
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("GENERATING OPENCV MASK VISUALISATIONS")
    print()

    for example_name, settings in EXAMPLES.items():

        # Find one image that represents this example.
        image_path = find_example_image(
            group_name=settings["group"],
            keyword=settings["keyword"],
        )

        if image_path is None:
            print(
                f"WARNING: no image found for "
                f"{example_name}"
            )
            continue

        # Load the original image.
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"WARNING: failed to read "
                f"{image_path}"
            )
            continue

        # Generate the combined OpenCV colour mask.
        mask = create_combined_mask(
            image
        )

        # Create an original-versus-mask image.
        comparison = create_comparison(
            image,
            mask,
        )

        # Save the mask by itself.
        mask_path = (
                OUTPUT_ROOT
                / f"{example_name}_mask.png"
        )

        cv2.imwrite(
            str(mask_path),
            mask,
        )

        # Save the original image beside the mask.
        comparison_path = (
                OUTPUT_ROOT
                / f"{example_name}_comparison.png"
        )

        cv2.imwrite(
            str(comparison_path),
            comparison,
        )

        print(
            f"{example_name}: "
            f"{image_path.relative_to(SOURCE_ROOT)}"
        )

    print()
    print(
        f"Visualisations saved to: "
        f"{OUTPUT_ROOT}"
    )


# Run the visualisation process only when this file is executed directly.
if __name__ == "__main__":
    main()
