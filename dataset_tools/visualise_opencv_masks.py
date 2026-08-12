# Generate a few OpenCV mask examples for the paper.

import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow imports from the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_sorting import config as cfg
from robot_sorting.detectors import OpenCVColourDetector


SOURCE_ROOT = PROJECT_ROOT / "dataset_raw"

OUTPUT_ROOT = (
        PROJECT_ROOT
        / "generated_visuals"
        / "opencv_masks"
)

# Representative dataset examples.
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
    """Find one matching image from a dataset group."""

    group_path = (
            SOURCE_ROOT
            / group_name
    )

    if not group_path.exists():
        return None

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

    # Keep the chosen example reproducible.
    image_paths = sorted(
        image_paths
    )

    for image_path in image_paths:

        if keyword is None:
            return image_path

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
    """Create one mask containing all configured block colours."""

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    combined_mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    # Reuse the same masks and ROI as the runtime detector.
    for colour in cfg.HSV_RANGES:
        mask = OpenCVColourDetector._create_mask(
            hsv,
            colour,
        )

        mask = OpenCVColourDetector._apply_roi(
            mask
        )

        combined_mask = cv2.bitwise_or(
            combined_mask,
            mask,
        )

    return combined_mask


def create_comparison(
        image,
        mask,
):
    """Place the source image and mask side by side."""

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
    """Generate the selected OpenCV mask visualisations."""

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: "
            f"{SOURCE_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("GENERATING OPENCV MASK VISUALISATIONS")
    print()

    for example_name, settings in EXAMPLES.items():

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

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"WARNING: failed to read "
                f"{image_path}"
            )
            continue

        mask = create_combined_mask(
            image
        )

        comparison = create_comparison(
            image,
            mask,
        )

        mask_path = (
                OUTPUT_ROOT
                / f"{example_name}_mask.png"
        )

        cv2.imwrite(
            str(mask_path),
            mask,
        )

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


if __name__ == "__main__":
    main()
