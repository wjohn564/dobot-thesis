from pathlib import Path

from robot_sorting.detectors import FasterRCNNDetector
from robot_sorting.workspace_sorter import run_workspace_sorting

PROJECT_ROOT = Path(__file__).resolve().parent

FASTER_RCNN_MODEL_PATH = (
        PROJECT_ROOT
        / "runs"
        / "faster_rcnn"
        / "faster_rcnn_mobilenet_v3_320_best.pth"
)


def main():
    """Run workspace sorting using Faster R-CNN."""

    detector = FasterRCNNDetector(
        model_path=FASTER_RCNN_MODEL_PATH,
        confidence_threshold=0.20,
    )

    run_workspace_sorting(
        detector=detector,
    )


if __name__ == "__main__":
    main()
