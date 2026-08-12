from pathlib import Path

from robot_sorting.detectors import YOLODetector
from robot_sorting.workspace_sorter import run_workspace_sorting

PROJECT_ROOT = Path(__file__).resolve().parent

YOLO_MODEL_PATH = (
        PROJECT_ROOT
        / "runs"
        / "yolo"
        / "yolo26n_blocks"
        / "weights"
        / "best_ncnn_model"
)


def main():
    """Run workspace sorting using YOLO26n."""

    detector = YOLODetector(
        model_path=YOLO_MODEL_PATH,
        confidence_threshold=0.25,
    )

    run_workspace_sorting(
        detector=detector,
    )


if __name__ == "__main__":
    main()
