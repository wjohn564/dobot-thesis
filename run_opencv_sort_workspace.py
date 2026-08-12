from robot_sorting.detectors import OpenCVColourDetector
from robot_sorting.workspace_sorter import run_workspace_sorting


def main():
    """Run workspace sorting using OpenCV."""

    detector = OpenCVColourDetector()

    run_workspace_sorting(
        detector=detector,
    )


if __name__ == "__main__":
    main()