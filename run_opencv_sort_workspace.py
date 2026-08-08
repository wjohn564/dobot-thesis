from robot_sorting.detectors import OpenCVColourDetector
from robot_sorting.workspace_sorter import run_workspace_sorting


def main():
    detector = OpenCVColourDetector()
    run_workspace_sorting(detector)


if __name__ == "__main__":
    main()
