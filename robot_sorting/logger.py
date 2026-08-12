import csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from robot_sorting import config as cfg
from robot_sorting.types import Detection

# detections.csv columns
DETECTION_FIELDS = [
    "run_id",
    "timestamp",
    "method",
    "scene_id",
    "repetition",
    "cycle",
    "stage",
    "image_name",
    "preview_name",
    "detection_index",
    "selected",
    "class_name",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "u",
    "v",
    "robot_x",
    "robot_y",
    "bin_name",
    "area",
    "inference_time_ms",
]

# attempts.csv columns
ATTEMPT_FIELDS = [
    "run_id",
    "timestamp",
    "method",
    "scene_id",
    "repetition",
    "cycle",
    "pre_image_name",
    "pre_preview_name",
    "verify_image_name",
    "verify_preview_name",
    "detections_before",
    "detections_after",
    "selected_colour",
    "selected_bin",
    "selected_u",
    "selected_v",
    "selected_robot_x",
    "selected_robot_y",
    "inference_time_ms",
    "verify_inference_time_ms",
    "cycle_time_ms",
    "attempted",
    "auto_success",
    "manual_success",
    "correct_bin",
    "notes",
]

# run_summary.csv columns
RUN_SUMMARY_FIELDS = [
    "run_id",
    "timestamp",
    "method",
    "scene_id",
    "repetition",
    "scene_success",
    "blocks_remaining",
    "notes",
]


class ExperimentLogger:
    """Log one physical experiment run."""

    def __init__(
            self,
            method: str,
            scene_id: str,
            repetition: int,
    ):
        # Run metadata.
        self.method = method

        self.scene_id = (
            str(scene_id)
            .strip()
            .lower()
        )

        self.repetition = int(
            repetition
        )

        if not self.scene_id:
            raise ValueError(
                "scene_id cannot be empty."
            )

        if self.repetition < 1:
            raise ValueError(
                "repetition must be 1 or greater."
            )

        # Unique run ID.
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.run_id = (
            f"{method}_"
            f"{self.scene_id}_"
            f"r{self.repetition:02d}_"
            f"{timestamp}"
        )

        # Create the run log folder.
        self.run_log_dir = (
                cfg.LOG_DIR
                / self.run_id
        )

        self.run_log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # CSV output paths.
        self.detections_csv = (
                self.run_log_dir
                / "detections.csv"
        )

        self.attempts_csv = (
                self.run_log_dir
                / "attempts.csv"
        )

        # One summary row for the full scene.
        self.run_summary_csv = (
                self.run_log_dir
                / "run_summary.csv"
        )

        print(
            f"Run ID: {self.run_id}"
        )

        print(
            f"Logs: {self.run_log_dir}"
        )

    @staticmethod
    def _append_row(
            path: Path,
            fieldnames,
            row,
    ):
        """Append one row to a CSV file."""

        file_exists = path.exists()

        with open(
                path,
                "a",
                newline="",
                encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    def log_detections(
            self,
            cycle: int,
            stage: str,
            image_name: str,
            preview_name: str,
            detections: List[Detection],
            inference_time_ms: float,
            selected: Optional[Detection] = None,
    ):
        """Log all detections from one image."""

        timestamp = datetime.now().isoformat(
            timespec="seconds"
        )

        for i, det in enumerate(detections):
            row = {
                "run_id": self.run_id,
                "timestamp": timestamp,
                "method": self.method,
                "scene_id": self.scene_id,
                "repetition": self.repetition,
                "cycle": cycle,
                "stage": stage,
                "image_name": image_name,
                "preview_name": preview_name,
                "detection_index": i,
                "selected": int(
                    det is selected
                ),
                "class_name": det.class_name,
                "confidence": (
                    f"{det.confidence:.4f}"
                ),
                "bbox_x": det.bbox_x,
                "bbox_y": det.bbox_y,
                "bbox_w": det.bbox_w,
                "bbox_h": det.bbox_h,
                "u": f"{det.u:.3f}",
                "v": f"{det.v:.3f}",
                "robot_x": (
                    ""
                    if det.robot_x is None
                    else f"{det.robot_x:.3f}"
                ),
                "robot_y": (
                    ""
                    if det.robot_y is None
                    else f"{det.robot_y:.3f}"
                ),
                "bin_name": (
                    ""
                    if det.bin_name is None
                    else det.bin_name
                ),
                "area": f"{det.area:.3f}",
                "inference_time_ms": (
                    f"{inference_time_ms:.3f}"
                ),
            }

            self._append_row(
                self.detections_csv,
                DETECTION_FIELDS,
                row,
            )

    def log_attempt(
            self,
            cycle: int,
            pre_image_name: str,
            pre_preview_name: str,
            verify_image_name: str,
            verify_preview_name: str,
            detections_before: int,
            detections_after: int,
            selected: Optional[Detection],
            inference_time_ms: float,
            verify_inference_time_ms: float,
            cycle_time_ms: float,
            attempted: bool,
            auto_success,
            manual_success,
            correct_bin,
            notes: str,
    ):
        """Log one pick attempt."""

        timestamp = datetime.now().isoformat(
            timespec="seconds"
        )

        if selected is None:
            selected_colour = ""
            selected_bin = ""
            selected_u = ""
            selected_v = ""
            selected_robot_x = ""
            selected_robot_y = ""

        else:
            selected_colour = (
                selected.class_name
            )

            selected_bin = (
                ""
                if selected.bin_name is None
                else selected.bin_name
            )

            selected_u = (
                f"{selected.u:.3f}"
            )

            selected_v = (
                f"{selected.v:.3f}"
            )

            selected_robot_x = (
                ""
                if selected.robot_x is None
                else f"{selected.robot_x:.3f}"
            )

            selected_robot_y = (
                ""
                if selected.robot_y is None
                else f"{selected.robot_y:.3f}"
            )

        row = {
            "run_id": self.run_id,
            "timestamp": timestamp,
            "method": self.method,
            "scene_id": self.scene_id,
            "repetition": self.repetition,
            "cycle": cycle,
            "pre_image_name": pre_image_name,
            "pre_preview_name": pre_preview_name,
            "verify_image_name": verify_image_name,
            "verify_preview_name": verify_preview_name,
            "detections_before": detections_before,
            "detections_after": detections_after,
            "selected_colour": selected_colour,
            "selected_bin": selected_bin,
            "selected_u": selected_u,
            "selected_v": selected_v,
            "selected_robot_x": selected_robot_x,
            "selected_robot_y": selected_robot_y,
            "inference_time_ms": (
                f"{inference_time_ms:.3f}"
            ),
            "verify_inference_time_ms": (
                f"{verify_inference_time_ms:.3f}"
            ),
            "cycle_time_ms": (
                f"{cycle_time_ms:.3f}"
            ),
            "attempted": int(
                attempted
            ),
            "auto_success": (
                ""
                if auto_success is None
                else int(
                    bool(auto_success)
                )
            ),
            "manual_success": (
                ""
                if manual_success is None
                else int(
                    bool(manual_success)
                )
            ),
            "correct_bin": (
                ""
                if correct_bin is None
                else int(
                    bool(correct_bin)
                )
            ),
            "notes": notes,
        }

        self._append_row(
            self.attempts_csv,
            ATTEMPT_FIELDS,
            row,
        )

    def log_run_summary(
            self,
            scene_success: bool,
            blocks_remaining: int,
            notes: str = "",
    ):
        """Log the final result for the whole scene."""

        row = {
            "run_id": self.run_id,
            "timestamp": (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            ),
            "method": self.method,
            "scene_id": self.scene_id,
            "repetition": self.repetition,
            "scene_success": int(
                bool(scene_success)
            ),
            "blocks_remaining": int(
                blocks_remaining
            ),
            "notes": notes,
        }

        self._append_row(
            self.run_summary_csv,
            RUN_SUMMARY_FIELDS,
            row,
        )
