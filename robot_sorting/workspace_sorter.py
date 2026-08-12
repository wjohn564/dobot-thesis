import time
from datetime import datetime
from typing import Optional

from robot_sorting import config as cfg
from robot_sorting.camera import capture_image
from robot_sorting.detectors import BaseDetector
from robot_sorting.dobot_controller import DobotController
from robot_sorting.geometry import (
    add_robot_coordinates,
    load_homography,
)
from robot_sorting.logger import ExperimentLogger
from robot_sorting.selection import (
    assign_bins,
    choose_next_target,
    selected_removed_after_pick,
)
from robot_sorting.types import Detection
from robot_sorting.visualisation import save_preview

# Number of unlogged detector runs before the experiment.
DETECTOR_WARMUP_RUNS = 3


def get_experiment_details():
    """Get the scene and repetition for this experiment run."""

    print()
    print("EXPERIMENT DETAILS")

    # Scene ID.
    while True:
        scene_id = input(
            "Scene ID (for example s01): "
        ).strip().lower()

        if scene_id:
            break

        print("Scene ID cannot be empty.")

    # Repetition number.
    while True:
        try:
            repetition = int(
                input(
                    "Repetition number: "
                ).strip()
            )

            if repetition < 1:
                raise ValueError

            break

        except ValueError:
            print(
                "Repetition must be 1 or greater."
            )

    return scene_id, repetition


def ask_yes_no(prompt: str) -> bool:
    """Ask the user for a yes or no answer."""

    while True:
        answer = input(
            f"{prompt} [y/n]: "
        ).strip().lower()

        if answer in [
            "y",
            "yes",
        ]:
            return True

        if answer in [
            "n",
            "no",
        ]:
            return False

        print("Please enter y or n.")


def ask_non_negative_int(
        prompt: str,
) -> int:
    """Ask for zero or a positive integer."""

    while True:
        try:
            value = int(
                input(
                    f"{prompt}: "
                ).strip()
            )

            if value < 0:
                raise ValueError

            return value

        except ValueError:
            print(
                "Please enter 0 or a positive integer."
            )


def make_preview_path(
        logger: ExperimentLogger,
        stage: str,
        cycle: int,
):
    """Create a unique path for a preview image."""

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    return (
            cfg.PREVIEW_DIR
            / (
                f"{logger.run_id}_"
                f"{stage}_"
                f"cycle_{cycle:02d}_"
                f"{timestamp}.jpg"
            )
    )


def capture_and_prepare(
        detector: BaseDetector,
        homography,
        prefix: str,
):
    """Capture an image and prepare its detections for sorting."""

    image_path, image = capture_image(
        prefix=prefix
    )

    # Detect objects.
    detections, inference_time_ms = detector.detect(
        image
    )

    # Map detections to robot coordinates.
    add_robot_coordinates(
        detections,
        homography,
    )

    # Assign target bins.
    assign_bins(detections)

    # Return everything needed by the sorting loop.
    return (
        image_path,
        image,
        detections,
        inference_time_ms,
    )


def save_detection_stage(
        logger: ExperimentLogger,
        cycle: int,
        stage: str,
        image_path,
        image,
        detections,
        inference_time_ms: float,
        selected: Optional[Detection] = None,
):
    """Save the preview and detection rows."""

    # Unique preview path for this stage.
    preview_path = make_preview_path(
        logger=logger,
        stage=stage,
        cycle=cycle,
    )

    # Save the annotated preview.
    save_preview(
        image=image,
        detections=detections,
        selected=selected,
        preview_path=preview_path,
    )

    # Log the detections.
    logger.log_detections(
        cycle=cycle,
        stage=stage,
        image_name=image_path.name,
        preview_name=preview_path.name,
        detections=detections,
        inference_time_ms=inference_time_ms,
        selected=selected,
    )

    return preview_path


def print_selected_target(
        selected: Detection,
):
    """Print the selected block and its target coordinates."""

    print()
    print("SELECTED TARGET")
    print(f"Colour: {selected.class_name}")
    print(f"Bin:    {selected.bin_name}")

    print(
        f"Pixel:  "
        f"u={selected.u:.3f}, "
        f"v={selected.v:.3f}"
    )

    print(
        f"Robot:  "
        f"X={selected.robot_x:.3f}, "
        f"Y={selected.robot_y:.3f}"
    )


def verify_pick(
        detector: BaseDetector,
        homography,
        logger: ExperimentLogger,
        selected: Detection,
        cycle: int,
):
    """Capture and check the workspace after a pick attempt."""

    (
        image_path,
        image,
        detections,
        inference_time_ms,
    ) = capture_and_prepare(
        detector=detector,
        homography=homography,
        prefix=(
            f"{logger.run_id}_verify"
        ),
    )

    # Save the verification stage.
    preview_path = save_detection_stage(
        logger=logger,
        cycle=cycle,
        stage="verify",
        image_path=image_path,
        image=image,
        detections=detections,
        inference_time_ms=inference_time_ms,
        selected=None,
    )

    # Selected colour disappearing is the automatic success proxy.
    auto_success = selected_removed_after_pick(
        selected,
        detections,
    )

    print()
    print("VERIFY RESULT")

    print(
        f"Auto success proxy: "
        f"{auto_success}"
    )

    print(
        f"Detections after pickup: "
        f"{len(detections)}"
    )

    return (
        image_path.name,
        preview_path.name,
        len(detections),
        inference_time_ms,
        auto_success,
    )


def warm_up_detector(
        detector: BaseDetector,
        logger: ExperimentLogger,
):
    """Run the detector before formal timing starts."""

    print()
    print(
        f"Warming up {detector.name}..."
    )

    # One image is reused for warm-up.
    _image_path, image = capture_image(
        prefix=(
            f"{logger.run_id}_warmup"
        )
    )

    # Warm-up runs are not logged.
    detector.warm_up(
        image=image,
        runs=DETECTOR_WARMUP_RUNS,
    )

    print(
        f"Warm-up complete "
        f"({DETECTOR_WARMUP_RUNS} runs)."
    )


def run_workspace_sorting(
        detector: BaseDetector,
):
    """Run the complete robotic workspace sorting process."""

    # Experiment metadata.
    scene_id, repetition = (
        get_experiment_details()
    )

    # Validate runtime config.
    cfg.ensure_runtime_config_ready()

    # Load the homography.
    homography = load_homography()

    # Create the experiment logger.
    logger = ExperimentLogger(
        method=detector.name,
        scene_id=scene_id,
        repetition=repetition,
    )

    print()
    print(
        f"STARTING {detector.name.upper()} "
        f"WORKSPACE SORTING"
    )

    print(
        f"Scene:      {scene_id}"
    )

    print(
        f"Repetition: {repetition}"
    )

    print("red/yellow -> warm_bin")
    print("blue/green -> cool_bin")
    print()

    # Connect to the dobot magician lite
    with DobotController() as robot:

        print(
            "Moving Dobot out of camera view."
        )

        # Clear the camera view.
        robot.move_camera_clear()

        time.sleep(0.5)

        # Warm up before formal inference timing.
        warm_up_detector(
            detector=detector,
            logger=logger,
        )

        # Main sorting loop.
        for cycle in range(
                1,
                cfg.MAX_CYCLES + 1,
        ):
            # Time the full cycle.
            cycle_start = time.perf_counter()

            print()
            print(
                f"===== CYCLE {cycle} ====="
            )

            (
                pre_image_path,
                pre_image,
                detections,
                inference_time_ms,
            ) = capture_and_prepare(
                detector=detector,
                homography=homography,
                prefix=(
                    f"{logger.run_id}_pre_pick"
                ),
            )

            # Choose the next target.
            selected = choose_next_target(
                detections
            )

            pre_preview_path = save_detection_stage(
                logger=logger,
                cycle=cycle,
                stage="pre_pick",
                image_path=pre_image_path,
                image=pre_image,
                detections=detections,
                inference_time_ms=inference_time_ms,
                selected=selected,
            )

            # Stop sorting if no pickable target is found.
            if selected is None:
                # Log the final no-target cycle.
                cycle_time_ms = (
                                        time.perf_counter()
                                        - cycle_start
                                ) * 1000.0

                logger.log_attempt(
                    cycle=cycle,
                    pre_image_name=pre_image_path.name,
                    pre_preview_name=pre_preview_path.name,
                    verify_image_name="",
                    verify_preview_name="",
                    detections_before=len(
                        detections
                    ),
                    detections_after=0,
                    selected=None,
                    inference_time_ms=(
                        inference_time_ms
                    ),
                    verify_inference_time_ms=0.0,
                    cycle_time_ms=cycle_time_ms,
                    attempted=False,
                    auto_success=None,
                    manual_success=None,
                    correct_bin=None,
                    notes=(
                        "workspace_empty_or_"
                        "no_pickable_detection"
                    ),
                )

                print(
                    "No pickable detections. "
                    "Stopping."
                )

                break

            # Show the selected target.
            print_selected_target(
                selected
            )

            # Target must have a bin.
            if selected.bin_name is None:
                raise RuntimeError(
                    "Selected target does not have a bin."
                )

            drop_pose = cfg.DROP_BINS[
                selected.bin_name
            ]

            # Drop pose must be configured.
            if drop_pose is None:
                raise RuntimeError(
                    f"Drop pose is not configured for "
                    f"{selected.bin_name}."
                )

            # Pick the selected block and move it to its configured bin.
            robot.pick_and_drop(
                selected,
                drop_pose,
            )

            print(
                "Returning Dobot out of camera view."
            )

            # Clear the camera view again.
            robot.move_camera_clear()

            time.sleep(
                cfg.AFTER_CYCLE_DELAY
            )

            # Defaults when verification is disabled.
            verify_image_name = ""
            verify_preview_name = ""
            verify_detections_count = 0
            verify_inference_time_ms = 0.0
            auto_success = None

            # Run post-pick verification if enabled.
            if cfg.VERIFY_AFTER_PICK:
                (
                    verify_image_name,
                    verify_preview_name,
                    verify_detections_count,
                    verify_inference_time_ms,
                    auto_success,
                ) = verify_pick(
                    detector=detector,
                    homography=homography,
                    logger=logger,
                    selected=selected,
                    cycle=cycle,
                )

            # Stop timing before asking the user questions.
            cycle_time_ms = (
                                    time.perf_counter()
                                    - cycle_start
                            ) * 1000.0

            print()
            print("MANUAL RESULT")

            # Record what actually happened physically.
            manual_success = ask_yes_no(
                "Was the block successfully picked "
                "and removed from the workspace?"
            )

            if manual_success:
                correct_bin = ask_yes_no(
                    "Did the block finish in the "
                    "correct bin?"
                )
            else:
                correct_bin = False

            # Save the complete attempt.
            logger.log_attempt(
                cycle=cycle,
                pre_image_name=pre_image_path.name,
                pre_preview_name=pre_preview_path.name,
                verify_image_name=verify_image_name,
                verify_preview_name=verify_preview_name,
                detections_before=len(
                    detections
                ),
                detections_after=(
                    verify_detections_count
                ),
                selected=selected,
                inference_time_ms=(
                    inference_time_ms
                ),
                verify_inference_time_ms=(
                    verify_inference_time_ms
                ),
                cycle_time_ms=cycle_time_ms,
                attempted=True,
                auto_success=auto_success,
                manual_success=manual_success,
                correct_bin=correct_bin,
                notes="",
            )

    print()
    print("SORTING RUN FINISHED")

    print(
        f"Logs saved under: "
        f"{logger.run_log_dir}"
    )

    print()
    print("FINAL SCENE RESULT")

    # Record the result for the whole scene.
    scene_success = ask_yes_no(
        "Were all starting blocks successfully "
        "sorted into the correct bins?"
    )

    blocks_remaining = ask_non_negative_int(
        "How many blocks remain in the workspace?"
    )

    logger.log_run_summary(
        scene_success=scene_success,
        blocks_remaining=blocks_remaining,
    )

    print()
    print(
        f"Run summary saved: "
        f"{logger.run_summary_csv}"
    )
