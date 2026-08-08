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


def make_preview_path(
        stage: str,
        cycle: int,
):
    """Create a unique path for a preview image."""
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    return (
            cfg.PREVIEW_DIR
            / f"{stage}_cycle_{cycle:02d}_{timestamp}.jpg"
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
    # Detect objects in the image and store the results.
    detections, inference_time_ms = detector.detect(
        image
    )
    # Add the robot coordinates to the detections.
    add_robot_coordinates(
        detections,
        homography,
    )
    # Assign the detections to bins.
    assign_bins(detections)

    # return the image path, image, detections, and inference time
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
    """Save annotated preview and save detection rows to CSV"""
    # Create a unique preview path for this stage.
    preview_path = make_preview_path(
        stage=stage,
        cycle=cycle,
    )
    # save the annotated preview
    save_preview(
        image=image,
        detections=detections,
        selected=selected,
        preview_path=preview_path,
    )
    # save the detections to CSV
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
        prefix="verify",
    )
    # Save verification image and detections
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
    # Check whether the selected class disappeared after the pickup.
    # This is used as the automatic success proxy.
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


def run_workspace_sorting(
        detector: BaseDetector,
):
    """Run the complete robotic workspace sorting process."""

    # validate runtime config
    cfg.ensure_runtime_config_ready()

    # load H
    homography = load_homography()

    # create experiment logger
    logger = ExperimentLogger(
        method=detector.name
    )

    print()
    print(
        f"STARTING {detector.name.upper()} "
        f"WORKSPACE SORTING"
    )
    print("red/yellow -> warm_bin")
    print("blue/green -> cool_bin")
    print()

    # Connect to the dobot magician lite
    with DobotController() as robot:
        print("Moving Dobot out of camera view.")
        # Move robot out of camera view
        robot.move_camera_clear()
        time.sleep(0.5)

        # main sorting loop
        for cycle in range(
                1,
                cfg.MAX_CYCLES + 1,
        ):
            # start timer for cycle
            cycle_start = time.perf_counter()

            print()
            print(f"===== CYCLE {cycle} =====")

            (
                # prepick image
                pre_image_path,
                pre_image,
                detections,
                inference_time_ms,
            ) = capture_and_prepare(
                detector=detector,
                homography=homography,
                prefix="pre_pick",
            )
            # choose the next target to pick
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
                # Calculate cycle duration and save attempt log
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
                    detections_before=len(detections),
                    detections_after=0,
                    selected=None,
                    inference_time_ms=inference_time_ms,
                    verify_inference_time_ms=0.0,
                    cycle_time_ms=cycle_time_ms,
                    attempted=False,
                    auto_success=None,
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
            # print the selected target if there is one
            print_selected_target(selected)

            # error handling for bins
            if selected.bin_name is None:
                raise RuntimeError(
                    "Selected target does not have a bin."
                )

            drop_pose = cfg.DROP_BINS[
                selected.bin_name
            ]
            # check if the drop pose is configured
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
            # move robot out of camera view
            robot.move_camera_clear()
            time.sleep(cfg.AFTER_CYCLE_DELAY)

            # create defaults because verify after pick is optional
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
            # Calculate cycle duration and save attempt log
            cycle_time_ms = (
                                    time.perf_counter()
                                    - cycle_start
                            ) * 1000.0

            logger.log_attempt(
                cycle=cycle,
                pre_image_name=pre_image_path.name,
                pre_preview_name=pre_preview_path.name,
                verify_image_name=verify_image_name,
                verify_preview_name=verify_preview_name,
                detections_before=len(detections),
                detections_after=verify_detections_count,
                selected=selected,
                inference_time_ms=inference_time_ms,
                verify_inference_time_ms=(
                    verify_inference_time_ms
                ),
                cycle_time_ms=cycle_time_ms,
                attempted=True,
                auto_success=auto_success,
                notes="",
            )

    print()
    print("SORTING RUN FINISHED")
    print(
        f"Logs saved under: "
        f"{logger.run_log_dir}"
    )
