import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root so project modules can be imported
# when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.calibration_data import (
    find_next_unfilled_row,
    read_rows,
    write_rows,
)
from robot_sorting import config as cfg
from robot_sorting.dobot_controller import DobotController


def clamp(value, low, high):
    """Keep a value within the given limits. This is different from automatic limits as it is manual clamping."""
    return max(low, min(high, value))


def averaged_pose(robot: DobotController):
    """Return an averaged Dobot position from several pose samples."""
    # Read the current pose several times and average the results.
    samples = []

    for _ in range(cfg.CALIBRATION_POSE_SAMPLES):
        pose = robot.get_pose()

        samples.append(
            (pose.x, pose.y, pose.z)
        )

        time.sleep(
            cfg.CALIBRATION_POSE_SAMPLE_DELAY
        )

    x = sum(p[0] for p in samples) / len(samples)
    y = sum(p[1] for p in samples) / len(samples)
    z = sum(p[2] for p in samples) / len(samples)

    return x, y, z


def print_pose(robot: DobotController):
    """Print the current Dobot position."""
    pose = robot.get_pose()
    # No need for R
    print(
        f"Pose("
        f"x={pose.x:.3f}, "
        f"y={pose.y:.3f}, "
        f"z={pose.z:.3f})"
    )


def print_help(step):
    """Print the calibration jog controls."""
    print()
    print("Jog controls")
    print("w = +X")
    print("s = -X")
    print("a = +Y")
    print("d = -Y")
    print("i = +Z / up")
    print("k = -Z / down")
    print("p = print pose")
    print("e = suction ON")
    print("f = suction OFF")
    print("step 0.5 = set XY jog step to 0.5 mm")
    print("step 1 = set XY jog step to 1 mm")
    print("step 2 = set XY jog step to 2 mm")
    print("save = save averaged robot pose to calibration CSV")
    print("q = quit without saving")
    print()
    print(f"Current XY step: {step:.3f} mm")
    print(
        f"Z step: "
        f"{cfg.CALIBRATION_Z_STEP_MM:.3f} mm"
    )
    print(
        f"R is fixed internally at "
        f"{cfg.TARGET_R:.1f} for all jog movements."
    )
    print()


def save_robot_point(
        row,
        rows,
        robot: DobotController,
):
    """Save the current averaged robot position for the calibration point."""
    print(
        f"Waiting "
        f"{cfg.CALIBRATION_SETTLE_SECONDS:.1f}s "
        f"before recording..."
    )

    time.sleep(
        cfg.CALIBRATION_SETTLE_SECONDS
    )

    x, y, z = averaged_pose(robot)

    row["robot_x"] = f"{x:.3f}"
    row["robot_y"] = f"{y:.3f}"
    row["robot_z"] = f"{z:.3f}"
    row["robot_time"] = datetime.now().isoformat(
        timespec="seconds"
    )

    write_rows(rows)

    print()
    print("ROBOT POINT SAVED")
    print(f"point_id = {row['point_id']}")
    print(f"u        = {row['u']}")
    print(f"v        = {row['v']}")
    print(f"robot_x  = {x:.3f}")
    print(f"robot_y  = {y:.3f}")
    print(f"robot_z  = {z:.3f}")
    print()
    print(
        "R is fixed for movement and is not "
        "stored or used in calibration."
    )
    print()
    print(
        f"CSV: {cfg.CALIBRATION_POINTS_CSV}"
    )


def main():
    """Workflow for manually jogging the robot to a calibration point."""
    rows = read_rows()

    # check to make sure there are rows to work with
    if not rows:
        print(
            "No calibration points found. "
            "Run calib_01_camera_point.py first."
        )
        return

    # next unfilled row
    row = find_next_unfilled_row(rows)

    if row is None:
        print(
            "No unfilled camera point found. "
            "Run calib_01_camera_point.py first."
        )
        return

    # Print reminders
    print()
    print("NEXT CALIBRATION POINT TO FILL")
    print(f"point_id = {row['point_id']}")
    print(f"colour   = {row['colour']}")
    print(f"u        = {row['u']}")
    print(f"v        = {row['v']}")
    print(f"image    = {row['image_name']}")
    print()
    print(
        "Before running this, the Dobot can be "
        "manually placed close to the cube."
    )
    print(
        "Once this script connects, use jog commands only."
    )
    print(
        "Centre the suction cup on the same cube centre, "
        "then type: save"
    )
    print()

    step = cfg.CALIBRATION_DEFAULT_STEP_MM

    with DobotController() as robot:
        print()
        print("Current pose after connection:")
        print_pose(robot)

        print_help(step)

        # Wait for user input before starting jogging
        while True:
            cmd = input(
                "calib jog > "
            ).strip().lower()

            if not cmd:
                continue

            # press q to quit without saving
            if cmd == "q":
                print("Quit without saving.")
                return

            # press h for help
            if cmd in ["h", "help"]:
                print_help(step)
                continue

            # press p for pose
            if cmd == "p":
                print_pose(robot)
                continue

            # press e to turn suction on
            if cmd == "e":
                robot.suction_on()
                print("Suction ON")
                continue

            # press f to turn suction off
            if cmd == "f":
                robot.suction_off()
                print("Suction OFF")
                continue

            # press step to set the XY jog step
            if cmd.startswith("step"):
                parts = cmd.split()

                if len(parts) != 2:
                    print("Use format: step 0.5")
                    continue

                try:
                    step = float(parts[1])

                    if step <= 0:
                        raise ValueError

                    print(
                        f"XY step set to "
                        f"{step:.3f} mm"
                    )

                except ValueError:
                    print("Invalid step value.")

                continue

            if cmd == "save":
                save_robot_point(
                    row,
                    rows,
                    robot,
                )
                return

            # Only accept commands made entirely from valid jog keys.
            # This still allows combinations such as "wwa".
            valid_jog_keys = {
                "w",
                "s",
                "a",
                "d",
                "i",
                "k",
            }

            if any(
                key not in valid_jog_keys
                for key in cmd
            ):
                print(
                    f"Invalid jog command: {cmd}"
                )
                continue

            for key in cmd:
                # Adjust X and Y using the selected jog step.
                # Z uses the fixed calibration Z step.
                pose = robot.get_pose()

                new_x = pose.x
                new_y = pose.y
                new_z = pose.z

                if key == "w":
                    new_x += step

                elif key == "s":
                    new_x -= step

                elif key == "a":
                    new_y += step

                elif key == "d":
                    new_y -= step

                elif key == "i":
                    new_z += (
                        cfg.CALIBRATION_Z_STEP_MM
                    )

                elif key == "k":
                    new_z -= (
                        cfg.CALIBRATION_Z_STEP_MM
                    )

                new_x = clamp(
                    new_x,
                    cfg.CALIBRATION_JOG_X_MIN,
                    cfg.CALIBRATION_JOG_X_MAX,
                )

                new_y = clamp(
                    new_y,
                    cfg.CALIBRATION_JOG_Y_MIN,
                    cfg.CALIBRATION_JOG_Y_MAX,
                )

                new_z = clamp(
                    new_z,
                    cfg.CALIBRATION_JOG_Z_MIN,
                    cfg.CALIBRATION_JOG_Z_MAX,
                )

                try:
                    robot.move_xyz(
                        new_x,
                        new_y,
                        new_z,
                        cfg.TARGET_R,
                    )

                    print(
                        f"Moved {key}: ",
                        end="",
                    )

                    print_pose(robot)

                except Exception as e:
                    print(
                        f"Move failed: {e}"
                    )
                    print("Current pose:")
                    print_pose(robot)


if __name__ == "__main__":
    main()