import time
from serial.tools import list_ports
from pydobot import Dobot
from pydobot.message import Message
from robot_sorting import config as cfg
from robot_sorting.config import Pose
from robot_sorting.geometry import validate_robot_xy
from robot_sorting.types import Detection


class DobotController:
    """Control the Dobot Magician Lite."""

    def __init__(self):
        self.device = None

    def __enter__(self):
        """Connect to the Dobot using the with statement."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the connection when exiting the with statement."""
        self.close()

    @staticmethod
    def find_port():
        """Find the Dobot serial port."""
        ports = list(list_ports.comports())

        for p in ports:
            if "ACM" in p.device:
                return p.device

        for p in ports:
            if "USB" in p.device:
                return p.device
        # Fall back to the configured default port.
        return cfg.DEFAULT_PORT

    def connect(self):
        """Open communication with the Dobot."""
        port = self.find_port()
        print(f"Connecting to Dobot on {port}...")
        self.device = Dobot(port=port, verbose=False)
        # Always start with suction off.
        self.suction_off()
        time.sleep(0.3)

    def close(self):
        """Close communication with the Dobot."""
        if self.device is None:
            return

        try:
            self.suction_off()
        except Exception:
            pass

        try:
            self.device.close()
        except Exception:
            pass

        self.device = None

    def _send_suction_raw(self, ctrl_enabled, sucked):
        """Send a raw suction command to the Dobot."""
        msg = Message()
        # Dobot suction-control command.
        msg.id = 62
        # Set the control flags.
        msg.ctrl = 0x03
        # Build the command parameters.
        msg.params = bytearray([ctrl_enabled, sucked])
        # Send the command.
        self.device._send_command(msg)

    def suction_on(self):
        """Turn suction on."""
        self._send_suction_raw(1, 1)

    def suction_off(self):
        """Turn suction off."""
        self._send_suction_raw(0, 0)

    def move_pose(self, pose: Pose):
        """Move the Dobot to a Pose."""
        print(f"Move pose: X={pose.x:.3f}, Y={pose.y:.3f}, Z={pose.z:.3f}, R={pose.r:.3f}")
        # Wait for the move to finish.
        self.device.move_to(pose.x, pose.y, pose.z, pose.r, wait=True)

    def move_xyz(self, x, y, z, r=cfg.TARGET_R):
        """Move directly to X, Y, Z and R coordinates."""
        print(f"Move: X={x:.3f}, Y={y:.3f}, Z={z:.3f}, R={r:.3f}")
        # Wait for the move to finish.
        self.device.move_to(x, y, z, r, wait=True)

    def move_camera_clear(self):
        """Move the Dobot to the camera clear position."""
        if cfg.CAMERA_CLEAR_POSE is None:
            raise RuntimeError("CAMERA_CLEAR_POSE is not configured.")
        self.move_pose(cfg.CAMERA_CLEAR_POSE)

    def pick_and_drop(self, detection: Detection, drop_pose: Pose):
        """Pick one selected block and move it to its bin."""

        # Robot coordinates must already be mapped.
        if detection.robot_x is None or detection.robot_y is None:
            raise RuntimeError("Detection does not have robot coordinates.")

        robot_x = detection.robot_x
        robot_y = detection.robot_y

        # Check the mapped pickup position.
        validate_robot_xy(robot_x, robot_y)

        # Build the pickup and drop movement poses.
        travel_z = getattr(cfg, "TRAVEL_Z", 70.0)
        drop_above = Pose(drop_pose.x, drop_pose.y, travel_z, drop_pose.r)
        pick_above = Pose(robot_x, robot_y, cfg.SAFE_Z, cfg.TARGET_R)
        pick_travel = Pose(robot_x, robot_y, travel_z, cfg.TARGET_R)

        # Print the planned movement.
        print()
        print("PICK AND DROP")
        print(f"Colour: {detection.class_name}")
        print(f"Pick:   X={robot_x:.3f}, Y={robot_y:.3f}")
        print(f"Drop:   X={drop_pose.x:.3f}, Y={drop_pose.y:.3f}, Z={drop_pose.z:.3f}")
        print(f"Travel Z: {travel_z:.3f}")
        print()

        # Start with suction off.
        self.suction_off()
        time.sleep(0.2)

        # Move above the block.
        self.move_pose(pick_above)

        # Lower to pickup height.
        self.move_xyz(robot_x, robot_y, cfg.PICKUP_Z, cfg.TARGET_R)
        time.sleep(cfg.PRE_SUCTION_SETTLE_TIME)

        # Grab the block.
        print("Suction ON")
        self.suction_on()
        time.sleep(cfg.SUCTION_GRAB_TIME)

        # Lift to the safe height.
        self.move_pose(pick_above)

        # Lift to travel height.
        self.move_pose(pick_travel)

        # Move above the bin.
        self.move_pose(drop_above)

        # Lower to release height.
        self.move_pose(drop_pose)

        # Release the block.
        print("Suction OFF / release")
        self.suction_off()
        time.sleep(cfg.POST_RELEASE_TIME)

        # Move back above the bin.
        self.move_pose(drop_above)

    def get_pose(self) -> Pose:
        """Return the current Dobot pose."""
        if self.device is None:
            raise RuntimeError("Dobot is not connected.")

        pose = self.device.pose()
        x, y, z, r = pose[:4]

        return Pose(
            x=float(x),
            y=float(y),
            z=float(z),
            r=float(r),
        )
