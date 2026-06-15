import math
import threading
from collections.abc import Sequence

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4")

# difference btw ST-initialize-zero and Moveit-zero-point
OFFSETs = (0.0, -math.pi / 2, math.pi / 2, 0.0)


def offset_angle(cmd: Sequence[float]) -> list[float]:
    """
    The same initial pose of real arm btw ST and Moveit
            |
            |
        ____|
        |
    =========

    but their data differ:

        ST          Moveit_plan
        j1:0        j1:0
        j2:0        j2:-1.5708       ---> j2(ST) = j2(Moveit) -1.5708
        j3:0        j3:1.5708        ---> j3(ST) = j3(Moveit) +1.5708
        j4:0        j4:0
    """
    assert len(cmd) == 4, "Command must have 4 elements"
    return [cmdi - offset for cmdi, offset in zip(cmd, OFFSETs)]


class MoveitRealArmInterface(Node):
    def __init__(self):
        super().__init__("joint_position_pub")
        self.eef_state: bool = False
        self._command_lock = threading.Lock()
        self._last_joint_command: list[float] | None = None
        self._warned_missing_joint_state = False

        self.pub = self.create_publisher(
            Float64MultiArray,
            "/real_robot_arm_joint",
            10,
        )

        self.eef_sub = self.create_subscription(
            Bool,
            "/SetEndEffector",
            self.eef_callback,
            10,
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )
        self.get_logger().info(
            "STM32 bridge mirroring /joint_states and /SetEndEffector"
        )

    def eef_callback(self, msg: Bool):
        with self._command_lock:
            self.eef_state = msg.data
            if self._last_joint_command is None:
                self.get_logger().info(
                    f"Stored end effector state: {self.eef_state}; "
                    "waiting for the first joint command"
                )
                return

            self._publish_command_locked(self._last_joint_command)
        self.get_logger().info(
            f"Published immediate end effector state: {self.eef_state}"
        )

    def joint_state_callback(self, msg: JointState):
        joint_positions = dict(zip(msg.name, msg.position))
        try:
            positions = [joint_positions[name] for name in JOINT_NAMES]
        except KeyError as e:
            if not self._warned_missing_joint_state:
                self.get_logger().warn(f"Ignored joint state missing {e}")
                self._warned_missing_joint_state = True
            return

        self._publish_joint_positions(positions)

    def _publish_joint_positions(self, positions: Sequence[float]) -> None:
        joint_command = offset_angle(positions)
        with self._command_lock:
            self._last_joint_command = joint_command
            self._publish_command_locked(joint_command)

    def _publish_command_locked(self, joint_command: Sequence[float]) -> None:
        self.pub.publish(
            Float64MultiArray(
                data=list(joint_command) + [float(self.eef_state)]
            )
        )


def main(args=None):
    rclpy.init(args=args)

    node = MoveitRealArmInterface()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
