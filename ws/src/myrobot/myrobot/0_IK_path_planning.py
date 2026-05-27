import sys
from math import acos, atan2, cos, hypot, pi, sin

import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
)
from rclpy.action import ActionClient
from rclpy.node import Node

Joint_NAMES = ("joint1", "joint2", "joint3", "joint4")
LINK_LENGTH = (0.0600, 0.0820, 0.1320, 0.1664, 0.0480, 0.0040)
JOINT_LIMITS = (
    (-1.2, 1.2),
    (-2.0, 2.0),
    (-1.67, 1.67),
    (-pi / 2, pi / 2),
)


class MoveGroupPythonInterface(Node):
    def __init__(self):
        super().__init__("move_group_python_interface")

        self.action_client = ActionClient(self, MoveGroup, "move_action")

        self.GROUP_NAME = "ldsc_arm"

        self.get_logger().info("Waiting for move_group action server...")
        self.action_client.wait_for_server()
        self.get_logger().info("MoveGroup Interface Initialized")

    def go_to_joint_state(
        self,
        joint_angles: tuple[float, float, float, float],
    ) -> None:
        joint_constraints = [
            JointConstraint(
                joint_name=name,
                position=angle,
                tolerance_above=0.01,
                tolerance_below=0.01,
                weight=1.0,
            )
            for name, angle in zip(Joint_NAMES, joint_angles)
        ]
        constraints = Constraints(joint_constraints=joint_constraints)

        motion_plan_request = MotionPlanRequest(
            group_name=self.GROUP_NAME,
            num_planning_attempts=10,
            allowed_planning_time=5.0,
            goal_constraints=[constraints],
        )

        goal_msg = MoveGroup.Goal(
            request=motion_plan_request,
            planning_options=PlanningOptions(plan_only=False, replan=True),
        )

        future = self.action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.error_code.val == 1:
            self.get_logger().info("Motion executed successfully")
        else:
            self.get_logger().error(f"Motion failed with error code: {result.error_code.val}")


def _inside_limit(angle: float, joint_index: int, tolerance: float = 1e-9) -> bool:
    lower, upper = JOINT_LIMITS[joint_index]
    return lower - tolerance <= angle <= upper + tolerance


def _clamp_to_limit(angle: float, joint_index: int) -> float:
    lower, upper = JOINT_LIMITS[joint_index]
    return max(lower, min(upper, angle))


def Your_IK(
    x: float,
    y: float,
    z: float,
    pitch: float = pi / 2,
) -> tuple[float, float, float, float]:
    """
    Analytic IK for the 4-DOF arm described by myrobot.urdf.

    x, y, z and pitch are in the world/link0 frame.  The default pitch keeps
    the tool axis parallel to the ground.
    """
    base_height = LINK_LENGTH[0] + LINK_LENGTH[1]
    upper_arm = LINK_LENGTH[2]
    forearm = LINK_LENGTH[3]
    tool_z = LINK_LENGTH[4]
    tool_x = LINK_LENGTH[5]

    joint1 = atan2(y, x) if hypot(x, y) > 1e-12 else 0.0
    if not _inside_limit(joint1, 0):
        raise ValueError(f"joint1 angle {joint1:.3f} rad exceeds the URDF limit")

    radius = hypot(x, y)

    # Remove the fixed tool offset after joint4.  In the pitch plane, positive
    # joint angles rotate the local z-axis toward positive radial x.
    tool_radius = tool_x * cos(pitch) + tool_z * sin(pitch)
    tool_height = -tool_x * sin(pitch) + tool_z * cos(pitch)
    wrist_radius = radius - tool_radius
    wrist_height = z - base_height - tool_height

    wrist_distance = hypot(wrist_radius, wrist_height)
    min_reach = abs(upper_arm - forearm)
    max_reach = upper_arm + forearm
    if wrist_distance < min_reach - 1e-9 or wrist_distance > max_reach + 1e-9:
        raise ValueError(
            "target is outside the reachable workspace "
            f"(wrist distance {wrist_distance:.3f} m, reachable "
            f"{min_reach:.3f} m to {max_reach:.3f} m)"
        )

    cos_joint3 = (
        wrist_radius**2
        + wrist_height**2
        - upper_arm**2
        - forearm**2
    ) / (2.0 * upper_arm * forearm)
    cos_joint3 = max(-1.0, min(1.0, cos_joint3))

    candidates = []
    for joint3 in (acos(cos_joint3), -acos(cos_joint3)):
        joint2 = atan2(wrist_radius, wrist_height) - atan2(
            forearm * sin(joint3),
            upper_arm + forearm * cos(joint3),
        )
        joint4 = pitch - joint2 - joint3
        joint_angles = (joint1, joint2, joint3, joint4)

        if all(_inside_limit(angle, index) for index, angle in enumerate(joint_angles)):
            ready_pose = (0.0, -pi / 2, pi / 2, 0.0)
            score = sum(abs(angle - ready) for angle, ready in zip(joint_angles, ready_pose))
            candidates.append((score, joint_angles))

    if not candidates:
        raise ValueError("target is reachable geometrically, but violates joint limits")

    _, solution = min(candidates, key=lambda item: item[0])
    return tuple(_clamp_to_limit(angle, index) for index, angle in enumerate(solution))


def main():
    rclpy.init(args=sys.argv)

    try:
        path_object = MoveGroupPythonInterface()

        print("Press Ctrl+C to exit")

        while rclpy.ok():
            try:
                print("\n--- Enter Target Position ---")
                x_input = float(input("x: "))
                y_input = float(input("y: "))
                z_input = float(input("z: "))

                path_object.go_to_joint_state(Your_IK(x_input, y_input, z_input))

            except ValueError as e:
                print(f"Invalid input: {e}")
                print("Moving to Home Position...")
                path_object.go_to_joint_state((0.0, -pi / 2, pi / 2, 0.0))

            except Exception as e:
                print(f"Error occurred: {e}")
                print("Moving to Home Position...")
                path_object.go_to_joint_state((0.0, -pi / 2, pi / 2, 0.0))

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        if "path_object" in locals():
            path_object.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
