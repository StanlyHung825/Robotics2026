import math
import time
from collections.abc import Sequence

from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4")
DEFAULT_ACTION_NAME = "ldsc_arm_controller/follow_joint_trajectory"

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

        self.declare_parameter("action_name", DEFAULT_ACTION_NAME)
        action_name = (
            self.get_parameter("action_name").get_parameter_value().string_value
            or DEFAULT_ACTION_NAME
        )

        self.pub = self.create_publisher(Float64MultiArray, "/real_robot_arm_joint", 10)

        self.action_server = ActionServer(
            self,
            FollowJointTrajectory,
            action_name,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )
        self.get_logger().info(
            f"Real arm FollowJointTrajectory server ready on {action_name}"
        )

    def destroy_node(self):
        self.action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        trajectory = goal_request.trajectory
        if not trajectory.points:
            self.get_logger().warn("Rejected empty trajectory goal")
            return GoalResponse.REJECT

        missing_joints = [
            joint_name
            for joint_name in JOINT_NAMES
            if joint_name not in trajectory.joint_names
        ]
        if missing_joints:
            self.get_logger().warn(
                f"Rejected trajectory missing joints: {missing_joints}"
            )
            return GoalResponse.REJECT

        self.get_logger().info(
            f"Accepted final execution trajectory with {len(trajectory.points)} points"
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle):
        self.get_logger().info("Cancel request accepted")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        result = FollowJointTrajectory.Result()

        start_time = self._trajectory_start_time(trajectory)
        joint_indices = [trajectory.joint_names.index(name) for name in JOINT_NAMES]

        for point in trajectory.points:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = "Trajectory canceled"
                return result

            if len(point.positions) < len(trajectory.joint_names):
                goal_handle.abort()
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = "Trajectory point has too few positions"
                return result

            wait = start_time + self._duration_to_sec(point.time_from_start)
            time.sleep(max(0.0, wait - time.monotonic()))

            positions = [point.positions[index] for index in joint_indices]
            self.pub.publish(Float64MultiArray(data=offset_angle(positions)))
            self.get_logger().info(
                f"t: {point.time_from_start}; cmd: {positions}"
            )

        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = "Trajectory streamed to real arm"
        return result

    def _trajectory_start_time(self, trajectory) -> float:
        stamp = trajectory.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            return time.monotonic()

        stamp_sec = stamp.sec + stamp.nanosec * 1e-9
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        return time.monotonic() + max(0.0, stamp_sec - now_sec)

    @staticmethod
    def _duration_to_sec(duration) -> float:
        return duration.sec + duration.nanosec * 1e-9


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
