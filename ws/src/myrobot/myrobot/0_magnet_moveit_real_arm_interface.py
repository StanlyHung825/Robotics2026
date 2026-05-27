import math
import time
from collections.abc import Sequence

import rclpy
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint

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

        self.pub = self.create_publisher(Float64MultiArray, "/real_robot_arm_joint", 10)

        self.eef_sub = self.create_subscription(Bool, "/SetEndEffector", self.eef_callback, 10)

        self.traj_sub = self.create_subscription(
            DisplayTrajectory, "/display_planned_path", self.traj_callback, 10
        )

    def eef_callback(self, msg: Bool):
        self.eef_state = msg.data

    def traj_callback(self, msg: DisplayTrajectory):
        traj: RobotTrajectory
        for traj in msg.trajectory:
            assert len(traj.joint_trajectory.points) > 0, "There is no trajectory point"

            # first point
            refs = iter(traj.joint_trajectory.points)
            ref: JointTrajectoryPoint = next(refs)
            self.pub.publish(
                Float64MultiArray(data=offset_angle(ref.positions) + [float(self.eef_state)])
            )

            t_now = ref.time_from_start
            self.get_logger().info(f"t: {t_now}; cmd: {ref.positions}")

            # other point
            for ref in refs:
                t_next = ref.time_from_start
                wait = (t_next.sec - t_now.sec) + (t_next.nanosec - t_now.nanosec) * 1e-9
                time.sleep(wait)
                t_now = t_next

                self.pub.publish(Float64MultiArray(data=offset_angle(ref.positions)))
                self.get_logger().info(f"t: {t_now}; cmd: {ref.positions}")


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
