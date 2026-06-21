import sys
from math import pi

import rclpy
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
)
from moveit_msgs.srv import GetMotionPlan
from rclpy.action import ActionClient
from rclpy.node import Node

from myrobot.hanoi_kinematics import ArmKinematics
from myrobot.hanoi_model import Joint_NAMES

JOINT_GOAL_TOLERANCE = 0.012
PLANNER_ATTEMPTS = (
    ("Pilz PTP", "PTP", "pilz_industrial_motion_planner"),
    ("OMPL fallback", "", "ompl"),
)


class MoveGroupPythonInterface(Node):
    def __init__(self):
        super().__init__("move_group_python_interface")

        self.planning_client = self.create_client(
            GetMotionPlan,
            "plan_kinematic_path",
        )
        self.execute_trajectory_client = ActionClient(
            self,
            ExecuteTrajectory,
            "execute_trajectory",
        )

        self.GROUP_NAME = "ldsc_arm"

        self.get_logger().info("Waiting for plan_kinematic_path service...")
        self.planning_client.wait_for_service()
        self.get_logger().info("Waiting for execute_trajectory action server...")
        self.execute_trajectory_client.wait_for_server()
        self.get_logger().info("Motion planning interface initialized")

    def go_to_joint_state(
        self,
        joint_angles: tuple[float, float, float, float],
    ) -> bool:
        planned_trajectory = self._plan_joint_goal(joint_angles)
        if planned_trajectory is None:
            self.get_logger().error(
                "Unable to plan motion to the requested joint state"
            )
            return False
        if not self._trajectory_reaches_joint_goal(planned_trajectory, joint_angles):
            self.get_logger().error(
                "Planned trajectory does not reach the requested joint state; "
                "no motion executed"
            )
            return False

        return self._execute_planned_trajectory(planned_trajectory)

    def _plan_joint_goal(
        self,
        joint_angles: tuple[float, float, float, float],
    ):
        for planner_name, planner_id, planning_pipeline_id in PLANNER_ATTEMPTS:
            self.get_logger().info(f"Trying {planner_name} planner")
            planned_trajectory = self._send_planning_goal(
                joint_angles,
                planner_id=planner_id,
                planning_pipeline_id=planning_pipeline_id,
            )
            if planned_trajectory is None:
                continue
            if not self._trajectory_reaches_joint_goal(
                planned_trajectory,
                joint_angles,
            ):
                self.get_logger().warn(
                    f"{planner_name} produced a trajectory that does not reach "
                    "the requested joint goal"
                )
                continue

            self.get_logger().info(f"Using trajectory from {planner_name}")
            return planned_trajectory

        return None

    def _send_planning_goal(
        self,
        joint_angles: tuple[float, float, float, float],
        *,
        planner_id: str,
        planning_pipeline_id: str,
    ):
        joint_constraints = [
            JointConstraint(
                joint_name=name,
                position=angle,
                tolerance_above=JOINT_GOAL_TOLERANCE,
                tolerance_below=JOINT_GOAL_TOLERANCE,
                weight=1.0,
            )
            for name, angle in zip(Joint_NAMES, joint_angles)
        ]
        constraints = Constraints(joint_constraints=joint_constraints)

        motion_plan_request = MotionPlanRequest(
            group_name=self.GROUP_NAME,
            num_planning_attempts=10,
            allowed_planning_time=5.0,
            max_velocity_scaling_factor=0.5,
            max_acceleration_scaling_factor=0.5,
            goal_constraints=[constraints],
        )
        if planner_id:
            motion_plan_request.planner_id = planner_id
        if hasattr(motion_plan_request, "pipeline_id"):
            motion_plan_request.pipeline_id = planning_pipeline_id

        future = self.planning_client.call_async(
            GetMotionPlan.Request(motion_plan_request=motion_plan_request)
        )
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response is None:
            self.get_logger().warn(
                f"Planning service returned no response for {planning_pipeline_id}"
            )
            return None

        motion_plan_response = response.motion_plan_response
        if motion_plan_response.error_code.val == 1:
            self.get_logger().info(
                f"Motion planned successfully with {planning_pipeline_id}"
            )
            return motion_plan_response.trajectory
        else:
            self.get_logger().warn(
                f"Planning failed with {planning_pipeline_id} "
                f"(error code: {motion_plan_response.error_code.val})"
            )
            return None

    def _trajectory_reaches_joint_goal(
        self,
        planned_trajectory,
        joint_angles: tuple[float, float, float, float],
    ) -> bool:
        joint_trajectory = planned_trajectory.joint_trajectory
        if not joint_trajectory.points:
            self.get_logger().warn("Planner returned an empty trajectory")
            return False

        final_point = joint_trajectory.points[-1]
        final_positions = dict(
            zip(joint_trajectory.joint_names, final_point.positions)
        )
        target_positions = dict(zip(Joint_NAMES, joint_angles))
        missing_joints = [
            joint_name
            for joint_name in Joint_NAMES
            if joint_name not in final_positions
        ]
        if missing_joints:
            self.get_logger().warn(
                f"Planned trajectory is missing joints: {missing_joints}"
            )
            return False

        max_error = max(
            abs(final_positions[joint_name] - target_positions[joint_name])
            for joint_name in Joint_NAMES
        )
        if max_error > JOINT_GOAL_TOLERANCE:
            self.get_logger().warn(
                f"Planned trajectory final joint error {max_error:.4f} rad exceeds "
                f"tolerance {JOINT_GOAL_TOLERANCE:.4f} rad"
            )
            return False

        return True

    def _execute_planned_trajectory(self, planned_trajectory) -> bool:
        goal_msg = ExecuteTrajectory.Goal(trajectory=planned_trajectory)

        self.get_logger().info("Executing verified planned trajectory")
        future = self.execute_trajectory_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Planned trajectory execution rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.error_code.val == 1:
            self.get_logger().info("Planned trajectory executed successfully")
            return True

        self.get_logger().error(
            f"Planned trajectory execution failed with error code: "
            f"{result.error_code.val}"
        )
        return False


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
    return ArmKinematics().solve(x, y, z, pitch)


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
                print("No motion executed.")

            except Exception as e:
                print(f"Error occurred: {e}")
                print("No motion executed.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        if "path_object" in locals():
            path_object.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
