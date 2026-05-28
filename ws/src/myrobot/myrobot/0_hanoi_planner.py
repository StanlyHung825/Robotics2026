import threading
import time

import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
)
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from myrobot.hanoi_waypoint_planning import HOME_POSITION, JOINT_NAMES
from myrobot.moveit2_acm_management import MoveIt2AcmManager
from myrobot.progress_handling import HanoiProgressHandler

try:
    from moveit_msgs.action import MoveGroupSequence
    from moveit_msgs.msg import MotionSequenceItem, MotionSequenceRequest
except ImportError:
    MoveGroupSequence = None
    MotionSequenceItem = None
    MotionSequenceRequest = None


class MoveGroupPythonInterface(Node):
    def __init__(self, executor: MultiThreadedExecutor):
        super().__init__("move_group_python_interface")

        self.joint_angles: list[float] | None = None
        self._executor = executor
        self.callback_group = ReentrantCallbackGroup()

        self.GROUP_NAME = "ldsc_arm"
        self.PLANNING_FRAME = "world"
        self.WAYPOINT_BLEND_RADIUS = 0.005
        self.JOINT_GOAL_TOLERANCE = 0.005
        self.JOINT_MATCH_TOLERANCE = 0.001

        self.action_client = ActionClient(self, MoveGroup, "move_action")
        self.sequence_action_client = None
        if MoveGroupSequence is not None:
            self.sequence_action_client = ActionClient(
                self,
                MoveGroupSequence,
                "sequence_move_group",
            )

        self.display_trajectory_publisher = self.create_publisher(
            DisplayTrajectory,
            "/move_group/display_planned_path",
            20,
        )
        self.pub_eef_state = self.create_publisher(Bool, "/SetEndEffector", 10)

        self.scene_manager = MoveIt2AcmManager(
            self,
            planning_frame=self.PLANNING_FRAME,
            wait_for_future=self.wait_for_future,
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
            callback_group=self.callback_group,
        )

        self.progress_handler = HanoiProgressHandler(
            node=self,
            motion_interface=self,
            scene_manager=self.scene_manager,
            callback_group=self.callback_group,
        )

        self._wait_for_joint_states()
        self._wait_for_trajectory_action_server()
        self._wait_for_sequence_action_server()

        self.get_logger().info("MoveGroup Python Interface already initialized")
        self.get_logger().info("Waiting for /set_hanoi_tower_stations requests")
        self.scene_manager.allow_hanoi_contacts()

    @property
    def hanoi_busy(self) -> bool:
        return self.progress_handler.busy

    def joint_state_callback(self, msg: JointState) -> None:
        try:
            joint_pair: dict[str, float] = dict(zip(msg.name, msg.position))
            self.joint_angles = [joint_pair[name] for name in JOINT_NAMES]
        except Exception as e:
            self.get_logger().error(f"Error in joint_state_callback: {str(e)}")

    def wait_for_future(self, future, timeout_sec: float = 30.0) -> bool:
        start_time = time.time()
        while rclpy.ok() and not future.done():
            if (time.time() - start_time) > timeout_sec:
                return False
            time.sleep(0.01)
        return future.done()

    def go_to_joint_state(
        self,
        joint_angles: tuple[float, float, float, float],
    ) -> bool:
        if self._is_current_joint_state(joint_angles):
            self.get_logger().info("Target joint state is already reached; skipping")
            return True

        self.scene_manager.allow_hanoi_contacts(log=False)

        motion_plan_request = self._build_motion_plan_request(joint_angles)

        goal_msg = MoveGroup.Goal(
            request=motion_plan_request,
            planning_options=PlanningOptions(plan_only=False, replan=True),
        )

        future = self.action_client.send_goal_async(goal_msg)
        if not self.wait_for_future(future):
            self.get_logger().error("Timed out while sending goal")
            return False

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        if not self.wait_for_future(result_future):
            self.get_logger().error("Timed out while waiting for motion result")
            return False

        result = result_future.result().result
        if result.error_code.val == 1:
            self.get_logger().info("Motion executed successfully")
            return True

        self.get_logger().error(
            f"Motion failed with error code: {result.error_code.val}"
        )
        return False

    def go_through_joint_states(
        self,
        joint_angle_sequence: list[tuple[float, float, float, float]],
    ) -> bool:
        joint_angle_sequence = self._remove_redundant_joint_targets(
            joint_angle_sequence
        )
        if not joint_angle_sequence:
            return True
        if len(joint_angle_sequence) == 1:
            return self.go_to_joint_state(joint_angle_sequence[0])

        if (
            self.sequence_action_client is None
            or MoveGroupSequence is None
            or MotionSequenceItem is None
            or MotionSequenceRequest is None
        ):
            self.get_logger().warn(
                "MoveGroupSequence is unavailable; falling back to single-point goals"
            )
            return self._go_through_joint_states_with_stops(joint_angle_sequence)
        if not self.sequence_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn(
                "MoveGroupSequence server is unavailable; "
                "falling back to single-point goals"
            )
            return self._go_through_joint_states_with_stops(joint_angle_sequence)

        if self._execute_joint_sequence(
            joint_angle_sequence,
            blend_radius=self.WAYPOINT_BLEND_RADIUS,
        ):
            return True

        self.get_logger().warn(
            "Blended waypoint sequence failed; retrying without blend radius"
        )
        if self._execute_joint_sequence(joint_angle_sequence, blend_radius=0.0):
            return True

        self.get_logger().warn(
            "Zero-blend waypoint sequence failed; falling back to single-point goals"
        )
        return self._go_through_joint_states_with_stops(joint_angle_sequence)

    def _execute_joint_sequence(
        self,
        joint_angle_sequence: list[tuple[float, float, float, float]],
        *,
        blend_radius: float,
    ) -> bool:
        self.scene_manager.allow_hanoi_contacts(log=False)

        sequence_items = []
        final_index = len(joint_angle_sequence) - 1
        for index, joint_angles in enumerate(joint_angle_sequence):
            sequence_items.append(
                MotionSequenceItem(
                    req=self._build_motion_plan_request(joint_angles),
                    blend_radius=(
                        0.0
                        if index == final_index
                        else blend_radius
                    ),
                )
            )

        goal_msg = MoveGroupSequence.Goal(
            request=MotionSequenceRequest(items=sequence_items),
            planning_options=PlanningOptions(plan_only=False, replan=True),
        )

        future = self.sequence_action_client.send_goal_async(goal_msg)
        if not self.wait_for_future(future):
            self.get_logger().error("Timed out while sending sequence goal")
            return False

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Sequence goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        if not self.wait_for_future(result_future):
            self.get_logger().error("Timed out while waiting for sequence result")
            return False

        result = result_future.result().result
        error_code = self._extract_moveit_error_code(result)
        if error_code == 1:
            self.get_logger().info("Waypoint sequence executed successfully")
            return True

        self.get_logger().error(
            f"Waypoint sequence failed with error code: {error_code}"
        )
        return False

    def switch_magnet(self, on: bool) -> None:
        self.pub_eef_state.publish(Bool(data=on))
        self.get_logger().info(f"Published end effector state: {on}")

    def allow_hanoi_contacts(self, log: bool = True) -> None:
        self.scene_manager.allow_hanoi_contacts(log=log)

    def remove_world_object(self, object_name: str) -> None:
        self.scene_manager.remove_world_object(object_name)

    def add_world_mesh(self, **kwargs) -> None:
        self.scene_manager.add_world_mesh(**kwargs)

    def attach_object(self, **kwargs) -> None:
        self.scene_manager.attach_object(**kwargs)

    def detach_object(self, **kwargs) -> None:
        self.scene_manager.detach_object(**kwargs)

    def _build_motion_plan_request(
        self,
        joint_angles: tuple[float, float, float, float],
        planner_id: str = "PTP",
        planning_pipeline_id: str = "pilz_industrial_motion_planner",
    ) -> MotionPlanRequest:
        joint_constraints = [
            JointConstraint(
                joint_name=name,
                position=angle,
                tolerance_above=self.JOINT_GOAL_TOLERANCE,
                tolerance_below=self.JOINT_GOAL_TOLERANCE,
                weight=1.0,
            )
            for name, angle in zip(JOINT_NAMES, joint_angles)
        ]
        constraints = Constraints(joint_constraints=joint_constraints)

        request = MotionPlanRequest(
            group_name=self.GROUP_NAME,
            planner_id=planner_id,
            num_planning_attempts=10,
            allowed_planning_time=5.0,
            max_velocity_scaling_factor=0.7,
            max_acceleration_scaling_factor=0.7,
            goal_constraints=[constraints],
        )
        if hasattr(request, "pipeline_id"):
            request.pipeline_id = planning_pipeline_id
        return request

    def _go_through_joint_states_with_stops(
        self,
        joint_angle_sequence: list[tuple[float, float, float, float]],
    ) -> bool:
        return all(
            self.go_to_joint_state(joint_angles)
            for joint_angles in joint_angle_sequence
        )

    def _remove_redundant_joint_targets(
        self,
        joint_angle_sequence: list[tuple[float, float, float, float]],
    ) -> list[tuple[float, float, float, float]]:
        filtered = []
        previous = tuple(self.joint_angles) if self.joint_angles is not None else None
        for joint_angles in joint_angle_sequence:
            if previous is not None and self._joint_distance(
                previous,
                joint_angles,
            ) <= self.JOINT_MATCH_TOLERANCE:
                self.get_logger().info("Skipping redundant sequence waypoint")
                previous = joint_angles
                continue

            filtered.append(joint_angles)
            previous = joint_angles

        return filtered

    def _is_current_joint_state(
        self,
        joint_angles: tuple[float, float, float, float],
    ) -> bool:
        return (
            self.joint_angles is not None
            and self._joint_distance(tuple(self.joint_angles), joint_angles)
            <= self.JOINT_MATCH_TOLERANCE
        )

    @staticmethod
    def _joint_distance(
        first: tuple[float, ...],
        second: tuple[float, ...],
    ) -> float:
        return max(abs(a - b) for a, b in zip(first, second))

    @staticmethod
    def _extract_moveit_error_code(result) -> int | None:
        if hasattr(result, "error_code"):
            return result.error_code.val
        if hasattr(result, "response") and hasattr(result.response, "error_code"):
            return result.response.error_code.val
        return None

    def _wait_for_joint_states(self) -> None:
        self.get_logger().info("Waiting for joint states...")
        timeout = 10.0
        start_time = time.time()
        while True:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.joint_angles is not None:
                self.get_logger().info("Joint states received!")
                break
            if (time.time() - start_time) > timeout:
                self.get_logger().warn("Joint states not received within timeout")
                break

    def _wait_for_trajectory_action_server(self) -> None:
        self.get_logger().info("Waiting for trajectory action server...")
        if self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().info("Trajectory action server connected!")
        else:
            self.get_logger().error("Trajectory action server not available!")

    def _wait_for_sequence_action_server(self) -> None:
        if self.sequence_action_client is None:
            self.get_logger().warn("MoveGroupSequence action is not available")
            return

        self.get_logger().info("Waiting for trajectory sequence action server...")
        if self.sequence_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().info("Trajectory sequence action server connected!")
        else:
            self.get_logger().warn(
                "Trajectory sequence action server not available; "
                "intermediate waypoints will stop"
            )


def main(args=None):
    rclpy.init(args=args)

    executor = MultiThreadedExecutor()

    try:
        path_object = MoveGroupPythonInterface(executor)
        executor.add_node(path_object)
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()

        try:
            while rclpy.ok():
                time.sleep(0.5)
        except KeyboardInterrupt:
            path_object.get_logger().info("Interrupted by user")
        finally:
            path_object.switch_magnet(False)

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
