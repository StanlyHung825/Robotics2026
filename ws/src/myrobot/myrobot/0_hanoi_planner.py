import threading
import time

import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion
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

from myrobot.hanoi_model import (
    HANOI_TOWER_NAMES,
    HOME_POSITION,
    JOINT_NAMES,
    MOTION_DELAY,
    OBSTACLE_POSITIONS,
    OBSTACLE_SIZE,
    STATION_POSITIONS,
    Tower_base,
    Tower_height,
    Tower_overlap,
)
from myrobot.hanoi_waypoint_planning import HanoiTowerWaypointPlanner
from myrobot.moveit2_acm_management import MoveIt2AcmManager
from myrobot.progress_handling import HanoiProgressHandler

try:
    from moveit_msgs.action import MoveGroupSequence
    from moveit_msgs.msg import MotionSequenceItem, MotionSequenceRequest
except ImportError:
    MoveGroupSequence = None
    MotionSequenceItem = None
    MotionSequenceRequest = None


BOX_SIZE = OBSTACLE_SIZE
BOX_POSITIONS = OBSTACLE_POSITIONS
BOX_NAMES = tuple(f"box_{index}" for index in range(1, len(BOX_POSITIONS) + 1))


class MoveGroupPythonInterface(Node):
    def __init__(self, executor: MultiThreadedExecutor):
        super().__init__("move_group_python_interface")

        self.initialized = False
        self.joint_angles: list[float] | None = None
        self._executor = executor
        self.callback_group = ReentrantCallbackGroup()

        self.GROUP_NAME = "ldsc_arm"
        self.PLANNING_FRAME = "world"
        self.declare_parameter("waypoint_blend_radius", 0.00)
        self.declare_parameter("motion_delay", MOTION_DELAY)
        self.declare_parameter("stop_at_transfer_waypoints", False)
        self.declare_parameter("stop_at_boundary_waypoints", False)
        self.declare_parameter("ompl_fallback_enabled", True)
        self.declare_parameter(
            "pilz_planning_pipeline_id",
            "pilz_industrial_motion_planner",
        )
        self.declare_parameter("pilz_planner_id", "PTP")
        self.declare_parameter("ompl_planning_pipeline_id", "ompl")
        self.declare_parameter("ompl_planner_id", "RRTConnectkConfigDefault")
        self.declare_parameter("planning_attempts", 20)
        self.declare_parameter("allowed_planning_time", 5.0)

        self.WAYPOINT_BLEND_RADIUS = float(
            self.get_parameter("waypoint_blend_radius").value
        )
        self.MOTION_DELAY = float(self.get_parameter("motion_delay").value)
        self.STOP_AT_TRANSFER_WAYPOINTS = bool(
            self.get_parameter("stop_at_transfer_waypoints").value
        )
        self.STOP_AT_BOUNDARY_WAYPOINTS = bool(
            self.get_parameter("stop_at_boundary_waypoints").value
        )
        self.OMPL_FALLBACK_ENABLED = bool(
            self.get_parameter("ompl_fallback_enabled").value
        )
        self.PILZ_PLANNING_PIPELINE_ID = str(
            self.get_parameter("pilz_planning_pipeline_id").value
        )
        self.PILZ_PLANNER_ID = str(self.get_parameter("pilz_planner_id").value)
        self.OMPL_PLANNING_PIPELINE_ID = str(
            self.get_parameter("ompl_planning_pipeline_id").value
        )
        self.OMPL_PLANNER_ID = str(self.get_parameter("ompl_planner_id").value)
        self.PLANNING_ATTEMPTS = int(self.get_parameter("planning_attempts").value)
        self.ALLOWED_PLANNING_TIME = float(
            self.get_parameter("allowed_planning_time").value
        )
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
            scene_initializer=self.spawn_hanoi_scene,
            planner=HanoiTowerWaypointPlanner(
                stop_at_transfer_waypoints=self.STOP_AT_TRANSFER_WAYPOINTS,
                stop_at_boundary_waypoints=self.STOP_AT_BOUNDARY_WAYPOINTS,
            ),
            motion_delay=self.MOTION_DELAY,
            callback_group=self.callback_group,
        )

        self._wait_for_joint_states()
        self._wait_for_trajectory_action_server()
        self._wait_for_sequence_action_server()

        self.scene_manager.preload_hanoi_tower_meshes()
        self.scene_manager.allow_hanoi_contacts()

        self.get_logger().info("MoveGroup Python Interface already initialized")
        self.get_logger().info("Waiting for /set_hanoi_tower_stations requests")
        self.initialized = True

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
            if not getattr(self, "initialized", False):
                rclpy.spin_once(self, timeout_sec=0.01)
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

        if self._execute_single_joint_goal(
            joint_angles,
            planner_id=self.PILZ_PLANNER_ID,
            planning_pipeline_id=self.PILZ_PLANNING_PIPELINE_ID,
            planner_label="Pilz",
        ):
            return True

        if not self.OMPL_FALLBACK_ENABLED:
            return False

        self.get_logger().warn(
            "Pilz single-goal planning failed; retrying with OMPL fallback"
        )
        return self._execute_single_joint_goal(
            joint_angles,
            planner_id=self.OMPL_PLANNER_ID,
            planning_pipeline_id=self.OMPL_PLANNING_PIPELINE_ID,
            planner_label="OMPL",
        )

    def _execute_single_joint_goal(
        self,
        joint_angles: tuple[float, float, float, float],
        *,
        planner_id: str,
        planning_pipeline_id: str,
        planner_label: str,
    ) -> bool:
        self.scene_manager.allow_hanoi_contacts(log=False)

        goal_msg = MoveGroup.Goal(
            request=self._build_motion_plan_request(
                joint_angles,
                planner_id=planner_id,
                planning_pipeline_id=planning_pipeline_id,
            ),
            planning_options=PlanningOptions(plan_only=False, replan=True),
        )

        future = self.action_client.send_goal_async(goal_msg)
        if not self.wait_for_future(future):
            self.get_logger().error(f"Timed out while sending {planner_label} goal")
            return False

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"{planner_label} goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        if not self.wait_for_future(result_future):
            self.get_logger().error(
                f"Timed out while waiting for {planner_label} motion result"
            )
            return False

        result = result_future.result().result
        if result.error_code.val == 1:
            self.get_logger().info(f"{planner_label} motion executed successfully")
            return True

        self.get_logger().warn(
            f"{planner_label} motion failed with error code: {result.error_code.val}"
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

        if self.WAYPOINT_BLEND_RADIUS > 0.0:
            if self._execute_joint_sequence(
                joint_angle_sequence,
                blend_radius=self.WAYPOINT_BLEND_RADIUS,
                planner_id=self.PILZ_PLANNER_ID,
                planning_pipeline_id=self.PILZ_PLANNING_PIPELINE_ID,
                planner_label="Pilz",
            ):
                return True

            self.get_logger().warn(
                "Blended waypoint sequence failed; retrying without blend radius"
            )

        if self._execute_joint_sequence(
            joint_angle_sequence,
            blend_radius=0.0,
            planner_id=self.PILZ_PLANNER_ID,
            planning_pipeline_id=self.PILZ_PLANNING_PIPELINE_ID,
            planner_label="Pilz",
        ):
            return True

        if self.OMPL_FALLBACK_ENABLED:
            self.get_logger().warn(
                "Pilz waypoint sequence failed; retrying with OMPL fallback"
            )
            if self.WAYPOINT_BLEND_RADIUS > 0.0 and self._execute_joint_sequence(
                joint_angle_sequence,
                blend_radius=self.WAYPOINT_BLEND_RADIUS,
                planner_id=self.OMPL_PLANNER_ID,
                planning_pipeline_id=self.OMPL_PLANNING_PIPELINE_ID,
                planner_label="OMPL",
            ):
                return True

            if self._execute_joint_sequence(
                joint_angle_sequence,
                blend_radius=0.0,
                planner_id=self.OMPL_PLANNER_ID,
                planning_pipeline_id=self.OMPL_PLANNING_PIPELINE_ID,
                planner_label="OMPL",
            ):
                return True

        self.get_logger().warn(
            "Waypoint sequence failed; falling back to single-point goals"
        )
        return self._go_through_joint_states_with_stops(joint_angle_sequence)

    def _execute_joint_sequence(
        self,
        joint_angle_sequence: list[tuple[float, float, float, float]],
        *,
        blend_radius: float,
        planner_id: str,
        planning_pipeline_id: str,
        planner_label: str,
    ) -> bool:
        self.scene_manager.allow_hanoi_contacts(log=False)

        sequence_items = []
        final_index = len(joint_angle_sequence) - 1
        for index, joint_angles in enumerate(joint_angle_sequence):
            sequence_items.append(
                MotionSequenceItem(
                    req=self._build_motion_plan_request(
                        joint_angles,
                        planner_id=planner_id,
                        planning_pipeline_id=planning_pipeline_id,
                    ),
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
            self.get_logger().error(
                f"Timed out while sending {planner_label} sequence goal"
            )
            return False

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"{planner_label} sequence goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        if not self.wait_for_future(result_future):
            self.get_logger().error(
                f"Timed out while waiting for {planner_label} sequence result"
            )
            return False

        result = result_future.result().result
        error_code = self._extract_moveit_error_code(result)
        if error_code == 1:
            self.get_logger().info(
                f"{planner_label} waypoint sequence executed successfully"
            )
            return True

        self.get_logger().warn(
            f"{planner_label} waypoint sequence failed with error code: {error_code}"
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

    def spawn_hanoi_scene(
        self,
        tower_stations: tuple[int, ...],
        obstacles: tuple[bool, ...],
    ) -> None:
        self.get_logger().info(
            "Refreshing Hanoi tower poses and obstacle state in planning scene"
        )

        self.scene_manager.remove_attached_objects(object_names=HANOI_TOWER_NAMES)

        stacks = self._build_stacks_from_tower_stations(tower_stations)
        tower_spacing = Tower_height - Tower_overlap
        tower_positions = {}
        for station_index, stack in enumerate(stacks):
            station_x, station_y = STATION_POSITIONS[station_index]
            for stack_index, tower_name in enumerate(stack):
                tower_positions[tower_name] = Point(
                    x=station_x,
                    y=station_y,
                    z=Tower_base + stack_index * tower_spacing,
                )
        self.scene_manager.move_world_meshes(tower_positions)

        enabled_boxes = {}
        for index, enabled in enumerate(obstacles):
            if index >= len(BOX_POSITIONS):
                break
            if not enabled:
                continue

            x, y, z = BOX_POSITIONS[index]
            enabled_boxes[BOX_NAMES[index]] = (
                Pose(
                    orientation=Quaternion(w=1.0),
                    position=Point(x=x, y=y, z=z),
                ),
                BOX_SIZE,
            )
        self.scene_manager.refresh_world_boxes(
            all_object_names=BOX_NAMES,
            enabled_boxes=enabled_boxes,
        )

        self.scene_manager.allow_hanoi_contacts(log=False)
        self.get_logger().info("Hanoi planning scene is ready")

    @staticmethod
    def _build_stacks_from_tower_stations(
        tower_stations: tuple[int, ...],
    ) -> list[list[str]]:
        stacks: list[list[str]] = [[] for _ in STATION_POSITIONS]
        for tower_name, station in zip(HANOI_TOWER_NAMES, tower_stations):
            stacks[station].append(tower_name)
        return stacks

    def _build_motion_plan_request(
        self,
        joint_angles: tuple[float, float, float, float],
        planner_id: str,
        planning_pipeline_id: str,
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
            num_planning_attempts=self.PLANNING_ATTEMPTS,
            allowed_planning_time=self.ALLOWED_PLANNING_TIME,
            max_velocity_scaling_factor=1.0,
            max_acceleration_scaling_factor=1.0,
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
