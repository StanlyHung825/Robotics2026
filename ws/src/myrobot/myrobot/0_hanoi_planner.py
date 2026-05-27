import threading
import time
from dataclasses import dataclass
from math import acos, atan2, cos, hypot, pi, sin
from pathlib import Path
from typing import Literal

from ament_index_python.packages import get_package_share_directory
import rclpy
import trimesh
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    AllowedCollisionEntry,
    AllowedCollisionMatrix,
    CollisionObject,
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    MotionPlanRequest,
    PlanningScene,
    PlanningSceneComponents,
    PlanningOptions,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from myrobot_interfaces.srv import SetHanoiTowerStations
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import Mesh, MeshTriangle
from std_msgs.msg import Bool, Header

Joint_NAMES = ("joint1", "joint2", "joint3", "joint4")
LINK_LENGTH = (0.0600, 0.0820, 0.1320, 0.1664, 0.0480, 0.0040)
JOINT_LIMITS = (
    (-1.2, 1.2),
    (-2.0, 2.0),
    (-1.67, 1.67),
    (-pi / 2, pi / 2),
)
HOME_POSITION = (0.0, -pi / 2, pi / 2, 0.0)

"""Hanoi tower geometry"""
# You can measure these in Lab402
Tower_base = 0.0014  # Height of tower base
Tower_height = 0.025  # Height of each tower
Tower_overlap = 0.015  # Height of tower overlap
Tower_mesh_height = 0.02375  # Actual STL height after the spawn-node mesh scale.
End_effector_contact_offset = 0.0

"""Hanoi tower station position"""
# You may want to slightly change this
STATION_POSITIONS = (
    (0.25, 0.15),
    (0.25, 0.0),
    (0.25, -0.15),
)
NUM_DISKS = 3
SOURCE_STATION = 1
TARGET_STATION = 0
APPROACH_HEIGHT = 0.08
MOTION_DELAY = 0.2
TOOL_LINK = "link5"
ROBOT_LINKS = ("link0", "link1", "link2", "link3", "link4", "link5")
HANOI_TOWER_NAMES = tuple(f"tower{index}" for index in range(1, NUM_DISKS + 1))
SRDF_ALLOWED_LINK_PAIRS = (
    ("link0", "link1"),
    ("link1", "link2"),
    ("link1", "link3"),
    ("link1", "link4"),
    ("link2", "link3"),
    ("link2", "link4"),
    ("link3", "link4"),
)
MESH_DIR = Path(get_package_share_directory("myplan")) / "mesh"
MESH_SCALE = (0.00095, 0.00095, 0.00095)
MESH_FILE_PATH = {
    tower_name: str(MESH_DIR / f"{tower_name}.stl")
    for tower_name in HANOI_TOWER_NAMES
}
MESH_ORIENTATION = Quaternion(x=0.7071081, y=0.0, z=0.0, w=0.7071081)
ATTACHED_MESH_ORIENTATION = Quaternion(x=0.0, y=0.0, z=0.7071081, w=0.7071081)
for mesh in MESH_FILE_PATH.values():
    if not Path(mesh).exists():
        raise FileNotFoundError(f"Mesh path error: {mesh}")

SceneAction = Literal["attach", "detach"]
HanoiMove = tuple[int, int]
StationStacks = list[list[str]]


@dataclass(frozen=True)
class HanoiWaypoint:
    x: float
    y: float
    z: float
    magnet_on: bool
    tower_name: str | None = None
    scene_action: SceneAction | None = None


@dataclass(frozen=True)
class HanoiTaskPlan:
    waypoints: list[HanoiWaypoint]
    collect_move_count: int
    final_move_count: int
    largest_station: int


"""
Hint:
    The output of your "Hanoi-Tower-Function" can be a series of [x, y, z, eef-state], where
    1.xyz in world frame
    2.eef-state: 1 for magnet on, 0 for off
"""


def load_mesh_from_file(
    file_path: str,
    scale: tuple[float, float, float],
) -> Mesh:
    mesh_data = trimesh.load(file_path, force="mesh")
    assert isinstance(mesh_data, trimesh.base.Trimesh)

    vertices = [
        Point(
            x=float(vertex[0]) * scale[0],
            y=float(vertex[1]) * scale[1],
            z=float(vertex[2]) * scale[2],
        )
        for vertex in mesh_data.vertices
    ]

    triangles = [
        MeshTriangle(vertex_indices=[int(face[0]), int(face[1]), int(face[2])])
        for face in mesh_data.faces
        if len(face) == 3
    ]
    return Mesh(triangles=triangles, vertices=vertices)


class MoveGroupPythonInterface(Node):
    def __init__(self, executor: MultiThreadedExecutor):
        super().__init__("move_group_python_interface")

        self.joint_angles: list[float] | None = None
        self.hanoi_busy = False

        self._executor = executor
        self.callback_group = ReentrantCallbackGroup()

        self.GROUP_NAME = "ldsc_arm"
        self.PLANNING_FRAME = "world"

        self.action_client = ActionClient(self, MoveGroup, "move_action")

        self.display_trajectory_publisher = self.create_publisher(
            DisplayTrajectory, "/move_group/display_planned_path", 20
        )

        self.pub_eef_state = self.create_publisher(Bool, "/SetEndEffector", 10)

        self.collision_object_publisher = self.create_publisher(
            CollisionObject, "/collision_object", 10
        )

        self.attached_collision_object_publisher = self.create_publisher(
            AttachedCollisionObject, "/attached_collision_object", 10
        )

        self.planning_scene_publisher = self.create_publisher(
            PlanningScene, "/planning_scene", 10
        )
        self.apply_planning_scene_client = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )
        self.get_planning_scene_client = self.create_client(
            GetPlanningScene, "/get_planning_scene"
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
            callback_group=self.callback_group,
        )

        self.hanoi_station_service = self.create_service(
            SetHanoiTowerStations,
            "/set_hanoi_tower_stations",
            self.handle_hanoi_station_request,
            callback_group=self.callback_group,
        )

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

        self.get_logger().info("Waiting for trajectory action server...")
        if self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().info("Trajectory action server connected!")
        else:
            self.get_logger().error("Trajectory action server not available!")

        self.get_logger().info("MoveGroup Python Interface already initialized")
        self.get_logger().info("Waiting for /set_hanoi_tower_stations requests")
        self.allow_hanoi_contacts()

    def joint_state_callback(self, msg: JointState):
        try:
            joint_pair: dict[str, float] = dict(zip(msg.name, msg.position))
            self.joint_angles = [joint_pair[name] for name in Joint_NAMES]
        except Exception as e:
            self.get_logger().error(f"Error in joint_state_callback: {str(e)}")

    def handle_hanoi_station_request(
        self,
        request: SetHanoiTowerStations.Request,
        response: SetHanoiTowerStations.Response,
    ) -> SetHanoiTowerStations.Response:
        if self.hanoi_busy:
            response.success = False
            response.message = "Hanoi planner is already executing a request"
            return response

        tower_stations = tuple(int(station) for station in request.tower_stations)
        target_station = int(request.target_station)

        self.hanoi_busy = True
        try:
            plan = build_hanoi_task_plan(tower_stations, target_station)
            self.get_logger().info(
                "Accepted Hanoi request: "
                f"tower_stations={tower_stations}, target_station={target_station}, "
                f"largest_station={plan.largest_station}, "
                f"collect_moves={plan.collect_move_count}, "
                f"final_moves={plan.final_move_count}, "
                f"waypoints={len(plan.waypoints)}"
            )

            execute_waypoints(self, plan.waypoints)
            self.switch_magnet(False)
            self.go_to_joint_state(HOME_POSITION)

            response.success = True
            response.message = (
                "Hanoi task completed: "
                f"collected towers at station {plan.largest_station}, "
                f"then moved tower to station {target_station}"
            )
        except (RuntimeError, ValueError) as e:
            self.get_logger().error(f"Hanoi request failed: {str(e)}")
            self.switch_magnet(False)
            self.go_to_joint_state(HOME_POSITION)
            response.success = False
            response.message = str(e)
        finally:
            self.hanoi_busy = False

        return response

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
        self.allow_hanoi_contacts(log=False)

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
        else:
            self.get_logger().error(f"Motion failed with error code: {result.error_code.val}")
            return False

    def ensure_acm_name(self, acm: AllowedCollisionMatrix, name: str) -> None:
        if name in acm.entry_names:
            return

        for entry in acm.entry_values:
            entry.enabled.append(False)
        acm.entry_names.append(name)
        acm.entry_values.append(
            AllowedCollisionEntry(enabled=[False] * len(acm.entry_names))
        )

    def set_allowed_collision(
        self,
        acm: AllowedCollisionMatrix,
        first_name: str,
        second_name: str,
        allowed: bool = True,
    ) -> None:
        self.ensure_acm_name(acm, first_name)
        self.ensure_acm_name(acm, second_name)

        size = len(acm.entry_names)
        for entry in acm.entry_values:
            if len(entry.enabled) < size:
                entry.enabled.extend([False] * (size - len(entry.enabled)))

        first_index = acm.entry_names.index(first_name)
        second_index = acm.entry_names.index(second_name)
        acm.entry_values[first_index].enabled[second_index] = allowed
        acm.entry_values[second_index].enabled[first_index] = allowed

    def get_current_allowed_collision_matrix(self) -> AllowedCollisionMatrix | None:
        if not self.get_planning_scene_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("get_planning_scene service is not available")
            return None

        request = GetPlanningScene.Request(
            components=PlanningSceneComponents(
                components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX,
            )
        )
        future = self.get_planning_scene_client.call_async(request)
        if not self.wait_for_future(future, timeout_sec=2.0) or future.result() is None:
            self.get_logger().warn("Could not read current planning scene")
            return None

        return future.result().scene.allowed_collision_matrix

    def allow_hanoi_contacts(self, log: bool = True) -> None:
        acm = self.get_current_allowed_collision_matrix()
        if acm is None:
            acm = AllowedCollisionMatrix()

        for tower_name in HANOI_TOWER_NAMES:
            for robot_link in ROBOT_LINKS:
                self.set_allowed_collision(acm, tower_name, robot_link)
            for other_tower_name in HANOI_TOWER_NAMES:
                self.set_allowed_collision(acm, tower_name, other_tower_name)

        for first_link, second_link in SRDF_ALLOWED_LINK_PAIRS:
            self.set_allowed_collision(acm, first_link, second_link)

        planning_scene = PlanningScene(
            is_diff=True,
            allowed_collision_matrix=acm,
        )

        if self.apply_planning_scene_client.wait_for_service(timeout_sec=1.0):
            future = self.apply_planning_scene_client.call_async(
                ApplyPlanningScene.Request(scene=planning_scene)
            )
            if not self.wait_for_future(future, timeout_sec=2.0):
                self.get_logger().warn("Timed out while applying planning-scene update")
            elif future.result() is not None and not future.result().success:
                self.get_logger().warn("MoveIt rejected the planning-scene update")
        else:
            for _ in range(5):
                self.planning_scene_publisher.publish(planning_scene)
                time.sleep(0.1)

        if log:
            self.get_logger().info(
                "Disabled collision checks between Hanoi towers, and between towers and arm links"
            )

    def switch_magnet(self, on: bool) -> None:
        """
        Description:
            Because moveit only plans the path,
            you have to publish end-effector state for playing hanoi.
        """
        self.pub_eef_state.publish(Bool(data=on))
        self.get_logger().info(f"Published end effector state: {on}")

    def wait_for_state_update(self) -> None:
        time.sleep(0.2)

    def remove_world_object(self, object_name: str) -> None:
        collision_object = CollisionObject(
            header=Header(
                frame_id=self.PLANNING_FRAME,
                stamp=self.get_clock().now().to_msg(),
            ),
            id=object_name,
            operation=CollisionObject.REMOVE,
        )

        self.collision_object_publisher.publish(collision_object)
        self.get_logger().info(f"Removed world object: {object_name}")
        self.wait_for_state_update()
        self.allow_hanoi_contacts(log=False)

    def add_world_mesh(
        self,
        *,
        object_name: str,
        position: Point,
    ) -> None:
        collision_object = CollisionObject(
            header=Header(
                frame_id=self.PLANNING_FRAME,
                stamp=self.get_clock().now().to_msg(),
            ),
            id=object_name,
            meshes=[load_mesh_from_file(MESH_FILE_PATH[object_name], MESH_SCALE)],
            mesh_poses=[Pose(position=position, orientation=MESH_ORIENTATION)],
            operation=CollisionObject.ADD,
        )

        self.collision_object_publisher.publish(collision_object)
        self.get_logger().info(
            f"Added world object: {object_name} at "
            f"({position.x:.3f}, {position.y:.3f}, {position.z:.3f})"
        )
        self.wait_for_state_update()
        self.allow_hanoi_contacts(log=False)

    def attach_object(self, *, object_name: str, link_name: str = TOOL_LINK) -> None:
        self.remove_world_object(object_name)

        attached_object = AttachedCollisionObject(
            link_name=link_name,
            object=CollisionObject(
                header=Header(
                    frame_id=link_name,
                    stamp=self.get_clock().now().to_msg(),
                ),
                id=object_name,
                meshes=[load_mesh_from_file(MESH_FILE_PATH[object_name], MESH_SCALE)],
                mesh_poses=[
                    Pose(
                        position=Point(
                            x=Tower_mesh_height + End_effector_contact_offset,
                            y=0.0,
                            z=0.0,
                        ),
                        orientation=ATTACHED_MESH_ORIENTATION,
                    )
                ],
                operation=CollisionObject.ADD,
            ),
            touch_links=list(ROBOT_LINKS),
        )

        self.attached_collision_object_publisher.publish(attached_object)
        self.get_logger().info(f"Attached object: {object_name} to {link_name}")
        self.wait_for_state_update()
        self.allow_hanoi_contacts(log=False)

    def detach_object(
        self,
        *,
        object_name: str,
        world_position: Point,
        link_name: str = TOOL_LINK,
    ) -> None:
        attached_object = AttachedCollisionObject(
            link_name=link_name,
            object=CollisionObject(id=object_name, operation=CollisionObject.REMOVE),
        )

        self.attached_collision_object_publisher.publish(attached_object)
        self.get_logger().info(f"Detached object: {object_name} from {link_name}")
        self.wait_for_state_update()
        self.add_world_mesh(object_name=object_name, position=world_position)
        self.allow_hanoi_contacts(log=False)


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
            score = sum(
                abs(angle - home)
                for angle, home in zip(joint_angles, HOME_POSITION)
            )
            candidates.append((score, joint_angles))

    if not candidates:
        raise ValueError("target is reachable geometrically, but violates joint limits")

    _, solution = min(candidates, key=lambda item: item[0])
    return tuple(_clamp_to_limit(angle, index) for index, angle in enumerate(solution))


def generate_hanoi_moves(
    num_disks: int,
    source: int,
    target: int,
    auxiliary: int,
) -> list[HanoiMove]:
    if num_disks <= 0:
        return []

    return (
        generate_hanoi_moves(num_disks - 1, source, auxiliary, target)
        + [(source, target)]
        + generate_hanoi_moves(num_disks - 1, auxiliary, target, source)
    )


def get_auxiliary_station(
    source: int,
    target: int,
    station_count: int = len(STATION_POSITIONS),
) -> int:
    available_stations = set(range(station_count))
    requested_stations = {source, target}

    if source == target:
        raise ValueError("source and target stations must be different")
    if not requested_stations.issubset(available_stations):
        raise ValueError(
            f"source and target stations must be between 0 and {station_count - 1}"
        )

    auxiliary_stations = available_stations - requested_stations
    if len(auxiliary_stations) != 1:
        raise ValueError(
            "Hanoi planner needs exactly one auxiliary station after choosing "
            "source and target"
        )

    return auxiliary_stations.pop()


def validate_hanoi_request(
    tower_stations: tuple[int, ...],
    target_station: int,
    station_count: int = len(STATION_POSITIONS),
) -> None:
    if len(tower_stations) != NUM_DISKS:
        raise ValueError(f"expected {NUM_DISKS} tower station values")

    valid_stations = set(range(station_count))
    invalid_tower_stations = [
        station for station in tower_stations if station not in valid_stations
    ]
    if invalid_tower_stations:
        raise ValueError(
            f"tower stations must be between 0 and {station_count - 1}: "
            f"{invalid_tower_stations}"
        )

    if target_station not in valid_stations:
        raise ValueError(
            f"target station must be between 0 and {station_count - 1}"
        )


def build_stacks_from_tower_stations(
    tower_stations: tuple[int, ...],
    station_count: int = len(STATION_POSITIONS),
) -> StationStacks:
    stacks: StationStacks = [[] for _ in range(station_count)]
    for tower_name, station in zip(HANOI_TOWER_NAMES, tower_stations):
        stacks[station].append(tower_name)
    return stacks


def tower_size_rank(tower_name: str) -> int:
    return int(tower_name.removeprefix("tower"))


def validate_legal_move(
    tower_name: str,
    target_stack: list[str],
) -> None:
    if not target_stack:
        return

    moving_rank = tower_size_rank(tower_name)
    target_top_rank = tower_size_rank(target_stack[-1])
    if moving_rank < target_top_rank:
        raise ValueError(
            f"illegal Hanoi move: cannot place {tower_name} on {target_stack[-1]}"
        )


def generate_moves_to_station(
    tower_stations: tuple[int, ...],
    target_station: int,
) -> list[HanoiMove]:
    state = list(tower_stations)
    moves: list[HanoiMove] = []

    def move_disk_and_smaller(disk_index: int, destination: int) -> None:
        if disk_index >= NUM_DISKS:
            return

        current_station = state[disk_index]
        if current_station == destination:
            move_disk_and_smaller(disk_index + 1, destination)
            return

        auxiliary = get_auxiliary_station(current_station, destination)
        move_disk_and_smaller(disk_index + 1, auxiliary)
        moves.append((current_station, destination))
        state[disk_index] = destination
        move_disk_and_smaller(disk_index + 1, destination)

    move_disk_and_smaller(0, target_station)
    return moves


def tower_top_z(stack_size: int) -> float:
    if stack_size <= 0:
        return Tower_base

    exposed_height = Tower_height - Tower_overlap
    return (
        Tower_base
        + (stack_size - 1) * exposed_height
        + Tower_mesh_height
        + End_effector_contact_offset
    )


def build_hanoi_waypoints(
    num_disks: int = NUM_DISKS,
    source: int = SOURCE_STATION,
    target: int = TARGET_STATION,
) -> list[HanoiWaypoint]:
    auxiliary = get_auxiliary_station(source, target)
    moves = generate_hanoi_moves(num_disks, source, target, auxiliary)
    stacks: StationStacks = [[] for _ in STATION_POSITIONS]
    stacks[source] = [f"tower{index}" for index in range(1, num_disks + 1)]
    return build_waypoints_from_moves(moves, stacks)


def build_waypoints_from_moves(
    moves: list[HanoiMove],
    initial_stacks: StationStacks,
) -> list[HanoiWaypoint]:
    stacks = [stack.copy() for stack in initial_stacks]
    waypoints: list[HanoiWaypoint] = []

    for source_index, target_index in moves:
        source_x, source_y = STATION_POSITIONS[source_index]
        target_x, target_y = STATION_POSITIONS[target_index]

        if not stacks[source_index]:
            raise ValueError(f"station {source_index} has no tower to move")

        pick_z = tower_top_z(len(stacks[source_index]))
        tower_name = stacks[source_index].pop()
        validate_legal_move(tower_name, stacks[target_index])
        place_z = tower_top_z(len(stacks[target_index]) + 1)

        source_approach_z = pick_z + APPROACH_HEIGHT
        target_approach_z = place_z + APPROACH_HEIGHT

        waypoints.extend(
            [
                HanoiWaypoint(source_x, source_y, source_approach_z, False),
                HanoiWaypoint(source_x, source_y, pick_z, True, tower_name, "attach"),
                HanoiWaypoint(source_x, source_y, source_approach_z, True),
                HanoiWaypoint(target_x, target_y, target_approach_z, True),
                HanoiWaypoint(target_x, target_y, place_z, False, tower_name, "detach"),
                HanoiWaypoint(target_x, target_y, target_approach_z, False),
            ]
        )

        stacks[target_index].append(tower_name)

    return waypoints


def build_hanoi_task_plan(
    tower_stations: tuple[int, ...],
    target_station: int,
) -> HanoiTaskPlan:
    validate_hanoi_request(tower_stations, target_station)

    largest_station = tower_stations[0]
    initial_stacks = build_stacks_from_tower_stations(tower_stations)
    collect_moves = generate_moves_to_station(tower_stations, largest_station)
    collect_waypoints = build_waypoints_from_moves(collect_moves, initial_stacks)

    final_moves: list[HanoiMove] = []
    final_waypoints: list[HanoiWaypoint] = []
    if largest_station != target_station:
        auxiliary = get_auxiliary_station(largest_station, target_station)
        final_moves = generate_hanoi_moves(
            NUM_DISKS,
            largest_station,
            target_station,
            auxiliary,
        )
        collected_stacks = build_stacks_from_tower_stations(
            tuple(largest_station for _ in HANOI_TOWER_NAMES)
        )
        final_waypoints = build_waypoints_from_moves(final_moves, collected_stacks)

    return HanoiTaskPlan(
        waypoints=collect_waypoints + final_waypoints,
        collect_move_count=len(collect_moves),
        final_move_count=len(final_moves),
        largest_station=largest_station,
    )


def update_scene_for_waypoint(
    path_object: MoveGroupPythonInterface,
    waypoint: HanoiWaypoint,
    current_eef_state: bool,
) -> bool:
    if waypoint.scene_action == "attach" and waypoint.tower_name is not None:
        if not current_eef_state:
            path_object.switch_magnet(True)
            current_eef_state = True
        path_object.attach_object(object_name=waypoint.tower_name)
        return current_eef_state

    if waypoint.scene_action == "detach" and waypoint.tower_name is not None:
        path_object.detach_object(
            object_name=waypoint.tower_name,
            world_position=Point(
                x=float(waypoint.x),
                y=float(waypoint.y),
                z=float(waypoint.z - Tower_mesh_height - End_effector_contact_offset),
            ),
        )
        if current_eef_state:
            path_object.switch_magnet(False)
            return False
        return current_eef_state

    if waypoint.magnet_on != current_eef_state:
        path_object.switch_magnet(waypoint.magnet_on)
        return waypoint.magnet_on

    return current_eef_state


def execute_waypoints(
    path_object: MoveGroupPythonInterface,
    waypoints: list[HanoiWaypoint],
) -> None:
    current_eef_state = False
    path_object.switch_magnet(current_eef_state)

    for index, waypoint in enumerate(waypoints, start=1):
        path_object.get_logger().info(
            f"Waypoint {index}/{len(waypoints)}: "
            f"x={waypoint.x:.3f}, y={waypoint.y:.3f}, z={waypoint.z:.3f}, "
            f"magnet={waypoint.magnet_on}, object={waypoint.tower_name}, "
            f"scene_action={waypoint.scene_action}"
        )
        joint_angles = Your_IK(waypoint.x, waypoint.y, waypoint.z)
        if not path_object.go_to_joint_state(joint_angles):
            raise RuntimeError(f"Motion planning failed at waypoint {index}")

        current_eef_state = update_scene_for_waypoint(
            path_object,
            waypoint,
            current_eef_state,
        )
        time.sleep(MOTION_DELAY)


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
