from dataclasses import dataclass
from math import acos, atan2, cos, hypot, pi, sin
from typing import Literal

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4")
Joint_NAMES = JOINT_NAMES
LINK_LENGTH = (0.0600, 0.0820, 0.1320, 0.1664, 0.0480, 0.0040)
JOINT_LIMITS = (
    (-1.2, 1.2),
    (-2.0, 2.0),
    (-1.67, 1.67),
    (-pi / 2, pi / 2),
)
HOME_POSITION = (0.0, -pi / 2, pi / 2, 0.0)

# You can measure these in Lab402.
Tower_base = 0.0014
Tower_height = 0.025
Tower_overlap = 0.015
Tower_mesh_height = 0.02375
End_effector_contact_offset = 0.01

# You may want to slightly change this.
STATION_POSITIONS = (
    (0.25, 0.15),
    (0.25, 0.0),
    (0.25, -0.15),
)
NUM_DISKS = 3
SOURCE_STATION = 1
TARGET_STATION = 0
APPROACH_HEIGHT = 0.1
MOTION_DELAY = 0.1
START_WAYPOINT_POSITION = (0.25, 0.0, 0.1)
END_WAYPOINT_POSITION = (0.25, 0.1, 0.25)
HANOI_TOWER_NAMES = tuple(f"tower{index}" for index in range(1, NUM_DISKS + 1))

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
    stop_at_waypoint: bool = False


@dataclass(frozen=True)
class HanoiTaskPlan:
    waypoints: list[HanoiWaypoint]
    collect_move_count: int
    final_move_count: int
    largest_station: int


class ArmKinematics:
    def __init__(
        self,
        *,
        link_lengths: tuple[float, float, float, float, float, float] = LINK_LENGTH,
        joint_limits: tuple[tuple[float, float], ...] = JOINT_LIMITS,
        home_position: tuple[float, float, float, float] = HOME_POSITION,
    ) -> None:
        self.link_lengths = link_lengths
        self.joint_limits = joint_limits
        self.home_position = home_position

    def solve(
        self,
        x: float,
        y: float,
        z: float,
        pitch: float = pi / 2,
    ) -> tuple[float, float, float, float]:
        """Analytic IK for the 4-DOF arm described by myrobot.urdf."""
        base_height = self.link_lengths[0] + self.link_lengths[1]
        upper_arm = self.link_lengths[2]
        forearm = self.link_lengths[3]
        tool_z = self.link_lengths[4]
        tool_x = self.link_lengths[5]

        joint1 = atan2(y, x) if hypot(x, y) > 1e-12 else 0.0
        if not self._inside_limit(joint1, 0):
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

            if all(
                self._inside_limit(angle, index)
                for index, angle in enumerate(joint_angles)
            ):
                score = sum(
                    abs(angle - home)
                    for angle, home in zip(joint_angles, self.home_position)
                )
                candidates.append((score, joint_angles))

        if not candidates:
            raise ValueError("target is reachable geometrically, but violates joint limits")

        _, solution = min(candidates, key=lambda item: item[0])
        return tuple(
            self._clamp_to_limit(angle, index)
            for index, angle in enumerate(solution)
        )

    def _inside_limit(
        self,
        angle: float,
        joint_index: int,
        tolerance: float = 1e-9,
    ) -> bool:
        lower, upper = self.joint_limits[joint_index]
        return lower - tolerance <= angle <= upper + tolerance

    def _clamp_to_limit(self, angle: float, joint_index: int) -> float:
        lower, upper = self.joint_limits[joint_index]
        return max(lower, min(upper, angle))


class HanoiTowerWaypointPlanner:
    def __init__(
        self,
        *,
        num_disks: int = NUM_DISKS,
        station_positions: tuple[tuple[float, float], ...] = STATION_POSITIONS,
        approach_height: float = APPROACH_HEIGHT,
        tower_names: tuple[str, ...] | None = None,
        start_waypoint_position: tuple[float, float, float] | None = (
            START_WAYPOINT_POSITION
        ),
        end_waypoint_position: tuple[float, float, float] | None = (
            END_WAYPOINT_POSITION
        ),
    ) -> None:
        self.num_disks = num_disks
        self.station_positions = station_positions
        self.approach_height = approach_height
        self.tower_names = tower_names or tuple(
            f"tower{index}" for index in range(1, num_disks + 1)
        )
        self.start_waypoint_position = start_waypoint_position
        self.end_waypoint_position = end_waypoint_position

    def build_default_waypoints(
        self,
        source: int = SOURCE_STATION,
        target: int = TARGET_STATION,
    ) -> list[HanoiWaypoint]:
        auxiliary = self.get_auxiliary_station(source, target)
        moves = self.generate_hanoi_moves(
            self.num_disks,
            source,
            target,
            auxiliary,
        )
        stacks: StationStacks = [[] for _ in self.station_positions]
        stacks[source] = list(self.tower_names)
        return self._with_boundary_waypoints(
            self.build_waypoints_from_moves(moves, stacks)
        )

    def build_task_plan(
        self,
        tower_stations: tuple[int, ...],
        target_station: int,
    ) -> HanoiTaskPlan:
        self.validate_request(tower_stations, target_station)

        largest_station = tower_stations[0]
        initial_stacks = self.build_stacks_from_tower_stations(tower_stations)
        collect_moves = self.generate_moves_to_station(tower_stations, largest_station)
        collect_waypoints = self.build_waypoints_from_moves(
            collect_moves,
            initial_stacks,
        )

        final_moves: list[HanoiMove] = []
        final_waypoints: list[HanoiWaypoint] = []
        if largest_station != target_station:
            auxiliary = self.get_auxiliary_station(largest_station, target_station)
            final_moves = self.generate_hanoi_moves(
                self.num_disks,
                largest_station,
                target_station,
                auxiliary,
            )
            collected_stacks = self.build_stacks_from_tower_stations(
                tuple(largest_station for _ in self.tower_names)
            )
            final_waypoints = self.build_waypoints_from_moves(
                final_moves,
                collected_stacks,
            )

        return HanoiTaskPlan(
            waypoints=self._with_boundary_waypoints(
                collect_waypoints + final_waypoints
            ),
            collect_move_count=len(collect_moves),
            final_move_count=len(final_moves),
            largest_station=largest_station,
        )

    def _with_boundary_waypoints(
        self,
        waypoints: list[HanoiWaypoint],
    ) -> list[HanoiWaypoint]:
        boundary_waypoints = list(waypoints)

        if self.start_waypoint_position is not None:
            x, y, z = self.start_waypoint_position
            boundary_waypoints.insert(
                0,
                HanoiWaypoint(
                    x=x,
                    y=y,
                    z=z,
                    magnet_on=False,
                    stop_at_waypoint=True,
                ),
            )

        if self.end_waypoint_position is not None:
            x, y, z = self.end_waypoint_position
            boundary_waypoints.append(
                HanoiWaypoint(
                    x=x,
                    y=y,
                    z=z,
                    magnet_on=False,
                    stop_at_waypoint=True,
                )
            )

        return boundary_waypoints

    def generate_hanoi_moves(
        self,
        num_disks: int,
        source: int,
        target: int,
        auxiliary: int,
    ) -> list[HanoiMove]:
        if num_disks <= 0:
            return []

        return (
            self.generate_hanoi_moves(num_disks - 1, source, auxiliary, target)
            + [(source, target)]
            + self.generate_hanoi_moves(num_disks - 1, auxiliary, target, source)
        )

    def generate_moves_to_station(
        self,
        tower_stations: tuple[int, ...],
        target_station: int,
    ) -> list[HanoiMove]:
        state = list(tower_stations)
        moves: list[HanoiMove] = []

        def move_disk_and_smaller(disk_index: int, destination: int) -> None:
            if disk_index >= self.num_disks:
                return

            current_station = state[disk_index]
            if current_station == destination:
                move_disk_and_smaller(disk_index + 1, destination)
                return

            auxiliary = self.get_auxiliary_station(current_station, destination)
            move_disk_and_smaller(disk_index + 1, auxiliary)
            moves.append((current_station, destination))
            state[disk_index] = destination
            move_disk_and_smaller(disk_index + 1, destination)

        move_disk_and_smaller(0, target_station)
        return moves

    def build_waypoints_from_moves(
        self,
        moves: list[HanoiMove],
        initial_stacks: StationStacks,
    ) -> list[HanoiWaypoint]:
        stacks = [stack.copy() for stack in initial_stacks]
        waypoints: list[HanoiWaypoint] = []

        for source_index, target_index in moves:
            source_x, source_y = self.station_positions[source_index]
            target_x, target_y = self.station_positions[target_index]

            if not stacks[source_index]:
                raise ValueError(f"station {source_index} has no tower to move")

            pick_z = self.tower_top_z(len(stacks[source_index]))
            tower_name = stacks[source_index].pop()
            self.validate_legal_move(tower_name, stacks[target_index])
            place_z = self.tower_top_z(len(stacks[target_index]) + 1)

            source_approach_z = pick_z + self.approach_height
            target_approach_z = place_z + self.approach_height

            waypoints.extend(
                [
                    HanoiWaypoint(source_x, source_y, source_approach_z, False),
                    HanoiWaypoint(
                        source_x,
                        source_y,
                        pick_z,
                        True,
                        tower_name,
                        "attach",
                    ),
                    HanoiWaypoint(source_x, source_y, source_approach_z, True),
                    HanoiWaypoint(target_x, target_y, target_approach_z, True),
                    HanoiWaypoint(
                        target_x,
                        target_y,
                        place_z,
                        False,
                        tower_name,
                        "detach",
                    ),
                    HanoiWaypoint(target_x, target_y, target_approach_z, False),
                ]
            )

            stacks[target_index].append(tower_name)

        return waypoints

    def validate_request(
        self,
        tower_stations: tuple[int, ...],
        target_station: int,
    ) -> None:
        if len(tower_stations) != self.num_disks:
            raise ValueError(f"expected {self.num_disks} tower station values")

        valid_stations = set(range(len(self.station_positions)))
        invalid_tower_stations = [
            station for station in tower_stations if station not in valid_stations
        ]
        if invalid_tower_stations:
            raise ValueError(
                f"tower stations must be between 0 and {len(self.station_positions) - 1}: "
                f"{invalid_tower_stations}"
            )

        if target_station not in valid_stations:
            raise ValueError(
                f"target station must be between 0 and {len(self.station_positions) - 1}"
            )

    def get_auxiliary_station(self, source: int, target: int) -> int:
        available_stations = set(range(len(self.station_positions)))
        requested_stations = {source, target}

        if source == target:
            raise ValueError("source and target stations must be different")
        if not requested_stations.issubset(available_stations):
            raise ValueError(
                "source and target stations must be between 0 and "
                f"{len(self.station_positions) - 1}"
            )

        auxiliary_stations = available_stations - requested_stations
        if len(auxiliary_stations) != 1:
            raise ValueError(
                "Hanoi planner needs exactly one auxiliary station after choosing "
                "source and target"
            )

        return auxiliary_stations.pop()

    def build_stacks_from_tower_stations(
        self,
        tower_stations: tuple[int, ...],
    ) -> StationStacks:
        stacks: StationStacks = [[] for _ in self.station_positions]
        for tower_name, station in zip(self.tower_names, tower_stations):
            stacks[station].append(tower_name)
        return stacks

    def validate_legal_move(
        self,
        tower_name: str,
        target_stack: list[str],
    ) -> None:
        if not target_stack:
            return

        moving_rank = self.tower_size_rank(tower_name)
        target_top_rank = self.tower_size_rank(target_stack[-1])
        if moving_rank < target_top_rank:
            raise ValueError(
                f"illegal Hanoi move: cannot place {tower_name} on {target_stack[-1]}"
            )

    @staticmethod
    def tower_size_rank(tower_name: str) -> int:
        return int(tower_name.removeprefix("tower"))

    @staticmethod
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


_DEFAULT_PLANNER = HanoiTowerWaypointPlanner()
_DEFAULT_KINEMATICS = ArmKinematics()


def Your_IK(
    x: float,
    y: float,
    z: float,
    pitch: float = pi / 2,
) -> tuple[float, float, float, float]:
    return _DEFAULT_KINEMATICS.solve(x, y, z, pitch)


def generate_hanoi_moves(
    num_disks: int,
    source: int,
    target: int,
    auxiliary: int,
) -> list[HanoiMove]:
    return _DEFAULT_PLANNER.generate_hanoi_moves(num_disks, source, target, auxiliary)


def get_auxiliary_station(
    source: int,
    target: int,
    station_count: int = len(STATION_POSITIONS),
) -> int:
    planner = HanoiTowerWaypointPlanner(
        station_positions=STATION_POSITIONS[:station_count],
    )
    return planner.get_auxiliary_station(source, target)


def validate_hanoi_request(
    tower_stations: tuple[int, ...],
    target_station: int,
    station_count: int = len(STATION_POSITIONS),
) -> None:
    planner = HanoiTowerWaypointPlanner(
        station_positions=STATION_POSITIONS[:station_count],
    )
    planner.validate_request(tower_stations, target_station)


def build_stacks_from_tower_stations(
    tower_stations: tuple[int, ...],
    station_count: int = len(STATION_POSITIONS),
) -> StationStacks:
    planner = HanoiTowerWaypointPlanner(
        station_positions=STATION_POSITIONS[:station_count],
    )
    return planner.build_stacks_from_tower_stations(tower_stations)


def tower_size_rank(tower_name: str) -> int:
    return HanoiTowerWaypointPlanner.tower_size_rank(tower_name)


def validate_legal_move(tower_name: str, target_stack: list[str]) -> None:
    _DEFAULT_PLANNER.validate_legal_move(tower_name, target_stack)


def generate_moves_to_station(
    tower_stations: tuple[int, ...],
    target_station: int,
) -> list[HanoiMove]:
    return _DEFAULT_PLANNER.generate_moves_to_station(tower_stations, target_station)


def tower_top_z(stack_size: int) -> float:
    return HanoiTowerWaypointPlanner.tower_top_z(stack_size)


def build_hanoi_waypoints(
    num_disks: int = NUM_DISKS,
    source: int = SOURCE_STATION,
    target: int = TARGET_STATION,
) -> list[HanoiWaypoint]:
    planner = HanoiTowerWaypointPlanner(num_disks=num_disks)
    return planner.build_default_waypoints(source, target)


def build_waypoints_from_moves(
    moves: list[HanoiMove],
    initial_stacks: StationStacks,
) -> list[HanoiWaypoint]:
    return _DEFAULT_PLANNER.build_waypoints_from_moves(moves, initial_stacks)


def build_hanoi_task_plan(
    tower_stations: tuple[int, ...],
    target_station: int,
) -> HanoiTaskPlan:
    return _DEFAULT_PLANNER.build_task_plan(tower_stations, target_station)
