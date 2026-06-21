from collections.abc import Sequence
from math import pi

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4")
Joint_NAMES = JOINT_NAMES
LINK_LENGTH = (0.0600, 0.0820, 0.1320, 0.1664, 0.0480, 0.0040)
JOINT_LIMITS = (
    (-1.2, 1.2),
    (-2.0, 2.0),
    (-1.75, 1.75),
    (-pi / 2, pi / 2),
)
HOME_POSITION = (0.0, -pi / 2, pi / 2, 0.0)
JOINT_OFFSETS = HOME_POSITION

Tower_base = 0.0014
Tower_height = 0.023
Tower_overlap = 0.005
Tower_mesh_height = 0.02
End_effector_contact_offset = 0.0

STATION_POSITIONS = (
    (0.25, 0.15),
    (0.25, 0.0),
    (0.25, -0.165),
)
NUM_DISKS = 3
SOURCE_STATION = 1
TARGET_STATION = 0
APPROACH_HEIGHT = 0.1
DIRECT_TRANSFER_CLEARANCE = 0.035
OBSTACLE_TRANSFER_CLEARANCE = 0.07
MOTION_DELAY = 0.01
START_WAYPOINT_POSITION = (0.25, 0.0, 0.20)
END_WAYPOINT_POSITION = (0.25, 0.1, 0.25)
HANOI_TOWER_NAMES = tuple(f"tower{index}" for index in range(1, NUM_DISKS + 1))
OBSTACLE_SIZE = (0.12, 0.001, 0.1)
OBSTACLE_POSITIONS = (
    (0.25, -0.075, 0.05),
    (0.25, 0.075, 0.05),
)


def status_station_to_planner_station(value: int) -> int | None:
    return value - 1 if 1 <= value <= len(STATION_POSITIONS) else None


def status_obstacles_to_planner(
    left_obstacle: int,
    right_obstacle: int,
) -> tuple[bool, bool]:
    return bool(left_obstacle), bool(right_obstacle)


def offset_angle(cmd: Sequence[float]) -> list[float]:
    if len(cmd) != len(JOINT_OFFSETS):
        raise ValueError(f"Command must have {len(JOINT_OFFSETS)} elements")
    return [cmdi - offset for cmdi, offset in zip(cmd, JOINT_OFFSETS)]
