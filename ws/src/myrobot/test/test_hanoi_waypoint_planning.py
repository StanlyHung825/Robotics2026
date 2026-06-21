from myrobot.hanoi_kinematics import ArmKinematics
from myrobot.hanoi_waypoint_planning import (
    ArmKinematics as CompatibleArmKinematics,
    HanoiTowerWaypointPlanner,
)


def test_public_kinematics_import_remains_compatible():
    assert CompatibleArmKinematics is ArmKinematics


def test_three_disk_plan_has_seven_moves():
    planner = HanoiTowerWaypointPlanner(
        start_waypoint_position=None,
        end_waypoint_position=None,
    )

    assert planner.generate_hanoi_moves(3, 0, 2, 1) == [
        (0, 2),
        (0, 1),
        (2, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (0, 2),
    ]
