import time
from typing import Any, Protocol

from geometry_msgs.msg import Point
from myrobot_interfaces.srv import SetHanoiTowerStations

from myrobot.hanoi_waypoint_planning import (
    ArmKinematics,
    End_effector_contact_offset,
    HOME_POSITION,
    HanoiTaskPlan,
    HanoiTowerWaypointPlanner,
    HanoiWaypoint,
    MOTION_DELAY,
    Tower_mesh_height,
)
from myrobot.moveit2_acm_management import MoveIt2AcmManager


class HanoiMotionInterface(Protocol):
    def go_to_joint_state(
        self,
        joint_angles: tuple[float, float, float, float],
    ) -> bool:
        ...

    def go_through_joint_states(
        self,
        joint_angle_sequence: list[tuple[float, float, float, float]],
    ) -> bool:
        ...

    def switch_magnet(self, on: bool) -> None:
        ...

    def get_logger(self) -> Any:
        ...


class HanoiProgressHandler:
    def __init__(
        self,
        *,
        node: Any,
        motion_interface: HanoiMotionInterface,
        scene_manager: MoveIt2AcmManager,
        planner: HanoiTowerWaypointPlanner | None = None,
        kinematics: ArmKinematics | None = None,
        callback_group: Any | None = None,
    ) -> None:
        self._node = node
        self._motion = motion_interface
        self._scene_manager = scene_manager
        self._planner = planner or HanoiTowerWaypointPlanner()
        self._kinematics = kinematics or ArmKinematics()
        self._busy = False

        self.service = node.create_service(
            SetHanoiTowerStations,
            "/set_hanoi_tower_stations",
            self.handle_hanoi_station_request,
            callback_group=callback_group,
        )

    @property
    def busy(self) -> bool:
        return self._busy

    def handle_hanoi_station_request(
        self,
        request: SetHanoiTowerStations.Request,
        response: SetHanoiTowerStations.Response,
    ) -> SetHanoiTowerStations.Response:
        if self._busy:
            response.success = False
            response.message = "Hanoi planner is already executing a request"
            return response

        tower_stations = tuple(int(station) for station in request.tower_stations)
        target_station = int(request.target_station)

        self._busy = True
        try:
            plan = self._planner.build_task_plan(tower_stations, target_station)
            self._log_plan_acceptance(plan, tower_stations, target_station)

            self.execute_waypoints(plan.waypoints)
            self._motion.switch_magnet(False)
            self._motion.go_to_joint_state(HOME_POSITION)

            response.success = True
            response.message = (
                "Hanoi task completed: "
                f"collected towers at station {plan.largest_station}, "
                f"then moved tower to station {target_station}"
            )
        except (RuntimeError, ValueError) as e:
            self._motion.get_logger().error(f"Hanoi request failed: {str(e)}")
            self._motion.switch_magnet(False)
            self._motion.go_to_joint_state(HOME_POSITION)
            response.success = False
            response.message = str(e)
        finally:
            self._busy = False

        return response

    def execute_waypoints(self, waypoints: list[HanoiWaypoint]) -> None:
        current_eef_state = False
        self._motion.switch_magnet(current_eef_state)
        motion_segment: list[tuple[int, HanoiWaypoint]] = []

        for index, waypoint in enumerate(waypoints, start=1):
            self._log_waypoint(index, len(waypoints), waypoint)
            motion_segment.append((index, waypoint))

            if not self._requires_stop(waypoint, current_eef_state):
                continue

            self._execute_motion_segment(motion_segment)

            current_eef_state = self.update_scene_for_waypoint(
                waypoint,
                current_eef_state,
            )
            motion_segment = []
            time.sleep(MOTION_DELAY)

        if motion_segment:
            self._execute_motion_segment(motion_segment)

    def _execute_motion_segment(
        self,
        motion_segment: list[tuple[int, HanoiWaypoint]],
    ) -> None:
        joint_angle_sequence = [
            self._kinematics.solve(waypoint.x, waypoint.y, waypoint.z)
            for _, waypoint in motion_segment
        ]
        if self._motion.go_through_joint_states(joint_angle_sequence):
            return

        first_index = motion_segment[0][0]
        last_index = motion_segment[-1][0]
        if first_index == last_index:
            raise RuntimeError(f"Motion planning failed at waypoint {last_index}")
        raise RuntimeError(
            f"Motion planning failed from waypoint {first_index} "
            f"to waypoint {last_index}"
        )

    @staticmethod
    def _requires_stop(
        waypoint: HanoiWaypoint,
        current_eef_state: bool,
    ) -> bool:
        return (
            waypoint.stop_at_waypoint
            or waypoint.scene_action is not None
            or waypoint.magnet_on != current_eef_state
        )

    def update_scene_for_waypoint(
        self,
        waypoint: HanoiWaypoint,
        current_eef_state: bool,
    ) -> bool:
        if waypoint.scene_action == "attach" and waypoint.tower_name is not None:
            if not current_eef_state:
                self._motion.switch_magnet(True)
                current_eef_state = True
            self._scene_manager.attach_object(object_name=waypoint.tower_name)
            return current_eef_state

        if waypoint.scene_action == "detach" and waypoint.tower_name is not None:
            self._scene_manager.detach_object(
                object_name=waypoint.tower_name,
                world_position=Point(
                    x=float(waypoint.x),
                    y=float(waypoint.y),
                    z=float(
                        waypoint.z
                        - Tower_mesh_height
                        - End_effector_contact_offset
                    ),
                ),
            )
            if current_eef_state:
                self._motion.switch_magnet(False)
                return False
            return current_eef_state

        if waypoint.magnet_on != current_eef_state:
            self._motion.switch_magnet(waypoint.magnet_on)
            return waypoint.magnet_on

        return current_eef_state

    def _log_plan_acceptance(
        self,
        plan: HanoiTaskPlan,
        tower_stations: tuple[int, ...],
        target_station: int,
    ) -> None:
        self._motion.get_logger().info(
            "Accepted Hanoi request: "
            f"tower_stations={tower_stations}, target_station={target_station}, "
            f"largest_station={plan.largest_station}, "
            f"collect_moves={plan.collect_move_count}, "
            f"final_moves={plan.final_move_count}, "
            f"waypoints={len(plan.waypoints)}"
        )

    def _log_waypoint(
        self,
        index: int,
        total: int,
        waypoint: HanoiWaypoint,
    ) -> None:
        self._motion.get_logger().info(
            f"Waypoint {index}/{total}: "
            f"x={waypoint.x:.3f}, y={waypoint.y:.3f}, z={waypoint.z:.3f}, "
            f"magnet={waypoint.magnet_on}, object={waypoint.tower_name}, "
            f"scene_action={waypoint.scene_action}"
        )
