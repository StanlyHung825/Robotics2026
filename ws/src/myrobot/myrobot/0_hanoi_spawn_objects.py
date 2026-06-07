import rclpy
from myrobot_interfaces.srv import SetHanoiTowerStations
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from myrobot.hanoi_waypoint_planning import (
    HANOI_TOWER_NAMES,
    OBSTACLE_POSITIONS,
    STATION_POSITIONS,
)


BOX_POSITIONS = OBSTACLE_POSITIONS


def read_station(prompt: str) -> int:
    while True:
        try:
            station = int(input(prompt))
        except ValueError:
            print("Please enter an integer station index: 0, 1, or 2.")
            continue

        if 0 <= station < len(STATION_POSITIONS):
            return station

        print("Station index must be 0, 1, or 2.")


def read_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes", "1", "true"):
            return True
        if answer in ("n", "no", "0", "false"):
            return False

        print("Please enter y or n.")


def read_hanoi_request() -> tuple[tuple[int, ...], tuple[bool, ...], int]:
    print("Station positions:")
    for index, (x, y) in enumerate(STATION_POSITIONS):
        print(f"  station {index}: x={x:.3f}, y={y:.3f}")

    tower_stations = tuple(
        read_station(f"Which station is {tower_name} on? ")
        for tower_name in HANOI_TOWER_NAMES
    )

    print("Obstacle positions:")
    for index, (x, y, z) in enumerate(BOX_POSITIONS, start=1):
        print(f"  box_{index}: x={x:.3f}, y={y:.3f}, z={z:.3f}")
    obstacles = tuple(
        read_yes_no(f"Enable box_{index}? [y/n] ")
        for index in range(1, len(BOX_POSITIONS) + 1)
    )

    target_station = read_station("Which station should the whole tower move to? ")
    return tower_stations, obstacles, target_station


class HanoiRequestClient(Node):
    def __init__(self, executor: SingleThreadedExecutor):
        super().__init__("hanoi_spawn_objects")
        self._executor = executor
        self.hanoi_station_client = self.create_client(
            SetHanoiTowerStations,
            "/set_hanoi_tower_stations",
        )

    def send_hanoi_station_request(
        self,
        tower_stations: tuple[int, ...],
        obstacles: tuple[bool, ...],
        target_station: int,
    ) -> bool:
        self.get_logger().info("Waiting for /set_hanoi_tower_stations service...")
        if not self.hanoi_station_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                "/set_hanoi_tower_stations service is not available"
            )
            return False

        request = SetHanoiTowerStations.Request()
        request.tower_stations = list(tower_stations)
        request.obstacle = list(obstacles)
        request.target_station = target_station

        future = self.hanoi_station_client.call_async(request)
        while rclpy.ok() and not future.done():
            self._executor.spin_once(timeout_sec=0.1)

        if future.result() is None:
            self.get_logger().error("Hanoi station service call failed")
            return False

        response = future.result()
        if response.success:
            self.get_logger().info(response.message)
        else:
            self.get_logger().error(response.message)
        return response.success


def main(args=None):
    tower_stations, obstacles, target_station = read_hanoi_request()

    rclpy.init(args=args)

    request_client: HanoiRequestClient | None = None
    try:
        executor = SingleThreadedExecutor()
        request_client = HanoiRequestClient(executor)
        executor.add_node(request_client)
        request_client.send_hanoi_station_request(
            tower_stations,
            obstacles,
            target_station,
        )

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if request_client is not None:
            request_client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
