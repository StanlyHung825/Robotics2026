import rclpy
from rclpy.node import Node

from hanoi_interface.srv import GetHanoiStatus
from myrobot_interfaces.srv import SetHanoiTowerStations


class HanoiStatusToSim(Node):
    def __init__(self) -> None:
        super().__init__("hanoi_status_to_sim")
        self.status_client = self.create_client(
            GetHanoiStatus, "get_hanoi_positions"
        )
        self.sim_client = self.create_client(
            SetHanoiTowerStations, "/set_hanoi_tower_stations"
        )
        self.timer = self.create_timer(1.0, self.try_send)
        self.pending = False
        self.last_sent: tuple[tuple[int, int, int], int] | None = None
        self.get_logger().info("HanoiStatusToSim started.")

    def try_send(self) -> None:
        if self.pending:
            return
        if not self.status_client.service_is_ready():
            self.get_logger().info("Waiting for get_hanoi_positions service...")
            return
        if not self.sim_client.service_is_ready():
            self.get_logger().info("Waiting for /set_hanoi_tower_stations service...")
            return

        request = GetHanoiStatus.Request()
        self.pending = True
        future = self.status_client.call_async(request)
        future.add_done_callback(self.on_status)

    def on_status(self, future) -> None:
        self.pending = False
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"Status request failed: {exc}")
            return

        tower_stations, target_station = self._convert_status(response)
        if tower_stations is None or target_station is None:
            self.get_logger().info("Status incomplete; skip sending.")
            return

        payload = (tower_stations, target_station)
        if self.last_sent == payload:
            return

        self.last_sent = payload
        self.send_hanoi_request(tower_stations, target_station)

    @staticmethod
    def _convert_station(value: int) -> int | None:
        if value == 1:
            return 0
        if value == 2:
            return 1
        if value == 3:
            return 2
        return None

    def _convert_status(self, response) -> tuple[tuple[int, int, int] | None, int | None]:
        # Assume tower1=large, tower2=medium, tower3=small
        large = self._convert_station(response.large_pos)
        medium = self._convert_station(response.medium_pos)
        small = self._convert_station(response.small_pos)
        target = self._convert_station(response.target_pos)

        if large is None or medium is None or small is None or target is None:
            return None, None

        return (large, medium, small), target

    def send_hanoi_request(
        self,
        tower_stations: tuple[int, int, int],
        target_station: int,
    ) -> None:
        request = SetHanoiTowerStations.Request()
        request.tower_stations = list(tower_stations)
        request.target_station = target_station

        future = self.sim_client.call_async(request)
        future.add_done_callback(self.on_sim_response)

    def on_sim_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"Sim request failed: {exc}")
            return

        if response.success:
            self.get_logger().info(response.message)
        else:
            self.get_logger().error(response.message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HanoiStatusToSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
