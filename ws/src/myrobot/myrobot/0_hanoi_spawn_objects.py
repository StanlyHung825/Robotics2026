import time
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import rclpy
import trimesh
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
)
from myrobot_interfaces.srv import SetHanoiTowerStations
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from std_msgs.msg import Header

"""Variable for end-effector"""
EefState = 0

"""Hanoi tower geometry"""
Tower_base = 0.0014  # Height of tower base
Tower_height = 0.025  # Height of each tower
Tower_overlap = 0.015  # Height of tower overlap

"""Hanoi tower position"""
STATION_POSITIONS = (
    (0.25, 0.15),
    (0.25, 0.0),
    (0.25, -0.15),
)
NUM_DISKS = 3
HANOI_TOWER_NAMES = tuple(f"tower{index}" for index in range(1, NUM_DISKS + 1))

"""Box geometry and position"""
BOX_SIZE = (0.1, 0.001, 0.1)
BOX_POSITIONS = (
    (0.25, -0.075, 0.05),
    (0.25, 0.075, 0.05),
)

"""Hanoi tower mesh file path"""
MESH_DIR = Path(get_package_share_directory("myplan")) / "mesh"
MESH_FILE_PATH = {
    tower_name: str(MESH_DIR / f"{tower_name}.stl")
    for tower_name in HANOI_TOWER_NAMES
}
for mesh in MESH_FILE_PATH.values():
    if not Path(mesh).exists():
        raise FileNotFoundError(f"Mesh path error: {mesh}")
MESH_SCALE = (0.00095, 0.00095, 0.00095)

"""Robot arm geometry"""
LINK_LENGTH = (0.0600, 0.0820, 0.1320, 0.1664, 0.0480, 0.0040)


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


def read_hanoi_request() -> tuple[tuple[int, ...], int]:
    print("Station positions:")
    for index, (x, y) in enumerate(STATION_POSITIONS):
        print(f"  station {index}: x={x:.3f}, y={y:.3f}")

    tower_stations = tuple(
        read_station(f"Which station is {tower_name} on? ")
        for tower_name in HANOI_TOWER_NAMES
    )
    target_station = read_station("Which station should the whole tower move to? ")
    return tower_stations, target_station


def build_stacks_from_tower_stations(tower_stations: tuple[int, ...]) -> list[list[str]]:
    stacks: list[list[str]] = [[] for _ in STATION_POSITIONS]
    for tower_name, station in zip(HANOI_TOWER_NAMES, tower_stations):
        stacks[station].append(tower_name)
    return stacks


class MoveGroupPythonInterface(Node):
    """MoveGroupPythonInterface for ROS2"""

    def __init__(self, executor: SingleThreadedExecutor):
        super().__init__("hanoi_spawn_objects")

        self.PLANNING_FRAME = "world"

        self._executor = executor

        self.get_logger().info("Initializing MoveGroupPythonInterface...")

        self.collision_object_publisher = self.create_publisher(
            CollisionObject, "/collision_object", 10
        )

        self.attached_collision_object_publisher = self.create_publisher(
            AttachedCollisionObject, "/attached_collision_object", 10
        )
        self.hanoi_station_client = self.create_client(
            SetHanoiTowerStations,
            "/set_hanoi_tower_stations",
        )

        self.get_logger().info(f"Planning frame: {self.PLANNING_FRAME}")

        time.sleep(1.0)

    def wait_for_state_update(self) -> None:
        self._executor.spin_once(timeout_sec=0.5)

    def add_box(
        self,
        *,
        box_name: str,
        box_pose: Pose,
        size: tuple[float, float, float],
    ) -> None:
        """
        Description:
            1. Add a box to rviz, Moveit_planner will think of which as an obstacle.
            2. An example is shown in the main function below.
            3. Google scene.add_box for more details
        """

        box = SolidPrimitive(
            type=SolidPrimitive.BOX,
            dimensions=size,
        )

        collision_object = CollisionObject(
            header=Header(
                frame_id=self.PLANNING_FRAME,
                stamp=self.get_clock().now().to_msg(),
            ),
            id=box_name,
            primitives=[box],
            primitive_poses=[box_pose],
            operation=CollisionObject.ADD,
        )

        self.collision_object_publisher.publish(collision_object)

        self.get_logger().info(f"Added box: {box_name}")
        self.wait_for_state_update()

    def add_mesh(
        self,
        *,
        mesh_name: str,
        mesh_position: Point,
        file_path: str,
        scale: tuple[float, float, float],
    ) -> None:
        """
        Description:
            1. Add a mesh to rviz, Moveit_planner will think of which as an obstacle.
            2. An example is shown in the main function below.
        """
        pose = Pose(
            position=mesh_position,
            # adjust mesh orientation
            orientation=Quaternion(x=0.7071081, y=0.0, z=0.0, w=0.7071081),
        )
        collision_object = CollisionObject(
            header=Header(
                frame_id=self.PLANNING_FRAME,
                stamp=self.get_clock().now().to_msg(),
            ),
            id=mesh_name,
            meshes=[load_mesh_from_file(file_path, scale)],
            mesh_poses=[pose],
            operation=CollisionObject.ADD,
        )

        self.collision_object_publisher.publish(collision_object)

        self.get_logger().info(f"Added mesh: {mesh_name}")
        self.wait_for_state_update()

    def attach_object(self, *, object_name: str, link_name: str) -> None:
        """
        Description:
            1. Make sure the object has been added to rviz
            2. Attach a object to link_frame(usually 'link5'), and the object will move with the link_frame.
            3. Google scene.attach_box for more details
        """
        attached_object = AttachedCollisionObject(
            link_name=link_name,
            object=CollisionObject(id=object_name, operation=CollisionObject.ADD),
            touch_links=[link_name],
        )

        self.attached_collision_object_publisher.publish(attached_object)

        self.get_logger().info(f"Attached object: {object_name} to {link_name}")
        self.wait_for_state_update()

    def detach_object(self, *, object_name: str, link_name: str) -> None:
        """
        Description:
            1. Detach a object from link_frame(usually 'link5'), and the object will not move with the link_frame.
            2. An example is shown in the main function below.
            3. Google scene.detach_box for more details
        """
        attached_object = AttachedCollisionObject(
            link_name=link_name,
            object=CollisionObject(id=object_name, operation=CollisionObject.REMOVE),
        )

        self.attached_collision_object_publisher.publish(attached_object)

        self.get_logger().info(f"Detached object: {object_name} from {link_name}")
        self.wait_for_state_update()

    def remove_object(self, object_name: str) -> None:
        """
        Description:
            Remove a object from rviz.
        """
        collision_object = CollisionObject(
            header=Header(
                frame_id=self.PLANNING_FRAME,
                stamp=self.get_clock().now().to_msg(),
            ),
            id=object_name,
            operation=CollisionObject.REMOVE,
        )

        self.collision_object_publisher.publish(collision_object)

        self.get_logger().info(f"Removed object: {object_name}")
        self.wait_for_state_update()

    def send_hanoi_station_request(
        self,
        tower_stations: tuple[int, ...],
        target_station: int,
    ) -> bool:
        self.get_logger().info("Waiting for /set_hanoi_tower_stations service...")
        if not self.hanoi_station_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("/set_hanoi_tower_stations service is not available")
            return False

        request = SetHanoiTowerStations.Request()
        request.tower_stations = list(tower_stations)
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
    tower_stations, target_station = read_hanoi_request()

    rclpy.init(args=args)

    path_plan_object: MoveGroupPythonInterface | None = None
    try:
        executor = SingleThreadedExecutor()
        # Declare the path-planning object
        path_plan_object = MoveGroupPythonInterface(executor)

        executor.add_node(path_plan_object)

        stacks = build_stacks_from_tower_stations(tower_stations)
        tower_spacing = Tower_height - Tower_overlap
        for station_index, stack in enumerate(stacks):
            station_x, station_y = STATION_POSITIONS[station_index]
            for stack_index, tower_name in enumerate(stack):
                tower_position = Point(
                    x=station_x,
                    y=station_y,
                    z=Tower_base + stack_index * tower_spacing,
                )
                path_plan_object.add_mesh(
                    mesh_name=tower_name,
                    mesh_position=tower_position,
                    file_path=MESH_FILE_PATH[tower_name],
                    scale=MESH_SCALE,
                )

        print("Spawned all tower meshes successfully!")

        for index, (x, y, z) in enumerate(BOX_POSITIONS):
            path_plan_object.add_box(
                box_name=f"box_{index + 1}",
                box_pose=Pose(
                    orientation=Quaternion(w=1.0),
                    position=Point(x=x, y=y, z=z),
                ),
                size=BOX_SIZE,
            )

        print("Spawned all boxes successfully!")

        path_plan_object.send_hanoi_station_request(tower_stations, target_station)

        input("Press Enter to exit...")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if path_plan_object is not None:
            path_plan_object.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
