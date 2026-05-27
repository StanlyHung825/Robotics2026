import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AllowedCollisionMatrix,
    AttachedCollisionObject,
    CollisionObject,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
import rclpy
import trimesh
from shape_msgs.msg import Mesh, MeshTriangle
from std_msgs.msg import Header

from myrobot.hanoi_waypoint_planning import (
    End_effector_contact_offset,
    HANOI_TOWER_NAMES,
    Tower_mesh_height,
)

TOOL_LINK = "link5"
ROBOT_LINKS = ("link0", "link1", "link2", "link3", "link4", "link5")
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


class MoveIt2AcmManager:
    def __init__(
        self,
        node: Any,
        *,
        planning_frame: str = "world",
        wait_for_future: Callable[[Any, float], bool] | None = None,
    ) -> None:
        self._node = node
        self._planning_frame = planning_frame
        self._wait_for_future = wait_for_future or self._default_wait_for_future

        self.collision_object_publisher = node.create_publisher(
            CollisionObject,
            "/collision_object",
            10,
        )
        self.attached_collision_object_publisher = node.create_publisher(
            AttachedCollisionObject,
            "/attached_collision_object",
            10,
        )
        self.planning_scene_publisher = node.create_publisher(
            PlanningScene,
            "/planning_scene",
            10,
        )
        self.apply_planning_scene_client = node.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
        )
        self.get_planning_scene_client = node.create_client(
            GetPlanningScene,
            "/get_planning_scene",
        )

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
        self._apply_planning_scene(planning_scene)

        if log:
            self._node.get_logger().info(
                "Disabled collision checks between Hanoi towers, "
                "and between towers and arm links"
            )

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
            self._node.get_logger().warn("get_planning_scene service is not available")
            return None

        request = GetPlanningScene.Request(
            components=PlanningSceneComponents(
                components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX,
            )
        )
        future = self.get_planning_scene_client.call_async(request)
        if not self._wait_for_future(future, 2.0) or future.result() is None:
            self._node.get_logger().warn("Could not read current planning scene")
            return None

        return future.result().scene.allowed_collision_matrix

    def remove_world_object(self, object_name: str) -> None:
        collision_object = CollisionObject(
            header=Header(
                frame_id=self._planning_frame,
                stamp=self._node.get_clock().now().to_msg(),
            ),
            id=object_name,
            operation=CollisionObject.REMOVE,
        )

        self.collision_object_publisher.publish(collision_object)
        self._node.get_logger().info(f"Removed world object: {object_name}")
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
                frame_id=self._planning_frame,
                stamp=self._node.get_clock().now().to_msg(),
            ),
            id=object_name,
            meshes=[load_mesh_from_file(MESH_FILE_PATH[object_name], MESH_SCALE)],
            mesh_poses=[Pose(position=position, orientation=MESH_ORIENTATION)],
            operation=CollisionObject.ADD,
        )

        self.collision_object_publisher.publish(collision_object)
        self._node.get_logger().info(
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
                    stamp=self._node.get_clock().now().to_msg(),
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
        self._node.get_logger().info(f"Attached object: {object_name} to {link_name}")
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
        self._node.get_logger().info(f"Detached object: {object_name} from {link_name}")
        self.wait_for_state_update()
        self.add_world_mesh(object_name=object_name, position=world_position)
        self.allow_hanoi_contacts(log=False)

    def wait_for_state_update(self) -> None:
        time.sleep(0.2)

    def _apply_planning_scene(self, planning_scene: PlanningScene) -> None:
        if self.apply_planning_scene_client.wait_for_service(timeout_sec=1.0):
            future = self.apply_planning_scene_client.call_async(
                ApplyPlanningScene.Request(scene=planning_scene)
            )
            if not self._wait_for_future(future, 2.0):
                self._node.get_logger().warn(
                    "Timed out while applying planning-scene update"
                )
            elif future.result() is not None and not future.result().success:
                self._node.get_logger().warn("MoveIt rejected the planning-scene update")
            return

        for _ in range(5):
            self.planning_scene_publisher.publish(planning_scene)
            time.sleep(0.1)

    @staticmethod
    def _default_wait_for_future(future: Any, timeout_sec: float = 30.0) -> bool:
        start_time = time.time()
        while rclpy.ok() and not future.done():
            if (time.time() - start_time) > timeout_sec:
                return False
            time.sleep(0.01)
        return future.done()
