import time
from collections.abc import Callable
from functools import lru_cache
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
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from std_msgs.msg import Header

from myrobot.hanoi_model import (
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
PRELOAD_TOWER_PARKING_POSITION = Point(x=0.0, y=0.0, z=-1.0)

for mesh in MESH_FILE_PATH.values():
    if not Path(mesh).exists():
        raise FileNotFoundError(f"Mesh path error: {mesh}")


@lru_cache(maxsize=None)
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
        self._world_mesh_object_names: set[str] = set()
        self._pick_place_scene_update_mode = str(
            self._get_or_declare_parameter(
                "pick_place_scene_update_mode",
                "async_topic",
            )
        )
        self._pick_place_scene_settle_time = max(
            0.0,
            float(
                self._get_or_declare_parameter(
                    "pick_place_scene_settle_time",
                    0.05,
                )
            ),
        )

        if self._pick_place_scene_update_mode not in ("async_topic", "sync_apply"):
            node.get_logger().warn(
                "Unknown pick_place_scene_update_mode="
                f"{self._pick_place_scene_update_mode!r}; using async_topic"
            )
            self._pick_place_scene_update_mode = "async_topic"

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

    def preload_hanoi_tower_meshes(self) -> None:
        self.preload_world_meshes(
            {
                tower_name: PRELOAD_TOWER_PARKING_POSITION
                for tower_name in HANOI_TOWER_NAMES
            }
        )

    def preload_world_meshes(self, object_positions: dict[str, Point]) -> None:
        if not object_positions:
            return

        planning_scene = PlanningScene(is_diff=True)
        planning_scene.world.collision_objects = [
            self._build_world_mesh_object(
                object_name=object_name,
                position=position,
            )
            for object_name, position in object_positions.items()
        ]
        self._apply_planning_scene(planning_scene)
        self._world_mesh_object_names.update(object_positions)
        self._node.get_logger().info(
            "Preloaded world mesh objects: "
            f"{', '.join(object_positions.keys())}"
        )
        self.wait_for_state_update()

    def move_world_meshes(self, object_positions: dict[str, Point]) -> None:
        if not object_positions:
            return

        collision_objects = []
        for object_name, position in object_positions.items():
            if object_name in self._world_mesh_object_names:
                collision_objects.append(
                    self._build_world_mesh_move_object(
                        object_name=object_name,
                        position=position,
                    )
                )
                continue

            collision_objects.append(
                self._build_world_mesh_object(
                    object_name=object_name,
                    position=position,
                )
            )
            self._world_mesh_object_names.add(object_name)

        planning_scene = PlanningScene(is_diff=True)
        planning_scene.world.collision_objects = collision_objects
        self._apply_planning_scene(planning_scene)
        self._node.get_logger().info(
            "Updated world mesh positions: "
            f"{', '.join(object_positions.keys())}"
        )

    def refresh_world_boxes(
        self,
        *,
        all_object_names: tuple[str, ...],
        enabled_boxes: dict[str, tuple[Pose, tuple[float, float, float]]],
    ) -> None:
        collision_objects = []
        for object_name in all_object_names:
            if object_name not in enabled_boxes:
                collision_objects.append(self._build_world_remove_object(object_name))
                continue

            pose, size = enabled_boxes[object_name]
            collision_objects.append(
                self._build_world_box_object(
                    object_name=object_name,
                    pose=pose,
                    size=size,
                )
            )

        planning_scene = PlanningScene(is_diff=True)
        planning_scene.world.collision_objects = collision_objects
        self._apply_planning_scene(planning_scene)
        self._node.get_logger().info(
            "Refreshed world boxes: "
            f"{', '.join(enabled_boxes) if enabled_boxes else 'none'}"
        )

    def remove_world_object(self, object_name: str) -> None:
        collision_object = self._build_world_remove_object(object_name)

        self.collision_object_publisher.publish(collision_object)
        self._world_mesh_object_names.discard(object_name)
        self._node.get_logger().info(f"Removed world object: {object_name}")
        self.wait_for_state_update()
        self.allow_hanoi_contacts(log=False)

    def remove_world_objects(self, object_names: tuple[str, ...]) -> None:
        if not object_names:
            return

        collision_objects = [
            self._build_world_remove_object(object_name)
            for object_name in object_names
        ]

        planning_scene = PlanningScene(is_diff=True)
        planning_scene.world.collision_objects = collision_objects
        self._apply_planning_scene(planning_scene)

        for collision_object in collision_objects:
            self.collision_object_publisher.publish(collision_object)

        self._world_mesh_object_names.difference_update(object_names)
        self._node.get_logger().info(
            f"Removed world objects: {', '.join(object_names)}"
        )
        self.wait_for_state_update()
        self.allow_hanoi_contacts(log=False)

    def add_world_mesh(
        self,
        *,
        object_name: str,
        position: Point,
    ) -> None:
        collision_object = self._build_world_mesh_object(
            object_name=object_name,
            position=position,
        )

        self.collision_object_publisher.publish(collision_object)
        self._world_mesh_object_names.add(object_name)
        self._node.get_logger().info(
            f"Added world object: {object_name} at "
            f"({position.x:.3f}, {position.y:.3f}, {position.z:.3f})"
        )
        self.wait_for_state_update()
        self.allow_hanoi_contacts(log=False)

    def add_world_box(
        self,
        *,
        object_name: str,
        pose: Pose,
        size: tuple[float, float, float],
    ) -> None:
        collision_object = self._build_world_box_object(
            object_name=object_name,
            pose=pose,
            size=size,
        )

        self.collision_object_publisher.publish(collision_object)
        self._node.get_logger().info(
            f"Added world box: {object_name} at "
            f"({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})"
        )
        self.wait_for_state_update()
        self.allow_hanoi_contacts(log=False)

    def attach_object(self, *, object_name: str, link_name: str = TOOL_LINK) -> None:
        world_remove_object = self._build_world_remove_object(object_name)
        attached_object = self._build_attached_mesh_object(
            object_name=object_name,
            link_name=link_name,
        )

        planning_scene = PlanningScene(is_diff=True)
        planning_scene.world.collision_objects = [world_remove_object]
        planning_scene.robot_state.is_diff = True
        planning_scene.robot_state.attached_collision_objects = [attached_object]
        applied = self._update_pick_place_scene(planning_scene)

        self._node.get_logger().info(
            f"Attached object: {object_name} to {link_name}"
        )
        if not applied:
            self.wait_for_state_update()
        self._world_mesh_object_names.discard(object_name)

    def remove_attached_object(
        self,
        *,
        object_name: str,
        link_name: str = TOOL_LINK,
    ) -> None:
        attached_object = AttachedCollisionObject(
            link_name=link_name,
            object=CollisionObject(id=object_name, operation=CollisionObject.REMOVE),
        )

        self.attached_collision_object_publisher.publish(attached_object)
        self._node.get_logger().info(
            f"Removed attached object: {object_name} from {link_name}"
        )
        self.wait_for_state_update()

    def remove_attached_objects(
        self,
        *,
        object_names: tuple[str, ...],
        link_name: str = TOOL_LINK,
    ) -> None:
        if not object_names:
            return

        planning_scene = PlanningScene(is_diff=True)
        planning_scene.robot_state.is_diff = True
        planning_scene.robot_state.attached_collision_objects = [
            self._build_attached_remove_object(
                object_name=object_name,
                link_name=link_name,
            )
            for object_name in object_names
        ]
        self._apply_planning_scene(planning_scene)
        self._node.get_logger().info(
            "Removed attached objects: "
            f"{', '.join(object_names)}"
        )

    def detach_object(
        self,
        *,
        object_name: str,
        world_position: Point,
        link_name: str = TOOL_LINK,
    ) -> None:
        planning_scene = PlanningScene(is_diff=True)
        planning_scene.world.collision_objects = [
            self._build_world_mesh_object(
                object_name=object_name,
                position=world_position,
            )
        ]
        planning_scene.robot_state.is_diff = True
        planning_scene.robot_state.attached_collision_objects = [
            self._build_attached_remove_object(
                object_name=object_name,
                link_name=link_name,
            )
        ]
        applied = self._update_pick_place_scene(planning_scene)

        self._node.get_logger().info(
            f"Detached object: {object_name} from {link_name}"
        )
        if not applied:
            self.wait_for_state_update()
        self._world_mesh_object_names.add(object_name)

    def wait_for_state_update(self) -> None:
        time.sleep(0.2)

    def _update_pick_place_scene(self, planning_scene: PlanningScene) -> bool:
        if self._pick_place_scene_update_mode == "sync_apply":
            return self._apply_planning_scene(planning_scene)

        self.planning_scene_publisher.publish(planning_scene)
        if self._pick_place_scene_settle_time > 0.0:
            time.sleep(self._pick_place_scene_settle_time)
        return True

    def _build_world_mesh_object(
        self,
        *,
        object_name: str,
        position: Point,
    ) -> CollisionObject:
        return CollisionObject(
            header=Header(
                frame_id=self._planning_frame,
                stamp=self._node.get_clock().now().to_msg(),
            ),
            id=object_name,
            meshes=[load_mesh_from_file(MESH_FILE_PATH[object_name], MESH_SCALE)],
            mesh_poses=[Pose(position=position, orientation=MESH_ORIENTATION)],
            operation=CollisionObject.ADD,
        )

    def _build_world_mesh_move_object(
        self,
        *,
        object_name: str,
        position: Point,
    ) -> CollisionObject:
        return CollisionObject(
            header=Header(
                frame_id=self._planning_frame,
                stamp=self._node.get_clock().now().to_msg(),
            ),
            id=object_name,
            mesh_poses=[Pose(position=position, orientation=MESH_ORIENTATION)],
            operation=CollisionObject.MOVE,
        )

    def _build_world_box_object(
        self,
        *,
        object_name: str,
        pose: Pose,
        size: tuple[float, float, float],
    ) -> CollisionObject:
        return CollisionObject(
            header=Header(
                frame_id=self._planning_frame,
                stamp=self._node.get_clock().now().to_msg(),
            ),
            id=object_name,
            primitives=[
                SolidPrimitive(
                    type=SolidPrimitive.BOX,
                    dimensions=list(size),
                )
            ],
            primitive_poses=[pose],
            operation=CollisionObject.ADD,
        )

    def _build_world_remove_object(self, object_name: str) -> CollisionObject:
        return CollisionObject(
            header=Header(
                frame_id=self._planning_frame,
                stamp=self._node.get_clock().now().to_msg(),
            ),
            id=object_name,
            operation=CollisionObject.REMOVE,
        )

    def _build_attached_mesh_object(
        self,
        *,
        object_name: str,
        link_name: str,
    ) -> AttachedCollisionObject:
        return AttachedCollisionObject(
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

    @staticmethod
    def _build_attached_remove_object(
        *,
        object_name: str,
        link_name: str,
    ) -> AttachedCollisionObject:
        return AttachedCollisionObject(
            link_name=link_name,
            object=CollisionObject(id=object_name, operation=CollisionObject.REMOVE),
        )

    def _apply_planning_scene(self, planning_scene: PlanningScene) -> bool:
        if self.apply_planning_scene_client.wait_for_service(timeout_sec=1.0):
            for attempt in range(3):
                future = self.apply_planning_scene_client.call_async(
                    ApplyPlanningScene.Request(scene=planning_scene)
                )
                if not self._wait_for_future(future, 5.0):
                    raise RuntimeError(
                        "Timed out while applying planning-scene update"
                    )
                elif future.result() is not None and future.result().success:
                    return True

                self._node.get_logger().warn(
                    "MoveIt rejected the planning-scene update "
                    f"(attempt {attempt + 1}/3), retrying in 0.5s..."
                )
                time.sleep(0.5)

            self._node.get_logger().error(
                "MoveIt rejected the planning-scene update after 3 attempts. "
                "Proceeding with warning..."
            )
            return False

        for _ in range(5):
            self.planning_scene_publisher.publish(planning_scene)
            time.sleep(0.1)
        return False

    def _get_or_declare_parameter(self, name: str, default: Any) -> Any:
        if hasattr(self._node, "has_parameter") and self._node.has_parameter(name):
            return self._node.get_parameter(name).value

        if hasattr(self._node, "declare_parameter"):
            return self._node.declare_parameter(name, default).value

        return default

    def _default_wait_for_future(self, future: Any, timeout_sec: float = 30.0) -> bool:
        start_time = time.time()
        while rclpy.ok() and not future.done():
            if not getattr(self._node, "initialized", False):
                rclpy.spin_once(self._node, timeout_sec=0.01)
            if (time.time() - start_time) > timeout_sec:
                return False
            time.sleep(0.01)
        return future.done()
