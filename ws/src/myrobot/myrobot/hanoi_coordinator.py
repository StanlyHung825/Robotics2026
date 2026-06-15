#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import threading
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
import trimesh

from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from std_msgs.msg import Header

from hanoi_interface.srv import GetHanoiStatus
from myrobot_interfaces.srv import SetHanoiTowerStations

# Positions for stations
STATION_POSITIONS = (
    (0.25, 0.15),   # Station 0 (Left / A)
    (0.25, 0.0),    # Station 1 (Middle / B)
    (0.25, -0.15),  # Station 2 (Right / C)
)

BOX_SIZE = (0.1, 0.001, 0.1)
BOX_POSITIONS = (
    (0.25, -0.075, 0.05),
    (0.25, 0.075, 0.05),
)

NUM_DISKS = 3
HANOI_TOWER_NAMES = ("tower1", "tower2", "tower3") # tower1=Large, tower2=Medium, tower3=Small

# Tower geometry constants
Tower_base = 0.0014
Tower_height = 0.025
Tower_overlap = 0.015

# Mesh files configuration
MESH_DIR = Path(get_package_share_directory("myplan")) / "mesh"
MESH_FILE_PATH = {
    tower_name: str(MESH_DIR / f"{tower_name}.stl")
    for tower_name in HANOI_TOWER_NAMES
}
MESH_SCALE = (0.00095, 0.00095, 0.00095)


def load_mesh_from_file(file_path: str, scale: tuple[float, float, float]) -> Mesh:
    mesh_data = trimesh.load(file_path, force="mesh")
    assert isinstance(mesh_data, trimesh.base.Trimesh)

    vertices = [
        Point(x=float(v[0]) * scale[0], y=float(v[1]) * scale[1], z=float(v[2]) * scale[2])
        for v in mesh_data.vertices
    ]
    triangles = [
        MeshTriangle(vertex_indices=[int(f[0]), int(f[1]), int(f[2])])
        for f in mesh_data.faces if len(f) == 3
    ]
    return Mesh(triangles=triangles, vertices=vertices)


class HanoiCoordinator(Node):
    def __init__(self):
        super().__init__('hanoi_coordinator')
        self.PLANNING_FRAME = "world"

        # Clients
        self.status_client = self.create_client(GetHanoiStatus, 'get_hanoi_positions')
        self.planner_client = self.create_client(SetHanoiTowerStations, '/set_hanoi_tower_stations')

        # Publisher for planning scene collision objects
        self.collision_object_publisher = self.create_publisher(CollisionObject, '/collision_object', 10)

        self.get_logger().info("Hanoi Coordinator Node initialized.")

    def query_status(self) -> tuple[tuple[int, int, int] | None, int | None, int, int]:
        """Queries get_hanoi_positions service for current camera & voice inputs"""
        if not self.status_client.service_is_ready():
            return None, None, 1, 1

        request = GetHanoiStatus.Request()
        future = self.status_client.call_async(request)
        
        # Wait for service result using spin_once inside loop to keep callbacks alive
        start_time = time.time()
        while rclpy.ok() and not future.done():
            if time.time() - start_time > 2.0:
                self.get_logger().warn("Timeout waiting for get_hanoi_positions response")
                return None, None, 1, 1
            time.sleep(0.05)

        try:
            response = future.result()
            if response is None:
                return None, None, 1, 1
            
            # Map response 1-based index (1=A, 2=B, 3=C) to 0-based (0=A, 1=B, 2=C)
            large = self._map_station(response.large_pos)
            medium = self._map_station(response.medium_pos)
            small = self._map_station(response.small_pos)
            target = self._map_station(response.target_pos)

            left_obstacle = getattr(response, 'left_obstacle', 1)
            right_obstacle = getattr(response, 'right_obstacle', 1)

            towers = None
            if large is not None and medium is not None and small is not None:
                towers = (large, medium, small)

            return towers, target, left_obstacle, right_obstacle
        except Exception as e:
            self.get_logger().error(f"Error querying status: {e}")
            return None, None, 1, 1

    def _map_station(self, pos: int) -> int | None:
        if pos == 1: return 0
        if pos == 2: return 1
        if pos == 3: return 2
        return None

    def remove_object(self, object_name: str):
        collision_object = CollisionObject(
            header=Header(frame_id=self.PLANNING_FRAME, stamp=self.get_clock().now().to_msg()),
            id=object_name,
            operation=CollisionObject.REMOVE
        )
        self.collision_object_publisher.publish(collision_object)
        time.sleep(0.1)

    def add_mesh(self, mesh_name: str, mesh_position: Point):
        pose = Pose(
            position=mesh_position,
            orientation=Quaternion(x=0.7071081, y=0.0, z=0.0, w=0.7071081)
        )
        collision_object = CollisionObject(
            header=Header(frame_id=self.PLANNING_FRAME, stamp=self.get_clock().now().to_msg()),
            id=mesh_name,
            meshes=[load_mesh_from_file(MESH_FILE_PATH[mesh_name], MESH_SCALE)],
            mesh_poses=[pose],
            operation=CollisionObject.ADD
        )
        self.collision_object_publisher.publish(collision_object)
        time.sleep(0.2)

    def add_box(self, box_name: str, box_pose: Pose, size: tuple[float, float, float]):
        box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=size)
        collision_object = CollisionObject(
            header=Header(frame_id=self.PLANNING_FRAME, stamp=self.get_clock().now().to_msg()),
            id=box_name,
            primitives=[box],
            primitive_poses=[box_pose],
            operation=CollisionObject.ADD
        )
        self.collision_object_publisher.publish(collision_object)
        time.sleep(0.1)

    def spawn_scene_objects(self, tower_stations: tuple[int, int, int], left_obstacle: int, right_obstacle: int):
        """Clear and spawn Hanoi meshes and boxes in MoveIt environment"""
        self.get_logger().info("Spawning collision objects in MoveIt planning scene...")
        
        # 1. Clear old objects
        for name in HANOI_TOWER_NAMES:
            self.remove_object(name)
        for i in range(len(BOX_POSITIONS)):
            self.remove_object(f"box_{i+1}")

        # 2. Build stacks
        stacks = [[] for _ in range(3)]
        # Append largest first, then medium, then small
        stacks[tower_stations[0]].append("tower1")
        stacks[tower_stations[1]].append("tower2")
        stacks[tower_stations[2]].append("tower3")

        # 3. Spawn meshes
        tower_spacing = Tower_height - Tower_overlap
        for station_idx, stack in enumerate(stacks):
            station_x, station_y = STATION_POSITIONS[station_idx]
            for stack_idx, name in enumerate(stack):
                pos = Point(
                    x=station_x,
                    y=station_y,
                    z=Tower_base + stack_idx * tower_spacing
                )
                self.add_mesh(name, pos)

        # 4. Spawn boxes (barriers) conditionally
        # BOX_POSITIONS[0] corresponds to human Left barrier (between B and C) -> box_1
        # BOX_POSITIONS[1] corresponds to human Right barrier (between A and B) -> box_2
        if left_obstacle == 1:
            pose = Pose(
                position=Point(x=BOX_POSITIONS[0][0], y=BOX_POSITIONS[0][1], z=BOX_POSITIONS[0][2]),
                orientation=Quaternion(w=1.0)
            )
            self.add_box("box_1", pose, BOX_SIZE)
        if right_obstacle == 1:
            pose = Pose(
                position=Point(x=BOX_POSITIONS[1][0], y=BOX_POSITIONS[1][1], z=BOX_POSITIONS[1][2]),
                orientation=Quaternion(w=1.0)
            )
            self.add_box("box_2", pose, BOX_SIZE)

        self.get_logger().info("Planning scene objects spawned successfully.")

    def trigger_hanoi_planner(self, tower_stations: tuple[int, int, int], target_station: int, left_obstacle: int, right_obstacle: int) -> bool:
        """Call set_hanoi_tower_stations service to start path execution"""
        if not self.planner_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/set_hanoi_tower_stations planner service not available!")
            return False

        request = SetHanoiTowerStations.Request()
        request.tower_stations = list(tower_stations)
        request.obstacle = [bool(left_obstacle), bool(right_obstacle)]
        request.target_station = target_station

        self.get_logger().info(f"Sending planning request: tower_stations={tower_stations}, target_station={target_station}, obstacles={[bool(left_obstacle), bool(right_obstacle)]}")
        future = self.planner_client.call_async(request)
        
        # Wait for result
        while rclpy.ok() and not future.done():
            time.sleep(0.1)

        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Execution Succeeded! Message: {response.message}")
                return True
            else:
                self.get_logger().error(f"Execution Failed! Message: {response.message}")
                return False
        except Exception as e:
            self.get_logger().error(f"Service call failed with exception: {e}")
            return False

    def run_loop(self):
        """Interactive loop waiting for valid status and keypress with non-blocking check"""
        import select
        station_names = {0: 'A (右)', 1: 'B (中)', 2: 'C (左)'}
        last_large, last_medium, last_small, last_target = -1, -1, -1, -1
        last_left_obstacle, last_right_obstacle = -1, -1
        last_known_towers = None

        prompt_printed = False

        while rclpy.ok():
            # Check for manual override keypress immediately, even if camera hasn't detected anything
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                line = sys.stdin.readline().strip()
                if line.lower() == 'm':
                    print("\n⌨️ [手動設定模式] 請輸入以下狀態 (跳過相機與語音)：")
                    try:
                        man_small = int(input("  ● 小河內塔 (tower3) 站點 (0=A, 1=B, 2=C): "))
                        man_medium = int(input("  ● 中河內塔 (tower2) 站點 (0=A, 1=B, 2=C): "))
                        man_large = int(input("  ● 大河內塔 (tower1) 站點 (0=A, 1=B, 2=C): "))
                        man_target = int(input("  ● 目標站點 (0=A, 1=B, 2=C): "))
                        man_left = int(input("  ● 左側障礙物 (0=無, 1=有): "))
                        man_right = int(input("  ● 右側障礙物 (0=無, 1=有): "))
                        
                        if not (0 <= man_small <= 2 and 0 <= man_medium <= 2 and 0 <= man_large <= 2 and 0 <= man_target <= 2):
                            print("❌ 輸入的站點編號必須在 0, 1, 2 之間，設定取消。")
                            prompt_printed = False
                            continue
                            
                        active_towers = (man_large, man_medium, man_small)
                        target_station = man_target
                        left_obstacle = man_left
                        right_obstacle = man_right
                        
                        print("\n✅ 手動狀態已就緒！")
                        print(f"  ● 塔起點: 大={man_large}, 中={man_medium}, 小={man_small}")
                        print(f"  ● 目標點: {man_target}")
                        print(f"  ● 障礙物: 左={man_left}, 右={man_right}")
                        
                        # Direct Execution
                        print("🚀 發送規劃請求中...")
                        start_time = time.time()
                        success = self.trigger_hanoi_planner(active_towers, target_station, left_obstacle, right_obstacle)
                        elapsed_time = time.time() - start_time
                        if success:
                            print(f"🎉 疊放流程全部執行完成！(總計耗時: {elapsed_time:.2f} 秒)")
                        else:
                            print(f"❌ 執行失敗，請檢查 MoveIt 行程與連線。(耗時: {elapsed_time:.2f} 秒)")
                        
                        print("\n準備進入下一輪偵測與控制...")
                        prompt_printed = False
                        time.sleep(2.0)
                        continue
                    except ValueError:
                        print("❌ 輸入格式錯誤，請輸入整數 0, 1 或 2，設定取消。")
                        prompt_printed = False
                        continue

            tower_stations, target_station, left_obstacle, right_obstacle = self.query_status()
            
            if tower_stations is not None:
                last_known_towers = tower_stations
            
            active_towers = tower_stations or last_known_towers
            
            if active_towers is None:
                print("\r[等待輸入] 正在等待相機辨識河內塔位置...", end="")
                time.sleep(0.5)
                continue

            large, medium, small = active_towers
            
            # Print current state if changed
            state_changed = (large != last_large or medium != last_medium or 
                             small != last_small or target_station != last_target or
                             left_obstacle != last_left_obstacle or right_obstacle != last_right_obstacle)
            
            if state_changed:
                print("\n" + "="*50)
                print("【目前偵測狀態】")
                print(f"  ● 小河內塔 (tower3)：{station_names.get(small, '未找到')}")
                print(f"  ● 中河內塔 (tower2)：{station_names.get(medium, '未找到')}")
                print(f"  ● 大河內塔 (tower1)：{station_names.get(large, '未找到')}")
                print(f"  ● 語音目標位置 (target)：{station_names.get(target_station, '尚未輸入')}")
                print(f"  ● 障礙物狀態 (obstacles)：左側={left_obstacle}, 右側={right_obstacle}")
                print("="*50)

                last_large, last_medium, last_small, last_target = large, medium, small, target_station
                last_left_obstacle, last_right_obstacle = left_obstacle, right_obstacle
                prompt_printed = False # Reset prompt so it prints again after state update

            if target_station is None:
                print("\r[等待輸入] 正在等待語音指令 (B區/C區/A區)...", end="")
                time.sleep(0.5)
                continue

            # If all are detected and target is set, prompt for start
            if not prompt_printed:
                print("\n" + "*"*60)
                print("  🌟 相機辨識 與 語音目標 均已就緒！")
                print("  👉 請在終端機輸入 【Enter】 鍵開始在模擬中堆疊河內塔...")
                print("  💡 (在您按下 Enter 之前，若再次說話或相機更新，狀態仍會即時改變)")
                print("*"*60)
                prompt_printed = True

            # Use non-blocking select to check for stdin keypress
            rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
            if rlist:
                line = sys.stdin.readline().strip()
                
                # If user entered 'm', switch to full manual keyboard input mode
                if line.lower() == 'm':
                    print("\n⌨️ [手動設定模式] 請輸入以下狀態 (跳過相機與語音)：")
                    try:
                        # 1. 取得大中小塔的站點 (0=A, 1=B, 2=C)
                        man_small = int(input("  ● 小河內塔 (tower3) 站點 (0=A, 1=B, 2=C): "))
                        man_medium = int(input("  ● 中河內塔 (tower2) 站點 (0=A, 1=B, 2=C): "))
                        man_large = int(input("  ● 大河內塔 (tower1) 站點 (0=A, 1=B, 2=C): "))
                        
                        # 2. 取得目標站點 (0=A, 1=B, 2=C)
                        man_target = int(input("  ● 目標站點 (0=A, 1=B, 2=C): "))
                        
                        # 3. 取得障礙物狀態
                        man_left = int(input("  ● 左側障礙物 (0=無, 1=有): "))
                        man_right = int(input("  ● 右側障礙物 (0=無, 1=有): "))
                        
                        if not (0 <= man_small <= 2 and 0 <= man_medium <= 2 and 0 <= man_large <= 2 and 0 <= man_target <= 2):
                            print("❌ 輸入的站點編號必須在 0, 1, 2 之間，設定取消。")
                            prompt_printed = False
                            continue
                            
                        # 更新為手動設定的值
                        active_towers = (man_large, man_medium, man_small)
                        target_station = man_target
                        left_obstacle = man_left
                        right_obstacle = man_right
                        
                        print("\n✅ 手動狀態已就緒！")
                        print(f"  ● 塔起點: 大={man_large}, 中={man_medium}, 小={man_small}")
                        print(f"  ● 目標點: {man_target}")
                        print(f"  ● 障礙物: 左={man_left}, 右={man_right}")
                    except ValueError:
                        print("❌ 輸入格式錯誤，請輸入整數 0, 1 或 2，設定取消。")
                        prompt_printed = False
                        continue
                
                # Start planning
                print("🚀 發送規劃請求中...")
                start_time = time.time()
                success = self.trigger_hanoi_planner(active_towers, target_station, left_obstacle, right_obstacle)
                elapsed_time = time.time() - start_time
                if success:
                    print(f"🎉 疊放流程全部執行完成！(總計耗時: {elapsed_time:.2f} 秒)")
                else:
                    print(f"❌ 執行失敗，請檢查 MoveIt 行程與連線。(耗時: {elapsed_time:.2f} 秒)")
                
                print("\n準備進入下一輪偵測與控制...")
                prompt_printed = False
                time.sleep(2.0)


def main(args=None):
    rclpy.init(args=args)
    node = HanoiCoordinator()

    # Spin the node in a separate thread so service client calls can process callbacks
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run_loop()
    except KeyboardInterrupt:
        print("\n使用者中斷程式。")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
