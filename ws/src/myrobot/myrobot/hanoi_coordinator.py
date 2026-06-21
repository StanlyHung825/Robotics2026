#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import threading

import rclpy
from rclpy.node import Node

from hanoi_interface.srv import GetHanoiStatus
from myrobot.hanoi_model import status_station_to_planner_station
from myrobot_interfaces.srv import SetHanoiTowerStations


class HanoiCoordinator(Node):
    def __init__(self):
        super().__init__('hanoi_coordinator')
        self.status_client = self.create_client(GetHanoiStatus, 'get_hanoi_positions')
        self.planner_client = self.create_client(SetHanoiTowerStations, '/set_hanoi_tower_stations')
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
            large = status_station_to_planner_station(response.large_pos)
            medium = status_station_to_planner_station(response.medium_pos)
            small = status_station_to_planner_station(response.small_pos)
            target = status_station_to_planner_station(response.target_pos)

            left_obstacle = getattr(response, 'left_obstacle', 1)
            right_obstacle = getattr(response, 'right_obstacle', 1)

            towers = None
            if large is not None and medium is not None and small is not None:
                towers = (large, medium, small)

            return towers, target, left_obstacle, right_obstacle
        except Exception as e:
            self.get_logger().error(f"Error querying status: {e}")
            return None, None, 1, 1

    def trigger_hanoi_planner(
        self,
        tower_stations: tuple[int, int, int],
        target_station: int,
        left_obstacle: int,
        right_obstacle: int,
    ) -> bool:
        """Call set_hanoi_tower_stations service to start path execution"""
        if not self.planner_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/set_hanoi_tower_stations planner service not available!")
            return False

        request = SetHanoiTowerStations.Request()
        request.tower_stations = list(tower_stations)
        request.obstacle = [bool(left_obstacle), bool(right_obstacle)]
        request.target_station = target_station

        self.get_logger().info(
            "Sending planning request: "
            f"tower_stations={tower_stations}, "
            f"target_station={target_station}, "
            f"obstacles={[bool(left_obstacle), bool(right_obstacle)]}"
        )
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
