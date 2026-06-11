import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

# 匯入剛剛建立的 Service 格式 
# (請務必將 hanoi_interfaces 替換成你們實際存放 srv 檔案的 package 名稱)
from hanoi_interface.srv import GetHanoiStatus

class HanoiVisionNode(Node):
    def __init__(self):
        super().__init__('hanoi_vision_node')
        self.bridge = CvBridge()
        
        # 1. 訂閱相機畫面
        # 請根據你們實際使用的相機 Topic 修改，例如 '/usb_cam/image_raw'
        self.subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        
        # 2. 建立 Service Server，提供座標給隊友的 MoveIt 節點
        self.srv = self.create_service(
            GetHanoiStatus, 'get_hanoi_positions', self.get_pos_callback)

        # 2.1 訂閱語音輸入結果，取得最終疊放位置 (1=A, 2=B, 3=C, 0=未知)
        self.voice_subscription = self.create_subscription(
            String, 'gpt_reply_to_user', self.voice_callback, 10)
        
        # 3. ArUco 字典與參數設定 
        # 假設你們列印的是 4x4 的字典，可依實際情況調整為 5x5 等
        # 注意: 若 OpenCV 版本較新 (4.7+)，可能需改用 cv2.aruco.ArucoDetector
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # 定義大、中、小河內塔對應的 ArUco ID
        # 請根據你們實際貼在塔上的 ID 進行修改
        self.ID_LARGE = 0
        self.ID_MEDIUM = 1
        self.ID_SMALL = 2
        
        # 儲存最新的世界座標 [x, y]
        self.world_positions = {
            self.ID_LARGE: [0.0, 0.0],
            self.ID_MEDIUM: [0.0, 0.0],
            self.ID_SMALL: [0.0, 0.0]
        }
        # 儲存最新的像素中心 (u, v) 與時間戳，供左中右排序
        self.pixel_centers = {
            self.ID_LARGE: (0, 0),
            self.ID_MEDIUM: (0, 0),
            self.ID_SMALL: (0, 0)
        }
        self.last_seen_ns = {
            self.ID_LARGE: None,
            self.ID_MEDIUM: None,
            self.ID_SMALL: None
        }
        self.target_pos = 0
        self.left_obstacle = 1
        self.right_obstacle = 1
        self.get_logger().info('河內塔視覺節點 (Hanoi Vision Node) 已啟動，等待影像輸入...')

    def image_callback(self, msg):
        """持續接收影像並進行 ArUco 辨識"""
        try:
            # 將 ROS Image 轉換為 OpenCV 格式 (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"影像轉換失敗: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 偵測 ArUco Tags
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None:
            # 將辨識到的邊界畫在畫面上 (方便 Debug 與校正)
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            
            for i in range(len(ids)):
                tag_id = ids[i][0]
                # 只處理我們關心的河內塔 ID
                if tag_id in [self.ID_LARGE, self.ID_MEDIUM, self.ID_SMALL]:
                    # 計算 ArUco Tag 的中心點像素座標 (u, v)
                    c = corners[i][0]
                    center_u = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
                    center_v = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)
                    
                    # 畫個紅點標示中心
                    cv2.circle(cv_image, (center_u, center_v), 5, (0, 0, 255), -1)
                    
                    # 在畫面上標示對應的塔名稱
                    label = "Unknown"
                    if tag_id == self.ID_LARGE: label = "Large"
                    elif tag_id == self.ID_MEDIUM: label = "Medium"
                    elif tag_id == self.ID_SMALL: label = "Small"
                    cv2.putText(cv_image, label, (center_u + 10, center_v - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 轉換：像素座標 (u, v) -> 機械臂世界座標 (x, y)
                    world_x, world_y = self.pixel_to_world(center_u, center_v)
                    self.world_positions[tag_id] = [world_x, world_y]
                    self.pixel_centers[tag_id] = (center_u, center_v)
                    self.last_seen_ns[tag_id] = self.get_clock().now().nanoseconds

        # 顯示畫面供開發者確認辨識狀況
        cv2.imshow("Hanoi Vision Tracking", cv_image)
        cv2.waitKey(1)

    def pixel_to_world(self, u, v):
        """
        【重要】影像像素轉實際座標的函數。
        這是一個佔位符範例。在實際比賽環境中，強烈建議使用 cv2.getPerspectiveTransform 
        計算單應性矩陣(Homography Matrix)，並在這裡進行轉換，以抵銷相機傾斜帶來的誤差。
        """
        # 假設畫面解析度為 640x480，中心是 (320, 240)
        # 這裡僅作簡單的線性映射示範，你需要根據實機量測進行校正
        x = (v - 240) * 0.001 + 0.25  # 將 y 軸像素映射到世界 x 軸 (範例)
        y = (u - 320) * 0.001         # 將 x 軸像素映射到世界 y 軸 (範例)
        return float(x), float(y)

    def voice_callback(self, msg: String):
        if not msg.data:
            return
        raw = msg.data.strip()
        self.get_logger().info(f"收到語音回應: {raw}")
        
        is_json = False
        target_val = None
        obstacles_val = None
        
        # 嘗試從字串中萃取出 JSON 部分 (即使有 markdown 區塊包住也沒關係)
        json_str = None
        if "{" in raw and "}" in raw:
            start_idx = raw.find("{")
            end_idx = raw.rfind("}")
            if end_idx > start_idx:
                json_str = raw[start_idx:end_idx+1]
                
        if json_str:
            try:
                payload = json.loads(json_str)
                # 支援多種 target 欄位名稱
                for key in ["target", "target_pos", "target_station", "target_station_idx"]:
                    if key in payload:
                        target_val = payload[key]
                        break
                obstacles_val = payload.get("obstacles")
                is_json = True
            except Exception as e:
                self.get_logger().error(f"JSON 解析失敗: {e}")

        # 解析 target
        if target_val is not None:
            parsed_target = self._parse_target(target_val)
            if parsed_target is not None:
                self.target_pos = parsed_target
        elif not is_json:
            # 如果不是 JSON 格式，則把整個字串當作目標來解析
            parsed_target = self._parse_target(raw)
            if parsed_target is not None:
                self.target_pos = parsed_target

        # 解析 obstacles (00, 10, 01, 11)
        if obstacles_val is not None:
            obs_str = str(obstacles_val).strip()
            if len(obs_str) == 2 and all(c in "01" for c in obs_str):
                self.left_obstacle = int(obs_str[0])
                self.right_obstacle = int(obs_str[1])
                self.get_logger().info(f"Obstacles set to: Left={self.left_obstacle}, Right={self.right_obstacle}")
            else:
                self.get_logger().warning(f"不合法的障礙物格式: {obs_str}")
        elif not is_json:
            # 備用方案：如果不是 JSON 格式，用正則表達式嘗試尋找 00/10/01/11
            import re
            obs_match = re.search(r'\b([0-1]{2})\b', raw)
            if obs_match:
                obs_str = obs_match.group(1)
                self.left_obstacle = int(obs_str[0])
                self.right_obstacle = int(obs_str[1])
                self.get_logger().info(f"從純文字中讀取障礙物: Left={self.left_obstacle}, Right={self.right_obstacle}")

        self.get_logger().info(f'Voice target set to: {self.target_pos}')

    def _parse_target(self, val):
        if val is None:
            return None

        # 1. 處理整數數值 (0-based 或 1-based)
        if isinstance(val, int):
            if val == 0: return 1  # 0-based A -> 1
            if val == 1: return 2  # 0-based B -> 2
            if val == 2: return 3  # 0-based C -> 3
            if val == 3: return 3  # 1-based C -> 3
            return None

        # 2. 處理字串
        if isinstance(val, str):
            raw = val.strip()
            if not raw:
                return None

            # 若字串本身是個 JSON，嘗試解開
            if raw.startswith("{") or "{" in raw:
                try:
                    start_idx = raw.find("{")
                    end_idx = raw.rfind("}")
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        payload = json.loads(raw[start_idx:end_idx+1])
                        for key in ["target", "target_pos", "target_station", "target_station_idx"]:
                            if key in payload:
                                return self._parse_target(payload[key])
                except Exception:
                    pass

            upper = raw.upper()

            # (A) 精確的單字或字元對比 (避免 partial match)
            if upper in ("A", "RIGHT", "R", "TOWER3", "TOWER 3", "STATION 0", "STATION0", "A區", "A ZONE"):
                return 1
            if upper in ("B", "MIDDLE", "M", "TOWER2", "TOWER 2", "STATION 1", "STATION1", "B區", "B ZONE"):
                return 2
            if upper in ("C", "LEFT", "L", "TOWER1", "TOWER 1", "STATION 2", "STATION2", "C區", "C ZONE"):
                return 3

            # (B) 數字字串 (如 "0", "1", "2")
            if upper == "0": return 1
            if upper == "1": return 2
            if upper == "2": return 3
            if upper == "3": return 3

            # (C) 清理 JSON 鍵名以防干擾 (排除 TARGET / OBSTACLES)
            import re
            cleaned = re.sub(r'TARGET(_POS|_STATION|_STATION_IDX)?|OBSTACLES|[{}":,\d\s\'-]', '', upper)

            # 在清理後的字串中進行模糊匹配，且優先比對中文詞彙
            if "右" in raw or "A" in cleaned or "RIGHT" in cleaned:
                return 1
            if "中" in raw or "B" in cleaned or "MIDDLE" in cleaned:
                return 2
            if "左" in raw or "C" in cleaned or "LEFT" in cleaned:
                return 3

        return None

    def get_pos_callback(self, request, response):
        """當隊友的節點呼叫 Service 時，回傳大中小河內塔的左右位置(1/2/3/0)"""
        self.get_logger().info('收到請求！正在回傳大中小河內塔的左右位置...')

        now_ns = self.get_clock().now().nanoseconds
        max_age_ns = int(2.0 * 1e9)

        # 收集近期可用的標記，依像素 u 從左到右排序
        visible = []
        for tag_id in [self.ID_LARGE, self.ID_MEDIUM, self.ID_SMALL]:
            seen_ns = self.last_seen_ns[tag_id]
            if seen_ns is None:
                continue
            if now_ns - seen_ns > max_age_ns:
                continue
            center_u, _ = self.pixel_centers[tag_id]
            visible.append((center_u, tag_id))

        visible.sort(key=lambda item: item[0], reverse=True)

        # 預設為 0 (未找到)
        response.large_pos = 0
        response.medium_pos = 0
        response.small_pos = 0
        response.target_pos = self.target_pos
        response.left_obstacle = self.left_obstacle
        response.right_obstacle = self.right_obstacle

        # 依序指定 1/2/3 對應左/中/右
        for idx, (_, tag_id) in enumerate(visible[:3], start=1):
            if tag_id == self.ID_LARGE:
                response.large_pos = idx
            elif tag_id == self.ID_MEDIUM:
                response.medium_pos = idx
            elif tag_id == self.ID_SMALL:
                response.small_pos = idx

        self.get_logger().info(
            'Large: %d, Medium: %d, Small: %d, Target: %d, LeftObstacle: %d, RightObstacle: %d'
            % (response.large_pos, response.medium_pos, response.small_pos, response.target_pos, response.left_obstacle, response.right_obstacle))

        return response

def main(args=None):
    rclpy.init(args=args)
    node = HanoiVisionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()