#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from typing import Any, Dict, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class TurtleControlListener(Node):
    def __init__(self):
        super().__init__('turtle_control_listener')

        self.cmd_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.sub = self.create_subscription(String, 'gpt_reply_to_user', self._callback, 10)

        self.publish_hz = 20.0
        # TODO: 在這裡設定預設的 rotate_duration。
        # self.default_rotate_duration = 1.0  # seconds
        self.eps = 1e-6

        self.get_logger().info("TurtleControlListener started. Waiting for GPT JSON commands...")

    def _callback(self, msg: String):
        self.get_logger().info(f"Received: {msg.data}")

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON decode error: {e}")
            return

        try:
            commands = self._normalize_commands(payload)
            for cmd in commands:
                self._execute_command(cmd)
        except Exception as e:
            self.get_logger().error(f"Command handling error: {e}")

    def _normalize_commands(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
            return payload
        raise ValueError("JSON 必須是 object 或 object list。")
    
    def _get_float(self, data: Dict[str, Any], keys: List[str], default: float) -> float:
        cur: Any = data
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return float(default)
            cur = cur[key]
        try:
            return float(cur)
        except (TypeError, ValueError):
            return float(default)

    # def _execute_command(self, command: Dict[str, Any]):
    #     action = str(command.get("action", "move")).lower()
    #     # TODO:
    #     if action != "move":
    #         raise ValueError(f"Unknown action: {command.get('action')}")

    #     linear_x = self._get_float(command, ["linear_x"], 0.0)
    #     distance = self._get_float(command, ["distance"], 0.0)

    #     # TODO:
    #     angular_z = 0.0
    #     executed = False

    #     if action == "move":
    #         if abs(distance) > self.eps:
    #             if abs(linear_x) <= self.eps:
    #                 raise ValueError("未給定移動速度")

    #             move_duration = abs(distance) / abs(linear_x)
    #             self.get_logger().info(
    #                 f"Move: v={linear_x}, dist={distance}, duration={move_duration:.3f}s"
    #             )
    #             self._publish_for_duration(linear_x=linear_x, angular_z=0.0, duration=move_duration)
    #             self._stop()
    #             executed = True
    #     # TODO: 如果要支援 rotate，可以在 SYSTEM_PROMPT 中要求 GPT 輸出 angular_z 和 rotate_duration，然後在這裡解析並執行。
    #     # elif action == "rotate":

    #     if not executed:
    #         self.get_logger().warning(f"Skip empty command: {command}")

# 確保前面這幾行（約85行後）的縮排如下：
    def _execute_command(self, command: Dict[str, Any]):
        action = str(command.get("action", "move")).lower()
        
        if action not in ["move", "rotate"]:
            raise ValueError(f"Unknown action: {action}")

        executed = False

        # --- 處理移動 ---
        if action == "move":
            linear_x = self._get_float(command, ["linear_x"], 0.5)
            distance = self._get_float(command, ["distance"], 1.0)
            
            if abs(distance) > self.eps:
                if abs(linear_x) <= self.eps:
                    raise ValueError("未給定移動速度")

                move_duration = abs(distance) / abs(linear_x)
                self.get_logger().info(f"執行移動: {distance}m")
                self._publish_for_duration(linear_x=linear_x, angular_z=0.0, duration=move_duration)
                self._stop()
                executed = True

        # --- 處理旋轉 ---
        elif action == "rotate":
            angular_z = self._get_float(command, ["angular_z"], 1.0)
            duration = self._get_float(command, ["rotate_duration"], 1.57)
            
            self.get_logger().info(f"執行旋轉: {angular_z} rad/s")
            self._publish_for_duration(linear_x=0.0, angular_z=angular_z, duration=duration)
            self._stop()
            executed = True

        if not executed:
            self.get_logger().warning(f"Skip empty command: {command}")

    def _publish_for_duration(self, linear_x: float, angular_z: float, duration: float):
        dt = 1.0 / self.publish_hz
        end_t = time.time() + max(0.0, duration)

        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)

        while time.time() < end_t and rclpy.ok():
            self.cmd_pub.publish(twist)
            time.sleep(dt)

    def _stop(self):
        twist = Twist()
        self.cmd_pub.publish(twist)
        time.sleep(0.05)


def main():
    rclpy.init()
    node = TurtleControlListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
