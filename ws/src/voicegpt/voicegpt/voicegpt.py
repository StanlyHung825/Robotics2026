#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import threading
import contextlib
import concurrent.futures
import speech_recognition as sr

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# # ===============================================================
# # TODO: 你可以修改 SYSTEM_PROMPT 來調整 GPT 的行為。
# SYSTEM_PROMPT = """You are a helpful ROS 2 assistant for controlling turtlesim.
# Given a user request (e.g., 'move forward 2m with speed 1m/s'), output ONLY plain JSON (no markdown).
# You may output either:
# (1) a single command object, or
# (2) a list of command objects to be executed sequentially.

# Each command object must follow:
# {
#   "action": "move",
#   "linear_x": <float>,
#   "distance": <float>,
# }

# Notes:
# - Always output exactly these 3 keys: action, linear_x, distance.
# - For action="move": set linear_x/distance to valid values that user gives, if user didn't mention them, set linear_x=0.5 and distance=1.0.
# - linear_x: forward > 0, backward < 0
# - distance: meters


# Output JSON only.
# """
# # ===============================================================原本程式

# ===============================================================
# TODO: 你可以修改 SYSTEM_PROMPT 來調整 GPT 的行為。
SYSTEM_PROMPT = """You are a helpful ROS 2 assistant for Hanoi Tower.
Given a user request, identify the final stacking position and the obstacle status. Output ONLY plain JSON (no markdown, no backticks, no markdown formatting).

Output JSON only in this format:
{
    "target": "A"|"B"|"C",
    "obstacles": "00"|"10"|"01"|"11"
}

Rules for "target":
- "target" must be exactly "A", "B", or "C".
- A=right, B=middle, C=left (from the perspective of a human facing the robot arm).
- If the user says left/middle/right/first/second/third/A/B/C/A區/B區/C區 in Chinese or English, map accordingly:
  - "left", "左", "左邊", "C", "C區", "third", "STATION 2" map to "C".
  - "right", "右", "右邊", "A", "A區", "first", "STATION 0" map to "A".
  - "middle", "中", "中間", "B", "B區", "second", "STATION 1" map to "B".
- Do NOT output integers (e.g. 0, 1, 2) or words (e.g. "middle"). Output only "A", "B", or "C".

Rules for "obstacles":
- This is a 2-digit binary string "XY" where X represents the left obstacle (between A and B) and Y represents the right obstacle (between B and C).
- '1' means the obstacle exists/is placed, '0' means the obstacle does not exist/is absent.
- Map the user instructions:
  - If they say neither obstacle exists, or say none, or "00", or say "不放/不要障礙物", set obstacles to "00".
  - If they say only the left obstacle exists (e.g., "只有左邊有障礙物", "左邊有障礙物", "左有右沒有", "左障礙物", or "10"), set obstacles to "10".
  - If they say only the right obstacle exists (e.g., "只有右邊有障礙物", "右邊有障礙物", "右有左沒有", "右障礙物", or "01"), set obstacles to "01".
  - If they say both obstacles exist, or say all obstacles, or "11", or "都有/都放障礙物", set obstacles to "11".
  - If the user does not specify anything about obstacles in their request, default to "11".

Output JSON only.
"""
# ===============================================================

class VoiceGPTNode(Node):
    def __init__(self):
        super().__init__('voice_gpt_node')

        # ROS2 Publisher
        self.pub = self.create_publisher(String, 'gpt_reply_to_user', 10)

        # Speech recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = float(os.environ.get("VOICEGPT_PAUSE_THRESHOLD", "1.0"))
        self.recognizer.non_speaking_duration = float(
            os.environ.get("VOICEGPT_NON_SPEAKING_DURATION", "0.6")
        )
        self.recognizer.phrase_threshold = float(os.environ.get("VOICEGPT_PHRASE_THRESHOLD", "0.3"))
        self.listen_timeout_sec = float(os.environ.get("VOICEGPT_LISTEN_TIMEOUT", "5"))
        self.phrase_time_limit_sec = float(os.environ.get("VOICEGPT_PHRASE_TIME_LIMIT", "15"))
        self.ambient_adjust_sec = float(os.environ.get("VOICEGPT_AMBIENT_ADJUST_SEC", "1.0"))
        # Optional mic index: use default device when unset.
        self.mic_device_index = self._parse_mic_index(os.environ.get("VOICEGPT_MIC_DEVICE"))

        # GitHub Models (Azure AI Inference) client
        endpoint = "https://models.github.ai/inference"
        model = "gpt-4o-mini"
        self.model = model
        self.gpt_timeout_sec = 20
        
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        # Remove any non-ASCII or hidden unicode characters (e.g. zero-width spaces from copy-paste)
        token = "".join(c for c in token if ord(c) < 128)
        
        if not token:
            raise RuntimeError("環境變數 GITHUB_TOKEN 未設定。請先 export GITHUB_TOKEN=your token")

        self.client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token),
        )

        # Start speech thread
        self._thread = threading.Thread(target=self._speech_loop, daemon=True)
        self._thread.start()

        self.get_logger().info("VoiceGPTNode started. Listening from microphone...")

    def _call_gpt(self, user_text: str) -> str:
        response = self.client.complete(
            messages=[
                SystemMessage(SYSTEM_PROMPT),
                UserMessage(user_text),
            ],
            model=self.model,
        )
        return response.choices[0].message.content

    def _call_gpt_with_timeout(self, user_text: str) -> str:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self._call_gpt, user_text)
        try:
            return future.result(timeout=self.gpt_timeout_sec)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise
        finally:
            # Do not block on hung network calls.
            pool.shutdown(wait=False, cancel_futures=True)

    def _speech_loop(self):
        mic = sr.Microphone(device_index=self.mic_device_index)

        while rclpy.ok():
            try:
                with self._silence_native_stderr(), mic as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=self.ambient_adjust_sec)
                    self.get_logger().info("請開始說話...")
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.listen_timeout_sec,
                        phrase_time_limit=self.phrase_time_limit_sec,
                    )
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                self.get_logger().error(f"Microphone/listen error: {e}")
                continue

            try:
                text = self.recognizer.recognize_google(audio, language="zh-CN")
                self.get_logger().info(f"Recognized: {text}")

                self.get_logger().info("Calling GPT...")
                gpt_reply = self._call_gpt_with_timeout(text)
                self.get_logger().info(f"GPT reply: {gpt_reply}")

                # Clean up any potential markdown formatting block wrappers
                cleaned_reply = gpt_reply.strip()
                if "{" in cleaned_reply and "}" in cleaned_reply:
                    start_idx = cleaned_reply.find("{")
                    end_idx = cleaned_reply.rfind("}")
                    if end_idx > start_idx:
                        cleaned_reply = cleaned_reply[start_idx:end_idx+1]

                msg = String()
                msg.data = cleaned_reply
                self.pub.publish(msg)

            except sr.UnknownValueError:
                self.get_logger().warning("語音無法辨識")
            except sr.RequestError as e:
                self.get_logger().error(f"SpeechRecognition API error: {e}")
            except concurrent.futures.TimeoutError:
                self.get_logger().error(
                    f"GPT request timed out after {self.gpt_timeout_sec}s. Check network/token."
                )
            except Exception as e:
                self.get_logger().error(f"GPT call/publish error: {e}")

    def _parse_mic_index(self, raw_value: str):
        if not raw_value:
            return None
        try:
            return int(raw_value)
        except ValueError:
            self.get_logger().warning(
                f"Invalid VOICEGPT_MIC_DEVICE='{raw_value}', fallback to default microphone."
            )
            return None

    @contextlib.contextmanager
    def _silence_native_stderr(self):
        # ALSA/PortAudio writes directly to fd=2; redirect fd to /dev/null in this block.
        stderr_fd = 2
        saved_fd = os.dup(stderr_fd)
        try:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                os.dup2(devnull.fileno(), stderr_fd)
                yield
        finally:
            os.dup2(saved_fd, stderr_fd)
            os.close(saved_fd)


def main():
    rclpy.init()
    node = VoiceGPTNode()
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
