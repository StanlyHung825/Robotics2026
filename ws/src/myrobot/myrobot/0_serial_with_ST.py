##real arm (Axis2 and Axis3 relationship is cared)
import struct
import threading
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from serial import Serial
from std_msgs.msg import Float64MultiArray

"""
Belt transmition of real arm:

    joint3 is influenced by joint2 by 0.7 
    delta(j3)= delta(j3) +0.7*delta(j2)

"""

COM_NAME = "/dev/ttyACM0"
BAUTRATE = 230400


@dataclass(slots=True, init=False)
class StmJointAnglePublisher:
    _serial: Serial
    _read_thread: threading.Thread
    _keep_reading: bool

    def __init__(self):
        try:
            self._serial = Serial(
                port=COM_NAME,
                baudrate=BAUTRATE,
                timeout=0.5,
            )
            print("ST Connect OK!")

        except Exception as e:
            raise Exception(f"ST Connect Error! {e}")

        self._keep_reading = True
        self._read_thread = threading.Thread(target=self._loop_readln, daemon=True)
        self._read_thread.start()

    def _loop_readln(self) -> None:
        while self._keep_reading:
            data: bytes = self._serial.readline()
            if data:
                print(f"ST print:\n{data}")

    def pub(self, cmd_jp: list[float]) -> None:
        """
        '<' means little-endian(Byte order)
        'i' means int
        refer to : https://docs.python.org/3/library/struct.html
        """
        if len(cmd_jp) != 4:
            raise ValueError("Command must have 4 elements")

        int_cmds = [int(x * 1000) for x in cmd_jp[:4]]
        packed_data = struct.pack("<4i", *int_cmds)
        str_cmd = b"s" + packed_data + b"e"

        self._serial.write(str_cmd)

    def close(self):
        self._keep_reading = False
        if self._read_thread.is_alive():
            self._read_thread.join(timeout=1.0)
        self._serial.close()
        print("ST Close Successfully")


class STM32Interface(Node):
    def __init__(self):
        super().__init__("stm32_interface")

        self.stm_joint_angle_publisher = StmJointAnglePublisher()

        self.subscription = self.create_subscription(
            Float64MultiArray, "/real_robot_arm_joint", self.callback, 10
        )

        self.get_logger().info("STM32Interface node has been started")

    def callback(self, msg: Float64MultiArray):
        cmd = list(msg.data)
        cmd[2] += cmd[1] * 0.7  # belt transmition of real arm

        self.stm_joint_angle_publisher.pub(cmd)


def main(args=None):
    rclpy.init(args=args)

    node = STM32Interface()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stm_joint_angle_publisher.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
