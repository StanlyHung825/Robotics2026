import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import sys
import termios
import tty


KEY_BINDINGS = {
    'u': (0, 1),
    'j': (0, -1),
    'i': (1, 1),
    'k': (1, -1),
    'o': (2, 1),
    'l': (2, -1),
    'p': (3, 1),
    ';': (3, -1),
}


class KeyboardControl(Node):
    def __init__(self):
        super().__init__('keyboard_control')

        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            '/four_joints_position_controllers/commands',
            # 'real_robot_arm_joint',
            10
        )

        self.joint_positions = [0.0, 0.0, 0.0, 0.0]
        self.joint_limit = [1.2, 2.0, 1.67, 1.57079632]
        self.declare_parameter('step', 0.05)
        self.step = self.get_parameter('step').value

    def publish_positions(self):
        msg = Float64MultiArray()
        msg.data = self.joint_positions
        self.publisher_.publish(msg)

    def update_joint(self, joint_index, direction):
        limit = self.joint_limit[joint_index]
        position = self.joint_positions[joint_index] + direction * self.step
        self.joint_positions[joint_index] = max(-limit, min(limit, position))
        self.publish_positions()
        self.get_logger().info(
            'joint positions (rad): '
            f'{[round(position, 3) for position in self.joint_positions]}'
        )

    def handle_key(self, key):
        binding = KEY_BINDINGS.get(key.lower())
        if binding is None:
            return False

        joint_index, direction = binding
        self.update_joint(joint_index, direction)
        return True


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    try:
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def print_usage():
    print(
        '\nKeyboard control for four arm joints (radian commands)\n'
        '  joint1: U increase, J decrease\n'
        '  joint2: I increase, K decrease\n'
        '  joint3: O increase, L decrease\n'
        '  joint4: P increase, ; decrease\n'
        '  Q or Ctrl-C: quit\n'
    )


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControl()

    if not sys.stdin.isatty():
        node.get_logger().error('Keyboard control requires a terminal.')
        node.destroy_node()
        rclpy.shutdown()
        return

    settings = termios.tcgetattr(sys.stdin)
    print_usage()
    node.publish_positions()

    try:
        while rclpy.ok():
            key = get_key(settings)
            if key.lower() == 'q' or key == '\x03':
                break
            node.handle_key(key)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
