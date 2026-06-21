from math import acos, atan2, cos, hypot, pi, sin

from myrobot.hanoi_model import HOME_POSITION, JOINT_LIMITS, LINK_LENGTH


class ArmKinematics:
    def __init__(
        self,
        *,
        link_lengths: tuple[float, float, float, float, float, float] = LINK_LENGTH,
        joint_limits: tuple[tuple[float, float], ...] = JOINT_LIMITS,
        home_position: tuple[float, float, float, float] = HOME_POSITION,
    ) -> None:
        self.link_lengths = link_lengths
        self.joint_limits = joint_limits
        self.home_position = home_position

    def solve(
        self,
        x: float,
        y: float,
        z: float,
        pitch: float = pi / 2,
    ) -> tuple[float, float, float, float]:
        """Analytic IK for the 4-DOF arm described by myrobot.urdf."""
        base_height = self.link_lengths[0] + self.link_lengths[1]
        upper_arm = self.link_lengths[2]
        forearm = self.link_lengths[3]
        tool_z = self.link_lengths[4]
        tool_x = self.link_lengths[5]

        joint1 = atan2(y, x) if hypot(x, y) > 1e-12 else 0.0
        if not self._inside_limit(joint1, 0):
            raise ValueError(f"joint1 angle {joint1:.3f} rad exceeds the URDF limit")

        radius = hypot(x, y)
        tool_radius = tool_x * cos(pitch) + tool_z * sin(pitch)
        tool_height = -tool_x * sin(pitch) + tool_z * cos(pitch)
        wrist_radius = radius - tool_radius
        wrist_height = z - base_height - tool_height

        wrist_distance = hypot(wrist_radius, wrist_height)
        min_reach = abs(upper_arm - forearm)
        max_reach = upper_arm + forearm
        if wrist_distance < min_reach - 1e-9 or wrist_distance > max_reach + 1e-9:
            raise ValueError(
                "target is outside the reachable workspace "
                f"(wrist distance {wrist_distance:.3f} m, reachable "
                f"{min_reach:.3f} m to {max_reach:.3f} m)"
            )

        cos_joint3 = (
            wrist_radius**2
            + wrist_height**2
            - upper_arm**2
            - forearm**2
        ) / (2.0 * upper_arm * forearm)
        cos_joint3 = max(-1.0, min(1.0, cos_joint3))

        candidates = []
        for joint3 in (acos(cos_joint3), -acos(cos_joint3)):
            joint2 = atan2(wrist_radius, wrist_height) - atan2(
                forearm * sin(joint3),
                upper_arm + forearm * cos(joint3),
            )
            joint4 = pitch - joint2 - joint3
            joint_angles = (joint1, joint2, joint3, joint4)

            if all(
                self._inside_limit(angle, index)
                for index, angle in enumerate(joint_angles)
            ):
                score = sum(
                    abs(angle - home)
                    for angle, home in zip(joint_angles, self.home_position)
                )
                candidates.append((score, joint_angles))

        if not candidates:
            raise ValueError("target is reachable geometrically, but violates joint limits")

        _, solution = min(candidates, key=lambda item: item[0])
        return tuple(
            self._clamp_to_limit(angle, index)
            for index, angle in enumerate(solution)
        )

    def _inside_limit(
        self,
        angle: float,
        joint_index: int,
        tolerance: float = 1e-9,
    ) -> bool:
        lower, upper = self.joint_limits[joint_index]
        return lower - tolerance <= angle <= upper + tolerance

    def _clamp_to_limit(self, angle: float, joint_index: int) -> float:
        lower, upper = self.joint_limits[joint_index]
        return max(lower, min(upper, angle))
