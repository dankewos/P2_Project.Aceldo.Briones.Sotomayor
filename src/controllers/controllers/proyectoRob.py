import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String


class GapFollowerRace(Node):
    def __init__(self):
        super().__init__('gap_follower_race')

        # -------- parámetros configurables --------
        self.declare_parameter('v_max', 8.2)
        self.declare_parameter('v_min', 3.5)
        self.declare_parameter('curve_exponent', 1.2)
        self.declare_parameter('lidar_cap', 3.5)
        self.declare_parameter('bubble_radius_m', 0.25)
        self.declare_parameter('max_steer_deg', 20.0)
        self.declare_parameter('far_bias', 0.8)
        self.declare_parameter('steer_smooth', 0.9)

        self.declare_parameter('finish_x', 0.0)
        self.declare_parameter('finish_y', 0.0)
        self.declare_parameter('finish_y_band', 5.0)
        self.declare_parameter('rearm_dist', 0.2)

        # -------- cache de parámetros --------
        self.v_max = float(self.get_parameter('v_max').value)
        self.v_min = float(self.get_parameter('v_min').value)
        self.curve_exp = float(self.get_parameter('curve_exponent').value)
        self.lidar_cap = float(self.get_parameter('lidar_cap').value)
        self.bubble_r = float(self.get_parameter('bubble_radius_m').value)
        self.max_steer = math.radians(float(self.get_parameter('max_steer_deg').value))
        self.far_bias = float(self.get_parameter('far_bias').value)
        self.steer_alpha = float(self.get_parameter('steer_smooth').value)

        self.finish_x = float(self.get_parameter('finish_x').value)
        self.finish_y = float(self.get_parameter('finish_y').value)
        self.finish_band = float(self.get_parameter('finish_y_band').value)
        self.rearm_dist = float(self.get_parameter('rearm_dist').value)

        # -------- estado --------
        self._last_pose = None
        self._arm_for_lap = True
        self._last_steer = 0.0

        # -------- pubs/subs --------
        qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 5)
        self.lap_pub = self.create_publisher(String, '/lap_trigger', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.on_scan, qos_sensor)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.on_odom, 10)

    # -------------------- ODOM --------------------
    def on_odom(self, msg: Odometry):
        pose = msg.pose.pose

        if self._last_pose is None:
            self._last_pose = pose
            return

        last_x = self._last_pose.position.x
        x = pose.position.x
        y = pose.position.y

        crossed = (last_x < self.finish_x) and (x >= self.finish_x)
        on_straight = abs(y - self.finish_y) < self.finish_band

        if crossed and on_straight and self._arm_for_lap:
            self.lap_pub.publish(String(data='Nueva vuelta'))
            self._arm_for_lap = False

        if not self._arm_for_lap:
            # rearmar cuando nos alejamos suficiente
            if abs(x - self.finish_x) > self.rearm_dist:
                self._arm_for_lap = True

        self._last_pose = pose

    # -------------------- LIDAR --------------------
    def on_scan(self, scan: LaserScan):
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        # limpiar valores no válidos y limitar rango
        ranges[~np.isfinite(ranges)] = self.lidar_cap
        np.clip(ranges, 0.0, self.lidar_cap, out=ranges)

        # 1) burbuja alrededor del obstáculo más cercano
        closest_idx = int(np.argmin(ranges))
        closest_dist = float(ranges[closest_idx])
        if closest_dist > 0.1:
            span = 2.0 * math.atan2(self.bubble_r, closest_dist)
            rad_idx = max(1, int(span / scan.angle_increment))
        else:
            rad_idx = 30  # fallback seguro

        a = max(0, closest_idx - rad_idx)
        b = min(len(ranges) - 1, closest_idx + rad_idx)
        ranges[a:b + 1] = 0.0

        # 2) mayor gap de valores > 0
        best_len = 0
        best_a = 0
        cur_len = 0
        cur_a = 0
        for i, r in enumerate(ranges):
            if r > 0.1:
                if cur_len == 0:
                    cur_a = i
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_len = cur_len
                    best_a = cur_a
                cur_len = 0
        if cur_len > best_len:
            best_len = cur_len
            best_a = cur_a
        best_b = best_a + best_len - 1

        drive = AckermannDriveStamped()

        if best_len > 0:
            gap = ranges[best_a:best_b + 1]
            far_rel = int(np.argmax(gap))
            far_abs = best_a + far_rel
            gap_center = (best_a + best_b) // 2

            target_idx = int(self.far_bias * far_abs + (1.0 - self.far_bias) * gap_center)
            target_ang = scan.angle_min + target_idx * scan.angle_increment

            # suavizado de dirección
            steer = self.steer_alpha * self._last_steer + (1.0 - self.steer_alpha) * target_ang
            steer = float(np.clip(steer, -self.max_steer, self.max_steer))
            self._last_steer = steer

            # reducción de velocidad según curvatura
            curve = min(abs(target_ang) / self.max_steer, 1.0)
            scale = (1.0 - curve) ** self.curve_exp
            speed = self.v_min + (self.v_max - self.v_min) * scale

            drive.drive.steering_angle = steer
            drive.drive.speed = float(speed)
        else:
            # sin gap: detenerse
            self._last_steer = 0.0
            drive.drive.steering_angle = 0.0
            drive.drive.speed = 0.0

        self.drive_pub.publish(drive)


def main(args=None):
    rclpy.init(args=args)
    node = GapFollowerRace()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
