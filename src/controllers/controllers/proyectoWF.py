#!/usr/bin/env python3
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

#Creacion de la clase que realizara el control pid (realmente pd)
class PID:
    def __init__(self, kp, ki, kd, out_min=-float('inf'), out_max=float('inf')):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.e_prev = 0.0
        self.i_term = 0.0
        self.t_prev = None

    def reset(self):
        self.e_prev = 0.0
        self.i_term = 0.0
        self.t_prev = None

    def update(self, e, t_now):
        if self.t_prev is None:
            dt = 0.0
        else:
            dt = max(1e-3, t_now - self.t_prev)
        self.t_prev = t_now

        self.i_term += e * dt
        d = (e - self.e_prev) / dt if dt > 0 else 0.0
        self.e_prev = e

        u = self.kp * e + self.ki * self.i_term + self.kd * d
        return float(np.clip(u, self.out_min, self.out_max))

#Clase que aplicara el algoritmo que sigue la pared
class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_following_node')

        # parámetros
        # lado a seguir: 'left' o 'right'
        self.declare_parameter('side', 'right')
        self.side = self.get_parameter('side').get_parameter_value().string_value or 'right'

        self.declare_parameter('desired_dist', 0.8)      # distancia objetivo al muro [m]
        self.declare_parameter('theta_deg', 50.0)        # ángulo base para mediciones
        self.declare_parameter('alpha_deg', 10.0)        # separación angular entre rayos
        self.declare_parameter('lookahead', 0.7)         # proyección hacia adelante [m]

        self.declare_parameter('max_steer_deg', 30.0)
        self.declare_parameter('v_min', 1.2)
        self.declare_parameter('v_base', 5.0)
        self.declare_parameter('v_max', 10.0)

        self.desired_dist = float(self.get_parameter('desired_dist').value)
        self.th_deg = float(self.get_parameter('theta_deg').value)
        self.al_deg = float(self.get_parameter('alpha_deg').value)
        self.L = float(self.get_parameter('lookahead').value)

        self.max_steer = math.radians(float(self.get_parameter('max_steer_deg').value))
        self.v_min = float(self.get_parameter('v_min').value)
        self.v_base = float(self.get_parameter('v_base').value)
        self.v_max = float(self.get_parameter('v_max').value)

        # filtros
        self.steer_alpha = 0.85
        self.speed_alpha = 0.7
        self.steer_filt = 0.0
        self.speed_filt = self.v_base

        # PID de dirección (error de distancia lateral proyectada)
        self.pid = PID(kp=1.2, ki=0.0, kd=0.10,
                       out_min=-self.max_steer, out_max=self.max_steer)

        # vuelta (igual que tu lógica)
        self.lap_start_time = time.time()
        self.lap_count = 0
        self.prev_in_start_zone = False
        self.lap_times = []
        self.has_left_start_zone = False

        # pubs/subs
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.on_scan, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.on_odom, 10)

        self.scan_meta = None  # (angle_min, angle_inc, n)
        self.max_lidar_range = 9.0
        self.get_logger().info(f"WallFollowing: siguiendo muro {self.side}")

    # Procesamiento de LIDAR
    def _preprocess(self, ranges):
        arr = np.asarray(ranges, dtype=np.float32)
        arr[~np.isfinite(arr) | (arr <= 0.0)] = self.max_lidar_range
        return np.clip(arr, 0.05, self.max_lidar_range)

    def _idx(self, ang, amin, ainc, n):
        j = int(round((ang - amin) / ainc))
        return max(0, min(n - 1, j))

    def _range_at(self, scan, angle_rad, win=2):
        n = len(scan.ranges)
        i = self._idx(angle_rad, scan.angle_min, scan.angle_increment, n)
        a = max(0, i - win)
        b = min(n, i + win + 1)
        val = np.array(scan.ranges[a:b], dtype=np.float32)
        val[~np.isfinite(val) | (val <= 0.0)] = self.max_lidar_range
        return float(np.median(np.clip(val, 0.05, self.max_lidar_range)))

    # Llama la informacion del LIDAR
    def on_scan(self, scan: LaserScan):
        t = self.get_clock().now().nanoseconds * 1e-9

        # elegir signos según lado
        sign = -1.0 if self.side.lower().startswith('right') else +1.0
        theta = math.radians(self.th_deg) * sign
        theta_a = math.radians(self.th_deg + self.al_deg) * sign

        # tomas a dos ángulos (método clásico de wall-following)
        b = self._range_at(scan, theta_a)     # más "atrás"
        a = self._range_at(scan, theta)       # más "adelante"

        # estimar ángulo del muro y distancia actual
        # ref: alpha = atan((a*cos(phi) - b) / (a*sin(phi)))
        phi = math.radians(self.al_deg)
        alpha = math.atan2((a * math.cos(phi) - b), (a * math.sin(phi)))
        dist_now = b * math.cos(alpha)

        # distancia proyectada a lookahead L
        dist_proj = dist_now + self.L * math.sin(alpha)

        # error: queremos que dist_proj == desired_dist
        e = self.desired_dist - dist_proj
        steer_cmd = self.pid.update(e, t)

        # suavizar y limitar
        steer_cmd = self.steer_alpha * self.steer_filt + (1.0 - self.steer_alpha) * steer_cmd
        steer_cmd = float(np.clip(steer_cmd, -self.max_steer, self.max_steer))
        self.steer_filt = steer_cmd

        # velocidad según giro y espacio frontal
        d_front = self._range_at(scan, 0.0, win=4)
        ang_factor = 1.0 - 0.6 * (abs(steer_cmd) / self.max_steer)
        free_factor = np.interp(d_front, [0.8, 3.0, 6.0], [0.3, 1.0, 1.2])
        v_target = np.clip(self.v_base * ang_factor * free_factor, self.v_min, self.v_max)
        v_cmd = self.speed_alpha * self.speed_filt + (1.0 - self.speed_alpha) * v_target
        self.speed_filt = v_cmd

        # publicar
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer_cmd
        msg.drive.speed = float(v_cmd)
        self.drive_pub.publish(msg)

    # Llamar la informacion de odometria del carrito
    def on_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        in_start_zone = (abs(x) < 2.0 and abs(y) < 2.0)

        if in_start_zone and not self.prev_in_start_zone and self.has_left_start_zone:
            t1 = time.time()
            lap = t1 - self.lap_start_time
            self.lap_times.append(lap)
            self.lap_count += 1
            self.get_logger().info(f"🏁 Vuelta {self.lap_count} en {lap:.2f} s")
            self.lap_start_time = t1
 
        if not in_start_zone:
            self.has_left_start_zone = True
        self.prev_in_start_zone = in_start_zone

#Defina al main que lama a la clase/nodo del algoritmo WF
def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped cleanly')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
