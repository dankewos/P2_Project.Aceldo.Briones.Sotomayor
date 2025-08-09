# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# import numpy as np
# import time
# from sensor_msgs.msg import LaserScan
# from ackermann_msgs.msg import AckermannDriveStamped
# from nav_msgs.msg import Odometry


# class ReactiveFollowGap(Node):
#     def __init__(self):
#         super().__init__('reactive_node')

#         # Topics
#         lidarscan_topic = '/scan'
#         drive_topic = '/drive'
#         odom_topic = '/ego_racecar/odom'

#         # Parámetros del algoritmo
#         self.base_speed = 4.8
#         self.max_speed = 11.0
#         self.min_speed = 1.5
#         self.max_lidar_range = 9.0
#         self.bubble_radius_m = 1.1
#         self.min_clearance = 1.6
#         self.max_steering_angle = np.radians(30)
#         self.gap_threshold = 0.5
#         self.center_bias = 0.5
#         self.speed_filter = self.base_speed
#         self.speed_filter_alpha = 0.7
#         self.steering_filter = 0.0
#         self.steering_filter_alpha = 0.8
#         self.scan_initialized = False
#         self.scan_count = 0

#         # Control de vueltas
#         self.lap_start_time = time.time()
#         self.lap_count = 0
#         self.prev_in_start_zone = False
#         self.lap_times = []
#         self.has_left_start_zone = False 

#         # Publishers & Subscribers
#         self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
#         self.scan_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
#         self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

#         self.get_logger().info("ReactiveFollowGap node initialized.")

#     def preprocess_lidar(self, ranges):
#         proc = np.array(ranges, dtype=np.float32)
#         invalid_mask = (proc == 0) | np.isinf(proc) | np.isnan(proc)
#         proc[invalid_mask] = self.max_lidar_range
#         proc = np.clip(proc, 0.1, self.max_lidar_range)
#         kernel = np.ones(3) / 3
#         return np.convolve(proc, kernel, mode='same')

#     def create_safety_bubble(self, ranges, closest_idx):
#         bubble_ranges = np.copy(ranges)
#         angle_increment = 2 * np.pi / len(ranges)
#         bubble_radius_idx = int(self.bubble_radius_m / (ranges[closest_idx] * angle_increment))
#         bubble_radius_idx = max(5, min(bubble_radius_idx, 20))
#         start_idx = max(0, closest_idx - bubble_radius_idx)
#         end_idx = min(len(ranges) - 1, closest_idx + bubble_radius_idx)
#         bubble_ranges[start_idx:end_idx + 1] = 0.0
#         return bubble_ranges

#     def find_gaps(self, ranges):
#         free_mask = ranges >= self.min_clearance
#         gaps = []
#         start_idx = None
#         for i, is_free in enumerate(free_mask):
#             if is_free and start_idx is None:
#                 start_idx = i
#             elif not is_free and start_idx is not None:
#                 gap_width_rad = (i - start_idx) * (2 * np.pi / len(ranges))
#                 gap_width_m = gap_width_rad * np.mean(ranges[start_idx:i])
#                 if gap_width_m >= self.gap_threshold:
#                     gaps.append((start_idx, i - 1))
#                 start_idx = None
#         if start_idx is not None:
#             gap_width_rad = (len(free_mask) - start_idx) * (2 * np.pi / len(ranges))
#             gap_width_m = gap_width_rad * np.mean(ranges[start_idx:])
#             if gap_width_m >= self.gap_threshold:
#                 gaps.append((start_idx, len(free_mask) - 1))
#         return gaps

#     def select_best_gap(self, gaps, ranges):
#         if not gaps:
#             return None
#         best_gap = None
#         best_score = -1
#         center_idx = len(ranges) // 2
#         for start_idx, end_idx in gaps:
#             gap_ranges = ranges[start_idx:end_idx + 1]
#             avg_distance = np.mean(gap_ranges)
#             max_distance = np.max(gap_ranges)
#             gap_width = end_idx - start_idx
#             gap_center = (start_idx + end_idx) // 2
#             center_distance = abs(gap_center - center_idx)
#             distance_score = min(avg_distance / 5.0, 1.0)
#             width_score = min(gap_width / 50.0, 1.0)
#             center_score = 1.0 - (center_distance / center_idx) * self.center_bias
#             total_score = (distance_score * 0.4 + width_score * 0.3 + center_score * 0.3)
#             if total_score > best_score:
#                 best_score = total_score
#                 best_gap = (start_idx, end_idx)
#         return best_gap

#     def get_target_point(self, start_idx, end_idx, ranges):
#         if start_idx is None or end_idx is None:
#             return None
#         gap_ranges = ranges[start_idx:end_idx + 1]
#         max_dist_idx = np.argmax(gap_ranges)
#         target_idx = start_idx + max_dist_idx
#         gap_center = (start_idx + end_idx) // 2
#         target_idx = int(0.7 * target_idx + 0.3 * gap_center)
#         return target_idx

#     def calculate_steering_angle(self, target_idx, scan_data):
#         if target_idx is None:
#             return 0.0
#         target_angle = scan_data.angle_min + target_idx * scan_data.angle_increment
#         center_angle = scan_data.angle_min + (len(scan_data.ranges) // 2) * scan_data.angle_increment
#         steering_angle = (target_angle - center_angle) * 0.8
#         steering_angle = (self.steering_filter_alpha * self.steering_filter +
#                           (1 - self.steering_filter_alpha) * steering_angle)
#         self.steering_filter = steering_angle
#         return np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

#     def calculate_speed(self, ranges, steering_angle):
#         center_idx = len(ranges) // 2
#         front_range = int(np.radians(30) / (2 * np.pi / len(ranges)))
#         front_distances = ranges[max(0, center_idx - front_range):min(len(ranges), center_idx + front_range)]
#         min_front_distance = np.min(front_distances)
#         avg_front_distance = np.mean(front_distances)

#         if min_front_distance < 1.0:
#             speed_factor = 0.3
#         elif min_front_distance < 2.0:
#             speed_factor = 0.6
#         elif avg_front_distance > 4.0:
#             speed_factor = 1.2
#         else:
#             speed_factor = 1.0

#         angle_factor = 1.0 - (abs(steering_angle) / self.max_steering_angle) * 0.5
#         target_speed = self.base_speed * speed_factor * angle_factor
#         target_speed = np.clip(target_speed, self.min_speed, self.max_speed)
#         self.speed_filter = (self.speed_filter_alpha * self.speed_filter +
#                              (1 - self.speed_filter_alpha) * target_speed)
#         return self.speed_filter

#     def lidar_callback(self, data):
#         self.scan_count += 1
#         if self.scan_count < 5:
#             return

#         ranges = self.preprocess_lidar(data.ranges)
#         closest_idx = np.argmin(ranges)
#         closest_distance = ranges[closest_idx]

#         bubble_ranges = (self.create_safety_bubble(ranges, closest_idx)
#                          if closest_distance < 1.5 else ranges)

#         gaps = self.find_gaps(bubble_ranges)
#         best_gap = self.select_best_gap(gaps, ranges)

#         drive_msg = AckermannDriveStamped()
#         drive_msg.header.stamp = self.get_clock().now().to_msg()
#         drive_msg.header.frame_id = "base_link"

#         if best_gap is not None:
#             start_idx, end_idx = best_gap
#             target_idx = self.get_target_point(start_idx, end_idx, ranges)

#             if target_idx is not None:
#                 steering_angle = self.calculate_steering_angle(target_idx, data)
#                 speed = self.calculate_speed(ranges, steering_angle)
#             else:
#                 speed = self.min_speed
#                 steering_angle = 0.0
#         else:
#             speed = self.min_speed
#             steering_angle = np.radians(10)

#         drive_msg.drive.speed = float(speed)
#         drive_msg.drive.steering_angle = float(steering_angle)
#         self.drive_pub.publish(drive_msg)

#     def odom_callback(self, msg):
#         x = msg.pose.pose.position.x
#         y = msg.pose.pose.position.y

#         # Zona de inicio
#         in_start_zone = (abs(x) < 2.0 and abs(y) < 2.0)

#         # Solo cuenta vuelta si:
#         # - Ya salió de la zona alguna vez
#         # - Y acaba de volver a entrar
#         if in_start_zone and not self.prev_in_start_zone and self.has_left_start_zone:
#             lap_end_time = time.time()
#             lap_duration = lap_end_time - self.lap_start_time
#             self.lap_times.append(lap_duration)
#             self.lap_count += 1

#             self.get_logger().info(f"🏁 Vuelta {self.lap_count} completada en {lap_duration:.2f} s.")
#             self.lap_start_time = lap_end_time

#             if self.lap_count == 10:
#                 self.get_logger().info(f"Número de vueltas requerido alcanzado ({self.lap_count})")
#                 shortest = min(self.lap_times)
#                 self.get_logger().info(f"Tiempo de vuelta más corto: {shortest:.2f} segundos")

#         # Marcar que ya ha salido de la zona al menos una vez
#         if not in_start_zone:
#             self.has_left_start_zone = True

#         # Guardar estado anterior
#         self.prev_in_start_zone = in_start_zone



# def main(args=None):
#     rclpy.init(args=args)
#     node = ReactiveFollowGap()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info("Node stopped cleanly")
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry


# -------------------- PID simple con anti-windup --------------------
class PID:
    def __init__(self, kp, ki, kd, i_min=-np.inf, i_max=np.inf, out_min=-np.inf, out_max=np.inf):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.i_min = i_min
        self.i_max = i_max
        self.out_min = out_min
        self.out_max = out_max
        self._e_prev = 0.0
        self._i = 0.0
        self._t_prev = None

    def reset(self):
        self._e_prev = 0.0
        self._i = 0.0
        self._t_prev = None

    def update(self, e, t_now):
        if self._t_prev is None:
            dt = 0.0
        else:
            dt = max(1e-3, t_now - self._t_prev)  # evita dt=0
        self._t_prev = t_now

        # Integración con clamp
        self._i += e * dt
        self._i = np.clip(self._i, self.i_min, self.i_max)

        d = (e - self._e_prev) / dt if dt > 0.0 else 0.0
        self._e_prev = e

        u = self.kp * e + self.ki * self._i + self.kd * d
        return float(np.clip(u, self.out_min, self.out_max))


class PIDFollowGap(Node):
    def __init__(self):
        super().__init__('pid_follow_gap')

        # Tópicos
        self.scan_topic = '/scan'
        self.drive_topic = '/drive'
        self.odom_topic = '/ego_racecar/odom'

        # ----- Parámetros de control -----
        # límites físicos
        self.max_steer = np.radians(30.0)
        self.min_speed = 1.0
        self.base_speed = 5.0
        self.max_speed = 11.0

        # objetivos de distancia
        self.side_target = 0.8         # m: queremos ~0.8 m respecto a paredes laterales
        self.front_target = 3.0        # m: mantener ~3 m libres al frente

        # zonas angulares (en grados, marco LiDAR)
        self.left_sector = (35.0, 85.0)    # sector para estimar pared izquierda
        self.right_sector = (-85.0, -35.0) # sector para pared derecha
        self.front_sector = (-15.0, 15.0)  # sector frontal

        # filtros suaves
        self.steer_alpha = 0.8
        self.speed_alpha = 0.7
        self._steer_filt = 0.0
        self._speed_filt = self.base_speed

        # seguridad
        self.max_lidar_range = 9.0
        self.block_front_thresh = 0.9  # si el frente < 0.9 m: maniobra de escape

        # PID de dirección (error lateral) y de velocidad (distancia frontal)
        # Ajusta estos gains para tu mapa.
        self.pid_steer = PID(
            kp=1.2, ki=0.0, kd=0.12,
            i_min=-1.0, i_max=1.0,
            out_min=-self.max_steer, out_max=self.max_steer
        )
        self.pid_speed = PID(
            kp=2.2, ki=0.0, kd=0.0,
            i_min=-5.0, i_max=5.0,
            out_min=self.min_speed, out_max=self.max_speed
        )

        # vuelta (opcional, igual que tu versión)
        self.lap_start_time = time.time()
        self.lap_count = 0
        self.prev_in_start_zone = False
        self.lap_times = []
        self.has_left_start_zone = False

        # Pub/Sub
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)

        self._scan_meta = None  # guardará (angle_min, angle_inc, n)
        self.get_logger().info('PIDFollowGap listo.')

    # -------------------- Utilidades LiDAR --------------------
    def preprocess(self, ranges):
        arr = np.asarray(ranges, dtype=np.float32)
        bad = (~np.isfinite(arr)) | (arr <= 0.0)
        arr[bad] = self.max_lidar_range
        return np.clip(arr, 0.05, self.max_lidar_range)

    def angle_to_index(self, angle_rad, angle_min, angle_inc, n):
        idx = int(np.round((angle_rad - angle_min) / angle_inc))
        return int(np.clip(idx, 0, n - 1))

    def sector_mean(self, arr, scan, deg_min, deg_max):
        """Media robusta (percentil 70) de un sector angular en grados."""
        a1 = np.radians(deg_min)
        a2 = np.radians(deg_max)
        if a2 < a1:
            a1, a2 = a2, a1
        n = len(arr)
        i1 = self.angle_to_index(a1, scan.angle_min, scan.angle_increment, n)
        i2 = self.angle_to_index(a2, scan.angle_min, scan.angle_increment, n)
        if i2 <= i1:
            return float(self.max_lidar_range)
        sector = arr[i1:i2+1]
        if sector.size == 0:
            return float(self.max_lidar_range)
        # media robusta: quita los más pequeños extremos (ruido)
        p70 = np.percentile(sector, 70)
        return float(np.mean(sector[sector >= p70])) if np.any(sector >= p70) else float(np.mean(sector))

    # -------------------- Callbacks --------------------
    def on_scan(self, scan: LaserScan):
        ranges = self.preprocess(scan.ranges)
        n = len(ranges)
        if self._scan_meta is None:
            self._scan_meta = (scan.angle_min, scan.angle_increment, n)

        # Distancias en sectores
        d_left  = self.sector_mean(ranges, scan, *self.left_sector)
        d_right = self.sector_mean(ranges, scan, *self.right_sector)
        d_front = self.sector_mean(ranges, scan, *self.front_sector)

        # ----- PID de dirección -----
        # error lateral = (dist derecha - target) - (dist izquierda - target) = d_right - d_left
        e_lateral = (d_right - d_left)
        t = self.get_clock().now().nanoseconds * 1e-9
        steer_cmd = self.pid_steer.update(e_lateral, t)

        # Low-pass al ángulo
        steer_cmd = self.steer_alpha * self._steer_filt + (1.0 - self.steer_alpha) * steer_cmd
        steer_cmd = float(np.clip(steer_cmd, -self.max_steer, self.max_steer))
        self._steer_filt = steer_cmd

        # ----- PID de velocidad -----
        # error frontal = distancia objetivo - distancia medida (positivo => acelerar)
        e_front = self.front_target - d_front
        speed_cmd = self.pid_speed.update(-e_front, t)  # signo para que si d_front < target, reduzca

        # seguridad: bloqueo frontal severo -> freno fuerte y ladeo al lado con más espacio
        if d_front < self.block_front_thresh:
            side = 1.0 if d_left > d_right else -1.0  # gira hacia el lado más libre
            steer_cmd = float(np.clip(side * self.max_steer * 0.6, -self.max_steer, self.max_steer))
            speed_cmd = max(self.min_speed * 0.2, 0.0)

        # Low-pass a la velocidad
        speed_cmd = self.speed_alpha * self._speed_filt + (1.0 - self.speed_alpha) * speed_cmd
        speed_cmd = float(np.clip(speed_cmd, self.min_speed, self.max_speed))
        self._speed_filt = speed_cmd

        # Publica comando
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer_cmd
        msg.drive.speed = speed_cmd
        self.drive_pub.publish(msg)

    # ---- vuelta (igual que tu lógica) ----
    def on_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        in_start_zone = (abs(x) < 2.0 and abs(y) < 2.0)

        if in_start_zone and not self.prev_in_start_zone and self.has_left_start_zone:
            lap_end_time = time.time()
            lap_duration = lap_end_time - self.lap_start_time
            self.lap_times.append(lap_duration)
            self.lap_count += 1
            self.get_logger().info(f"🏁 Vuelta {self.lap_count} en {lap_duration:.2f} s.")
            self.lap_start_time = lap_end_time
            if self.lap_count == 10:
                self.get_logger().info(f"10 vueltas completadas. Mejor: {min(self.lap_times):.2f} s")

        if not in_start_zone:
            self.has_left_start_zone = True
        self.prev_in_start_zone = in_start_zone


def main(args=None):
    rclpy.init(args=args)
    node = PIDFollowGap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped cleanly')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
