#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

class ReactiveFollowGap(Node):
    def __init__(self):
        super().__init__('reactive_node')

        # Topics
        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        odom_topic = '/ego_racecar/odom'

        # Parámetros del algoritmo
        self.base_speed = 4.8
        self.max_speed = 10.0
        self.min_speed = 1.5
        self.max_lidar_range = 9.0
        self.bubble_radius_m = 1.1
        self.min_clearance = 1.6
        self.max_steering_angle = np.radians(30)
        self.gap_threshold = 0.5
        self.center_bias = 0.5
        self.speed_filter = self.base_speed
        self.speed_filter_alpha = 0.7
        self.steering_filter = 0.0
        self.steering_filter_alpha = 0.8
        self.scan_initialized = False
        self.scan_count = 0

        # Control de vueltas
        self.lap_start_time = time.time()
        self.lap_count = 0
        self.prev_in_start_zone = False
        self.lap_times = []
        self.has_left_start_zone = False 

        # Publishers & Subscribers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        self.get_logger().info("ReactiveFollowGap node initialized.")

    def preprocess_lidar(self, ranges):
        proc = np.array(ranges, dtype=np.float32)
        invalid_mask = (proc == 0) | np.isinf(proc) | np.isnan(proc)
        proc[invalid_mask] = self.max_lidar_range
        proc = np.clip(proc, 0.1, self.max_lidar_range)
        kernel = np.ones(3) / 3
        return np.convolve(proc, kernel, mode='same')

    def create_safety_bubble(self, ranges, closest_idx):
        bubble_ranges = np.copy(ranges)
        angle_increment = 2 * np.pi / len(ranges)
        bubble_radius_idx = int(self.bubble_radius_m / (ranges[closest_idx] * angle_increment))
        bubble_radius_idx = max(5, min(bubble_radius_idx, 20))
        start_idx = max(0, closest_idx - bubble_radius_idx)
        end_idx = min(len(ranges) - 1, closest_idx + bubble_radius_idx)
        bubble_ranges[start_idx:end_idx + 1] = 0.0
        return bubble_ranges

    def find_gaps(self, ranges):
        free_mask = ranges >= self.min_clearance
        gaps = []
        start_idx = None
        for i, is_free in enumerate(free_mask):
            if is_free and start_idx is None:
                start_idx = i
            elif not is_free and start_idx is not None:
                gap_width_rad = (i - start_idx) * (2 * np.pi / len(ranges))
                gap_width_m = gap_width_rad * np.mean(ranges[start_idx:i])
                if gap_width_m >= self.gap_threshold:
                    gaps.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None:
            gap_width_rad = (len(free_mask) - start_idx) * (2 * np.pi / len(ranges))
            gap_width_m = gap_width_rad * np.mean(ranges[start_idx:])
            if gap_width_m >= self.gap_threshold:
                gaps.append((start_idx, len(free_mask) - 1))
        return gaps

    def select_best_gap(self, gaps, ranges):
        if not gaps:
            return None
        best_gap = None
        best_score = -1
        center_idx = len(ranges) // 2
        for start_idx, end_idx in gaps:
            gap_ranges = ranges[start_idx:end_idx + 1]
            avg_distance = np.mean(gap_ranges)
            max_distance = np.max(gap_ranges)
            gap_width = end_idx - start_idx
            gap_center = (start_idx + end_idx) // 2
            center_distance = abs(gap_center - center_idx)
            distance_score = min(avg_distance / 5.0, 1.0)
            width_score = min(gap_width / 50.0, 1.0)
            center_score = 1.0 - (center_distance / center_idx) * self.center_bias
            total_score = (distance_score * 0.4 + width_score * 0.3 + center_score * 0.3)
            if total_score > best_score:
                best_score = total_score
                best_gap = (start_idx, end_idx)
        return best_gap

    def get_target_point(self, start_idx, end_idx, ranges):
        if start_idx is None or end_idx is None:
            return None
        gap_ranges = ranges[start_idx:end_idx + 1]
        max_dist_idx = np.argmax(gap_ranges)
        target_idx = start_idx + max_dist_idx
        gap_center = (start_idx + end_idx) // 2
        target_idx = int(0.7 * target_idx + 0.3 * gap_center)
        return target_idx

    def calculate_steering_angle(self, target_idx, scan_data):
        if target_idx is None:
            return 0.0
        target_angle = scan_data.angle_min + target_idx * scan_data.angle_increment
        center_angle = scan_data.angle_min + (len(scan_data.ranges) // 2) * scan_data.angle_increment
        steering_angle = (target_angle - center_angle) * 0.8
        steering_angle = (self.steering_filter_alpha * self.steering_filter +
                          (1 - self.steering_filter_alpha) * steering_angle)
        self.steering_filter = steering_angle
        return np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

    def calculate_speed(self, ranges, steering_angle):
        center_idx = len(ranges) // 2
        front_range = int(np.radians(30) / (2 * np.pi / len(ranges)))
        front_distances = ranges[max(0, center_idx - front_range):min(len(ranges), center_idx + front_range)]
        min_front_distance = np.min(front_distances)
        avg_front_distance = np.mean(front_distances)

        if min_front_distance < 1.0:
            speed_factor = 0.3
        elif min_front_distance < 2.0:
            speed_factor = 0.6
        elif avg_front_distance > 4.0:
            speed_factor = 1.2
        else:
            speed_factor = 1.0

        angle_factor = 1.0 - (abs(steering_angle) / self.max_steering_angle) * 0.5
        target_speed = self.base_speed * speed_factor * angle_factor
        target_speed = np.clip(target_speed, self.min_speed, self.max_speed)
        self.speed_filter = (self.speed_filter_alpha * self.speed_filter +
                             (1 - self.speed_filter_alpha) * target_speed)
        return self.speed_filter

    def lidar_callback(self, data):
        self.scan_count += 1
        if self.scan_count < 5:
            return

        ranges = self.preprocess_lidar(data.ranges)
        closest_idx = np.argmin(ranges)
        closest_distance = ranges[closest_idx]

        bubble_ranges = (self.create_safety_bubble(ranges, closest_idx)
                         if closest_distance < 1.5 else ranges)

        gaps = self.find_gaps(bubble_ranges)
        best_gap = self.select_best_gap(gaps, ranges)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"

        if best_gap is not None:
            start_idx, end_idx = best_gap
            target_idx = self.get_target_point(start_idx, end_idx, ranges)

            if target_idx is not None:
                steering_angle = self.calculate_steering_angle(target_idx, data)
                speed = self.calculate_speed(ranges, steering_angle)
            else:
                speed = self.min_speed
                steering_angle = 0.0
        else:
            speed = self.min_speed
            steering_angle = np.radians(10)

        drive_msg.drive.speed = float(speed)
        drive_msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(drive_msg)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Zona de inicio
        in_start_zone = (abs(x) < 2.0 and abs(y) < 2.0)

        # Solo cuenta vuelta si:
        # - Ya salió de la zona alguna vez
        # - Y acaba de volver a entrar
        if in_start_zone and not self.prev_in_start_zone and self.has_left_start_zone:
            lap_end_time = time.time()
            lap_duration = lap_end_time - self.lap_start_time
            self.lap_times.append(lap_duration)
            self.lap_count += 1

            self.get_logger().info(f"🏁 Vuelta {self.lap_count} completada en {lap_duration:.2f} s.")
            self.lap_start_time = lap_end_time

            if self.lap_count == 10:
                self.get_logger().info(f"Número de vueltas requerido alcanzado ({self.lap_count})")
                shortest = min(self.lap_times)
                self.get_logger().info(f"Tiempo de vuelta más corto: {shortest:.2f} segundos")

        # Marcar que ya ha salido de la zona al menos una vez
        if not in_start_zone:
            self.has_left_start_zone = True

        # Guardar estado anterior
        self.prev_in_start_zone = in_start_zone



def main(args=None):
    rclpy.init(args=args)
    node = ReactiveFollowGap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped cleanly")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

