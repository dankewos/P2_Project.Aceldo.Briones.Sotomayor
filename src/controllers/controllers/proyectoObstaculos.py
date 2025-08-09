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

        # Tópicos
        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        odom_topic = '/ego_racecar/odom'

        # Parámetros del algoritmo optimizados para seguridad
        self.max_speed = 8.0
        self.min_speed = 1.5
        self.base_speed = 5.0
        self.max_lidar_range = 9.0
        self.bubble_radius_m = 1.5  # Burbuja de seguridad más grande
        self.min_clearance = 2.5    # Distancia de seguridad aumentada
        self.max_steering_angle = np.radians(30)
        self.gap_threshold_m = 1.0  # Umbral de brecha más grande
        self.center_bias = 0.9      # Mayor prioridad a las brechas centrales

        # Parámetros de filtrado
        self.speed_filter = self.base_speed
        self.speed_filter_alpha = 0.7  # Menor suavizado de velocidad para reaccionar más rápido
        self.steering_filter = 0.0
        self.steering_filter_alpha = 0.8 # Suavizado de dirección para evitar movimientos bruscos

        # Publishers & Subscribers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        # Control de vueltas (sin cambios)
        self.lap_start_time = time.time()
        self.lap_count = 0
        self.prev_in_start_zone = False
        self.lap_times = []
        self.has_left_start_zone = False

        self.get_logger().info("ReactiveFollowGap node initialized with enhanced safety.")

    def preprocess_lidar(self, ranges):
        """Preprocesa los datos del LiDAR y los suaviza."""
        proc = np.array(ranges, dtype=np.float32)
        invalid_mask = (proc == 0) | np.isinf(proc) | np.isnan(proc)
        proc[invalid_mask] = self.max_lidar_range
        proc = np.clip(proc, 0.1, self.max_lidar_range)
        kernel = np.ones(9) / 9.0  # Kernel de suavizado más grande para una detección más estable
        return np.convolve(proc, kernel, mode='same')

    def create_safety_bubble(self, ranges, closest_idx):
        """Crea una burbuja de seguridad más grande y adaptativa."""
        bubble_ranges = np.copy(ranges)
        # El radio de la burbuja es inversamente proporcional a la distancia del obstáculo
        bubble_radius_idx = int(self.bubble_radius_m / ranges[closest_idx] * (len(ranges) / (2 * np.pi)))
        bubble_radius_idx = max(5, min(bubble_radius_idx, 50))
        
        start_idx = max(0, closest_idx - bubble_radius_idx)
        end_idx = min(len(ranges) - 1, closest_idx + bubble_radius_idx)
        
        # Eliminar las lecturas dentro de la burbuja
        bubble_ranges[start_idx:end_idx + 1] = 0.0
        return bubble_ranges

    def find_gaps(self, ranges):
        """Identifica brechas candidatas que son lo suficientemente grandes."""
        free_mask = ranges >= self.min_clearance
        gaps = []
        start_idx = None
        for i, is_free in enumerate(free_mask):
            if is_free and start_idx is None:
                start_idx = i
            elif not is_free and start_idx is not None:
                gap_width = (i - start_idx) * (2 * np.pi / len(ranges)) * np.mean(ranges[start_idx:i])
                if gap_width >= self.gap_threshold_m:
                    gaps.append((start_idx, i - 1))
                start_idx = None
        
        if start_idx is not None:
            gap_width = (len(free_mask) - start_idx) * (2 * np.pi / len(ranges)) * np.mean(ranges[start_idx:])
            if gap_width >= self.gap_threshold_m:
                gaps.append((start_idx, len(free_mask) - 1))
        
        return gaps

    def select_best_gap(self, gaps, ranges):
        """Selecciona la brecha más segura y óptima."""
        if not gaps:
            return None
        
        best_gap = None
        best_score = -np.inf
        center_idx = len(ranges) // 2

        for start_idx, end_idx in gaps:
            gap_center = (start_idx + end_idx) // 2
            
            # Puntuación mejorada para la seguridad
            avg_distance_in_gap = np.mean(ranges[start_idx:end_idx + 1])
            distance_to_center = abs(gap_center - center_idx)
            gap_width = end_idx - start_idx

            # Se penaliza fuertemente estar lejos del centro
            score = (avg_distance_in_gap * 0.6) - (distance_to_center / center_idx * self.center_bias) + (gap_width / len(ranges) * 0.4)

            if score > best_score:
                best_score = score
                best_gap = (start_idx, end_idx)
        
        return best_gap

    def get_target_point(self, start_idx, end_idx, ranges):
        """Selecciona un punto de destino seguro y previsible."""
        if start_idx is None or end_idx is None:
            return None
        
        # Elige el punto más lejano dentro del 80% del segmento de la brecha
        # Esto evita que el coche se dirija a los bordes de la brecha
        gap_ranges = ranges[start_idx:end_idx + 1]
        safe_segment_length = int(len(gap_ranges) * 0.8)
        start_safe = start_idx + (len(gap_ranges) - safe_segment_length) // 2
        end_safe = start_safe + safe_segment_length
        
        safe_gap_ranges = ranges[start_safe:end_safe + 1]
        
        if safe_gap_ranges.size == 0:
            return (start_idx + end_idx) // 2
            
        max_dist_idx_in_safe_gap = np.argmax(safe_gap_ranges)
        target_idx = start_safe + max_dist_idx_in_safe_gap
        
        return target_idx

    def calculate_steering_angle(self, target_idx, scan_data):
        """Calcula y suaviza el ángulo de dirección."""
        if target_idx is None:
            return 0.0
        
        target_angle = scan_data.angle_min + target_idx * scan_data.angle_increment
        center_angle = 0.0
        
        steering_angle = (target_angle - center_angle) * 0.95  # Factor de ganancia ajustado
        
        # Suavizado de dirección
        self.steering_filter = (self.steering_filter_alpha * self.steering_filter +
                                (1 - self.steering_filter_alpha) * steering_angle)
        
        return np.clip(self.steering_filter, -self.max_steering_angle, self.max_steering_angle)

    def calculate_speed(self, ranges, steering_angle):
        """Modula la velocidad de manera más cautelosa."""
        center_idx = len(ranges) // 2
        # Cono frontal más amplio para una detección anticipada
        front_beam_width = int(np.radians(60) / (2 * np.pi / len(ranges)))
        front_distances = ranges[max(0, center_idx - front_beam_width):min(len(ranges), center_idx + front_beam_width)]
        
        if not front_distances.size:
            return self.min_speed
            
        min_front_distance = np.min(front_distances)
        
        # Lógica de velocidad más conservadora
        if min_front_distance > 5.0:
            speed_factor = 1.0  # Velocidad base
        elif min_front_distance > 3.5:
            speed_factor = 0.8
        elif min_front_distance > 2.0:
            speed_factor = 0.5
        else:
            speed_factor = 0.3 # Reduce la velocidad drásticamente si está muy cerca de un muro
            
        # Reducir la velocidad en curvas de manera más agresiva
        angle_factor = 1.0 - (abs(steering_angle) / self.max_steering_angle)**2.0 * 0.8
        
        target_speed = self.base_speed * speed_factor * angle_factor
        
        # Suavizado de velocidad
        self.speed_filter = (self.speed_filter_alpha * self.speed_filter +
                             (1 - self.speed_filter_alpha) * target_speed)
                             
        return np.clip(self.speed_filter, self.min_speed, self.max_speed)

    def lidar_callback(self, data):
        """Callback principal para procesar los datos del LiDAR y publicar la dirección."""
        ranges = self.preprocess_lidar(data.ranges)
        
        closest_idx = np.argmin(ranges)
        closest_distance = ranges[closest_idx]
        
        if closest_distance < self.min_clearance:
            bubble_ranges = self.create_safety_bubble(ranges, closest_idx)
        else:
            bubble_ranges = ranges

        gaps = self.find_gaps(bubble_ranges)
        best_gap = self.select_best_gap(gaps, ranges)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"

        if best_gap is not None:
            start_idx, end_idx = best_gap
            target_idx = self.get_target_point(start_idx, end_idx, ranges)
            
            steering_angle = self.calculate_steering_angle(target_idx, data)
            speed = self.calculate_speed(ranges, steering_angle)
        else:
            # Estrategia de fallback: girar en una dirección segura y moverse lentamente
            speed = self.min_speed
            steering_angle = np.radians(20) if closest_idx > len(ranges) // 2 else np.radians(-20)
            self.get_logger().warn("No gaps found, resorting to fallback strategy.")
            self.steering_filter = steering_angle 

        drive_msg.drive.speed = float(speed)
        drive_msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(drive_msg)

    def odom_callback(self, msg):
        """Callback para el control de vueltas."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        in_start_zone = (abs(x) < 2.0 and abs(y) < 2.0)

        if in_start_zone and not self.prev_in_start_zone and self.has_left_start_zone:
            lap_end_time = time.time()
            lap_duration = lap_end_time - self.lap_start_time
            self.lap_times.append(lap_duration)
            self.lap_count += 1
            self.get_logger().info(f"🏁 Vuelta {self.lap_count} completada en {lap_duration:.2f} s.")
            self.lap_start_time = lap_end_time
            if self.lap_count >= 10:
                self.get_logger().info(f"Número de vueltas requerido alcanzado ({self.lap_count})")
                if self.lap_times:
                    shortest = min(self.lap_times)
                    self.get_logger().info(f"Tiempo de vuelta más corto: {shortest:.2f} segundos")

        if not in_start_zone:
            self.has_left_start_zone = True

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