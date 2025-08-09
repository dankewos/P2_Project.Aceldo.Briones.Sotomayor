#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

class PIDController:
    """Controlador PID para regular el ángulo de dirección."""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        self.output_limit = np.radians(30)
        self.reset()

    def reset(self):
        """Reinicia el estado del controlador."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()

    def update(self, error):
        """Calcula y actualiza la señal de control."""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt == 0:
            return 0.0

        # Componente Proporcional
        proportional = self.kp * error

        # Componente Integral
        self.integral += error * dt
        integral_term = self.ki * self.integral

        # Componente Derivativo
        derivative = (error - self.last_error) / dt
        derivative_term = self.kd * derivative

        output = proportional + integral_term + derivative_term
        self.last_error = error
        self.last_time = current_time

        # Limitar la salida para evitar movimientos bruscos
        return np.clip(output, -self.output_limit, self.output_limit)

class ReactiveFollowGap(Node):
    def __init__(self):
        super().__init__('reactive_node')

        # Tópicos
        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        odom_topic = '/ego_racecar/odom'

        # Parámetros del algoritmo
        self.base_speed = 4.0
        self.max_speed = 10.0
        self.min_speed = 1.0
        self.max_lidar_range = 9.0
        self.bubble_radius_m = 1.2
        self.min_clearance = 1.8
        self.max_steering_angle = np.radians(30)
        self.gap_threshold_m = 0.7 
        self.center_bias = 0.7

        # Parámetros PID
        self.pid_kp = 0.8
        self.pid_ki = 0.0
        self.pid_kd = 0.1
        self.pid_controller = PIDController(self.pid_kp, self.pid_ki, self.pid_kd)

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

        self.get_logger().info("ReactiveFollowGap node initialized with PID controller.")

    def preprocess_lidar(self, ranges):
        """Preprocesa los datos del LiDAR para manejar valores inválidos y suavizar."""
        proc = np.array(ranges, dtype=np.float32)
        invalid_mask = (proc == 0) | np.isinf(proc) | np.isnan(proc)
        proc[invalid_mask] = self.max_lidar_range
        proc = np.clip(proc, 0.1, self.max_lidar_range)
        kernel = np.ones(5) / 5.0 # Mayor suavizado
        return np.convolve(proc, kernel, mode='same')

    def create_safety_bubble(self, ranges, closest_idx):
        """Crea una burbuja de seguridad virtual alrededor del obstáculo más cercano."""
        bubble_ranges = np.copy(ranges)
        # Radio de la burbuja en índices, adaptado a la distancia
        bubble_radius_idx = int(self.bubble_radius_m * (1/ranges[closest_idx]) * (len(ranges) / (2 * np.pi)))
        bubble_radius_idx = max(2, min(bubble_radius_idx, 30))
        
        start_idx = max(0, closest_idx - bubble_radius_idx)
        end_idx = min(len(ranges) - 1, closest_idx + bubble_radius_idx)
        bubble_ranges[start_idx:end_idx + 1] = 0.0
        return bubble_ranges

    def find_gaps(self, ranges):
        """Identifica brechas candidatas en los datos del LiDAR."""
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
        """Selecciona la mejor brecha basándose en una heurística mejorada."""
        if not gaps:
            return None
        
        best_gap = None
        best_score = -np.inf
        center_idx = len(ranges) // 2

        for start_idx, end_idx in gaps:
            gap_center = (start_idx + end_idx) // 2
            
            # Prioriza brechas más lejanas y centradas
            distance_to_center = abs(gap_center - center_idx)
            avg_distance_in_gap = np.mean(ranges[start_idx:end_idx + 1])

            # Heurística de puntuación: pondera la distancia, el tamaño y la centralidad.
            score = (avg_distance_in_gap * 0.7) - (distance_to_center / center_idx * self.center_bias) + ((end_idx - start_idx) / len(ranges) * 0.2)

            if score > best_score:
                best_score = score
                best_gap = (start_idx, end_idx)
        
        return best_gap

    def get_target_point(self, start_idx, end_idx, ranges):
        """Encuentra el punto de destino dentro de la brecha seleccionada."""
        if start_idx is None or end_idx is None:
            return None
        
        gap_ranges = ranges[start_idx:end_idx + 1]
        
        # Elige el punto más lejano dentro de la brecha
        max_dist_idx_in_gap = np.argmax(gap_ranges)
        target_idx = start_idx + max_dist_idx_in_gap
        
        return target_idx

    def calculate_steering_angle(self, target_idx, scan_data):
        """Calcula el ángulo de dirección usando el controlador PID."""
        if target_idx is None:
            self.pid_controller.reset()
            return 0.0
        
        # Ángulo objetivo en radianes
        target_angle = scan_data.angle_min + target_idx * scan_data.angle_increment
        center_angle = 0.0  # El centro del láser es 0 grados
        
        # El 'error' es la diferencia entre el ángulo objetivo y el centro
        error = target_angle - center_angle
        
        # Usar el controlador PID para obtener el ángulo de dirección
        steering_angle = self.pid_controller.update(error)
        
        return np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

    def calculate_speed(self, ranges, steering_angle):
        """Modula la velocidad basándose en el entorno y el ángulo de dirección."""
        center_idx = len(ranges) // 2
        # Distancia en un cono frontal más amplio
        front_beam_width = int(np.radians(45) / (2 * np.pi / len(ranges)))
        front_distances = ranges[max(0, center_idx - front_beam_width):min(len(ranges), center_idx + front_beam_width)]
        
        if len(front_distances) == 0:
            return self.min_speed
            
        min_front_distance = np.min(front_distances)
        
        # Reducir la velocidad si hay obstáculos cerca
        if min_front_distance < 1.0:
            speed_factor = 0.2
        elif min_front_distance < 2.5:
            speed_factor = 0.5
        elif min_front_distance < 4.0:
            speed_factor = 0.7
        else:
            speed_factor = 1.0
            
        # Reducir la velocidad en curvas cerradas
        angle_factor = 1.0 - (abs(steering_angle) / self.max_steering_angle) * 0.6
        
        target_speed = self.base_speed * speed_factor * angle_factor
        return np.clip(target_speed, self.min_speed, self.max_speed)

    def lidar_callback(self, data):
        """Callback principal para procesar los datos del LiDAR y publicar la dirección."""
        ranges = self.preprocess_lidar(data.ranges)
        
        closest_idx = np.argmin(ranges)
        closest_distance = ranges[closest_idx]
        
        # Activar la burbuja de seguridad solo si hay un obstáculo cercano
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
            self.pid_controller.reset()

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