#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower_node')

        # Topics
        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        odom_topic = '/ego_racecar/odom'

        # Parámetros del algoritmo de seguimiento de pared
        self.max_speed = 2.5  # Velocidad máxima reducida a la mitad
        self.min_speed = 1.5  # Velocidad mínima ajustada
        self.max_lidar_range = 10.0
        self.wall_distance_ref = 1.5  # Distancia de referencia a la pared aumentada a 2 metros
        
        # PID para el ángulo
        self.steering_pid_kp = 0.7
        self.steering_pid_ki = 0.0
        self.steering_pid_kd = 0.1
        self.last_error_angle = 0.0
        self.integral_angle = 0.0

        # PID para la velocidad
        self.speed_pid_kp = 0.05
        self.speed_pid_ki = 0.0
        self.speed_pid_kd = 0.01
        self.last_error_speed = 0.0
        self.integral_speed = 0.0

        # Control de vueltas (se mantiene sin cambios)
        self.lap_start_time = time.time()
        self.lap_count = 0
        self.prev_in_start_zone = False
        self.lap_times = []
        self.has_left_start_zone = False

        # Publishers & Subscribers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        self.get_logger().info("WallFollower node initialized.")

    def preprocess_lidar(self, ranges):
        """Preprocesa los datos del LiDAR: reemplaza valores inválidos y aplica un filtro."""
        proc = np.array(ranges, dtype=np.float32)
        invalid_mask = (proc == 0) | np.isinf(proc) | np.isnan(proc)
        proc[invalid_mask] = self.max_lidar_range
        proc = np.clip(proc, 0.1, self.max_lidar_range)
        kernel = np.ones(3) / 3
        return np.convolve(proc, kernel, mode='same')

    def select_cleanest_wall(self, ranges, angle_increment):
        """Selecciona el lado con la pared más 'limpia' (menos protuberancias)."""
        num_ranges = len(ranges)
        mid_idx = num_ranges // 2
        
        # Rango de ángulos para considerar los lados derecho e izquierdo
        angle_range_deg = 45
        points_to_check = int(np.radians(angle_range_deg) / angle_increment)

        right_ranges = ranges[mid_idx - points_to_check : mid_idx]
        left_ranges = ranges[mid_idx : mid_idx + points_to_check]
        
        # Calcular la desviación estándar para cada lado
        std_dev_right = np.std(right_ranges)
        std_dev_left = np.std(left_ranges)

        # Si el lado derecho es más 'limpio' (menor desviación), lo seguimos.
        if std_dev_right < std_dev_left:
            return 'right', right_ranges
        else:
            return 'left', left_ranges

    def calculate_pid(self, error, last_error, integral, kp, ki, kd):
        """Función genérica para el cálculo PID."""
        integral += error
        derivative = error - last_error
        output = kp * error + ki * integral + kd * derivative
        return output, integral

    def get_wall_following_angle(self, side_to_follow, side_ranges):
        """Calcula el ángulo de dirección para seguir la pared 'limpia'."""
        avg_dist = np.mean(side_ranges)
        error = self.wall_distance_ref - avg_dist
        
        steering_angle, self.integral_angle = self.calculate_pid(
            error, self.last_error_angle, self.integral_angle, 
            self.steering_pid_kp, self.steering_pid_ki, self.steering_pid_kd
        )
        self.last_error_angle = error
        
        if side_to_follow == 'left':
            steering_angle = -steering_angle
            
        return np.clip(steering_angle, -0.5, 0.5)

    def calculate_speed(self, ranges, steering_angle):
        """Ajusta la velocidad en función de la distancia frontal y el ángulo de dirección."""
        # Reducir la velocidad en curvas cerradas
        speed = self.max_speed - abs(steering_angle) * 1.5
        
        # Usar PID para ajustar la velocidad según la distancia frontal
        front_range = int(np.radians(20) / (2 * np.pi / len(ranges)))
        front_distances = ranges[len(ranges) // 2 - front_range : len(ranges) // 2 + front_range]
        min_front_distance = np.min(front_distances)

        speed_error = min_front_distance - 2.0
        speed_adjustment, self.integral_speed = self.calculate_pid(
            speed_error, self.last_error_speed, self.integral_speed,
            self.speed_pid_kp, self.speed_pid_ki, self.speed_pid_kd
        )
        self.last_error_speed = speed_error
        
        speed += speed_adjustment
        
        return np.clip(speed, self.min_speed, self.max_speed)

    def lidar_callback(self, data):
        ranges = self.preprocess_lidar(data.ranges)
        
        # Seleccionar la pared más 'limpia'
        side_to_follow, side_ranges = self.select_cleanest_wall(ranges, data.angle_increment)
        
        # Calcular ángulo y velocidad
        steering_angle = self.get_wall_following_angle(side_to_follow, side_ranges)
        speed = self.calculate_speed(ranges, steering_angle)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.speed = float(speed)
        drive_msg.drive.steering_angle = float(steering_angle)

        self.drive_pub.publish(drive_msg)

    def odom_callback(self, msg):
        """Mantiene la lógica para contar las vueltas."""
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

            if self.lap_count == 10:
                self.get_logger().info(f"Número de vueltas requerido alcanzado ({self.lap_count})")
                shortest = min(self.lap_times)
                self.get_logger().info(f"Tiempo de vuelta más corto: {shortest:.2f} segundos")

        if not in_start_zone:
            self.has_left_start_zone = True

        self.prev_in_start_zone = in_start_zone

def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped cleanly")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()