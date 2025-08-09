#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

class WallFollowingNode(Node):
    def __init__(self):
        super().__init__('wall_following_node')

        # Topics
        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        odom_topic = '/ego_racecar/odom'

        # Parámetros de wall following
        self.desired_distance = 0.8  # Distancia deseada a la pared (80cm)
        self.lookahead_distance = 1.5  # Distancia de lookahead
        self.max_speed = 5.0
        self.min_speed = 1.0
        self.base_speed = 3.0
        self.max_steering_angle = np.radians(30)
        
        # Parámetros del controlador PID
        self.kp = 1.5  # Proporcional
        self.ki = 0.1  # Integral  
        self.kd = 0.8  # Derivativo
        
        # Variables del PID
        self.prev_error = 0.0
        self.integral = 0.0
        
        # Variables de estado
        self.wall_side = "right"  # Por defecto seguir pared derecha
        self.switch_wall_counter = 0
        self.last_wall_switch = time.time()
        self.scan_count = 0  # Contador de scans
        
        # Detección de obstáculos frontales
        self.front_obstacle_threshold = 1.0
        
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

        self.get_logger().info("Wall Following Node initialized - Right wall following")

    def preprocess_lidar(self, ranges):
        """Preprocesamiento del LiDAR"""
        proc = np.array(ranges, dtype=np.float32)
        
        # Reemplazar valores inválidos
        invalid_mask = (proc == 0) | np.isinf(proc) | np.isnan(proc)
        proc[invalid_mask] = 10.0
        proc = np.clip(proc, 0.1, 10.0)
        
        return proc

    def get_wall_distance(self, ranges):
        """Obtiene la distancia a la pared que estamos siguiendo"""
        num_ranges = len(ranges)
        angle_increment = 2 * np.pi / num_ranges
        
        if self.wall_side == "right":
            # Ángulos para pared derecha (270° ± 30°)
            right_angle_idx = int((3 * np.pi / 2) / angle_increment) % num_ranges
            range_span = int(np.radians(60) / angle_increment)
        else:
            # Ángulos para pared izquierda (90° ± 30°)
            right_angle_idx = int((np.pi / 2) / angle_increment) % num_ranges
            range_span = int(np.radians(60) / angle_increment)
        
        start_idx = max(0, right_angle_idx - range_span // 2)
        end_idx = min(num_ranges, right_angle_idx + range_span // 2)
        
        # Tomar la distancia mínima en ese rango (más conservador)
        wall_distances = ranges[start_idx:end_idx]
        return np.min(wall_distances)

    def get_lookahead_distance(self, ranges):
        """Obtiene la distancia de lookahead"""
        num_ranges = len(ranges)
        angle_increment = 2 * np.pi / num_ranges
        
        if self.wall_side == "right":
            # Lookahead a 45° hacia la derecha
            lookahead_angle = 3 * np.pi / 4  # 135°
        else:
            # Lookahead a 45° hacia la izquierda  
            lookahead_angle = np.pi / 4  # 45°
            
        lookahead_idx = int(lookahead_angle / angle_increment) % num_ranges
        return ranges[lookahead_idx]

    def detect_front_obstacle(self, ranges):
        """Detecta obstáculos frontales"""
        center_idx = len(ranges) // 2
        front_range = int(np.radians(30) / (2 * np.pi / len(ranges)))
        
        front_distances = ranges[max(0, center_idx - front_range):
                               min(len(ranges), center_idx + front_range)]
        
        min_front = np.min(front_distances)
        return min_front < self.front_obstacle_threshold

    def should_switch_wall(self, ranges):
        """Determina si debe cambiar de pared a seguir"""
        current_time = time.time()
        
        # No cambiar muy frecuentemente
        if current_time - self.last_wall_switch < 2.0:
            return False
            
        center_idx = len(ranges) // 2
        left_space = np.mean(ranges[:center_idx])
        right_space = np.mean(ranges[center_idx:])
        
        # Cambiar si la pared actual está muy cerca y la otra tiene más espacio
        current_wall_dist = self.get_wall_distance(ranges)
        
        if self.wall_side == "right":
            other_wall_space = left_space
        else:
            other_wall_space = right_space
            
        # Cambiar si la pared actual está muy cerca Y la otra tiene mucho más espacio
        should_switch = (current_wall_dist < 0.4 and other_wall_space > current_wall_dist * 2.0)
        
        return should_switch

    def calculate_wall_following_steering(self, wall_distance, lookahead_distance):
        """Controlador PID para wall following"""
        # Error: diferencia entre distancia deseada y actual
        error = self.desired_distance - wall_distance
        
        # Componente proporcional
        proportional = self.kp * error
        
        # Componente integral
        self.integral += error
        self.integral = np.clip(self.integral, -1.0, 1.0)  # Anti-windup
        integral_term = self.ki * self.integral
        
        # Componente derivativo
        derivative = self.kd * (error - self.prev_error)
        self.prev_error = error
        
        # Control PID total
        pid_output = proportional + integral_term + derivative
        
        # Ajuste por lookahead (anticipación)
        lookahead_error = self.desired_distance - lookahead_distance
        lookahead_adjustment = lookahead_error * 0.3
        
        # Combinar PID con lookahead
        steering_output = pid_output + lookahead_adjustment
        
        # Invertir si seguimos pared izquierda
        if self.wall_side == "left":
            steering_output = -steering_output
            
        return np.clip(steering_output, -self.max_steering_angle, self.max_steering_angle)

    def calculate_speed(self, ranges, steering_angle):
        """Velocidad adaptativa para wall following"""
        center_idx = len(ranges) // 2
        front_range = int(np.radians(45) / (2 * np.pi / len(ranges)))
        
        front_distances = ranges[max(0, center_idx - front_range):
                               min(len(ranges), center_idx + front_range)]
        
        min_front = np.min(front_distances)
        avg_front = np.mean(front_distances)
        
        # Factor base de velocidad
        if min_front < 0.5:
            speed_factor = 0.2
        elif min_front < 1.0:
            speed_factor = 0.4
        elif min_front < 2.0:
            speed_factor = 0.7
        elif avg_front > 3.0:
            speed_factor = 1.2
        else:
            speed_factor = 1.0
        
        # Penalización por ángulo de dirección
        angle_factor = 1.0 - (abs(steering_angle) / self.max_steering_angle) * 0.4
        
        target_speed = self.base_speed * speed_factor * angle_factor
        return np.clip(target_speed, self.min_speed, self.max_speed)

    def lidar_callback(self, data):
        """Callback principal de wall following"""
        self.scan_count += 1  # Incrementar contador
        ranges = self.preprocess_lidar(data.ranges)
        
        # Verificar si hay obstáculo frontal
        front_obstacle = self.detect_front_obstacle(ranges)
        
        # Verificar si debe cambiar de pared
        if self.should_switch_wall(ranges):
            self.wall_side = "left" if self.wall_side == "right" else "right"
            self.last_wall_switch = time.time()
            self.integral = 0.0  # Reset del integral
            self.get_logger().info(f"Switching to {self.wall_side.upper()} wall following")
        
        # Obtener distancias de la pared
        wall_distance = self.get_wall_distance(ranges)
        lookahead_distance = self.get_lookahead_distance(ranges)
        
        # Preparar mensaje
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        
        if front_obstacle:
            # Obstáculo frontal - girar hacia el lado con más espacio
            center_idx = len(ranges) // 2
            left_space = np.mean(ranges[:center_idx])
            right_space = np.mean(ranges[center_idx:])
            
            if left_space > right_space:
                steering_angle = np.radians(25)
                self.wall_side = "right"  # Después del giro, seguir pared derecha
            else:
                steering_angle = np.radians(-25)
                self.wall_side = "left"   # Después del giro, seguir pared izquierda
                
            speed = 1.5
            self.get_logger().info(f"Front obstacle! Turning {'LEFT' if left_space > right_space else 'RIGHT'}")
            
        else:
            # Wall following normal
            steering_angle = self.calculate_wall_following_steering(wall_distance, lookahead_distance)
            speed = self.calculate_speed(ranges, steering_angle)
        
        drive_msg.drive.speed = float(speed)
        drive_msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(drive_msg)
        
        # Debug info
        if self.scan_count % 20 == 0:  # Cada 2 segundos aprox
            self.get_logger().info(f"Wall: {self.wall_side} | Dist: {wall_distance:.2f}m | Speed: {speed:.1f}")

    def odom_callback(self, msg):
        """Callback de odometría simplificado"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Zona de inicio
        in_start_zone = (abs(x) < 2.0 and abs(y) < 2.0)

        if in_start_zone and not self.prev_in_start_zone and self.has_left_start_zone:
            lap_end_time = time.time()
            lap_duration = lap_end_time - self.lap_start_time
            self.lap_times.append(lap_duration)
            self.lap_count += 1

            self.get_logger().info(f"🏁 Vuelta {self.lap_count} completada en {lap_duration:.2f} s.")
            
            if self.lap_times:
                best_time = min(self.lap_times)
                self.get_logger().info(f"Mejor tiempo: {best_time:.2f}s")
            
            self.lap_start_time = lap_end_time

        if not in_start_zone:
            self.has_left_start_zone = True

        self.prev_in_start_zone = in_start_zone


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped cleanly")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()