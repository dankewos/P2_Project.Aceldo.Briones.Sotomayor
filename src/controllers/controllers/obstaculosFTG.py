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
        self.base_speed = 4.0  # Velocidad base para moverse lentamente
        self.max_speed = 5.0   # Velocidad máxima
        self.min_speed = 3.0   # Velocidad mínima
        self.max_lidar_range = 10
        self.bubble_radius_m = 2.0  # Aumentamos el radio de la burbuja de seguridad
        self.min_clearance = 1.6    # Mayor clearance para buscar caminos más anchos
        self.max_steering_angle = np.radians(30)
        self.gap_threshold = 0.3    # Umbral reducido para aceptar gaps más estrechos
        self.center_bias = 0.5
        self.speed_filter = self.base_speed
        self.speed_filter_alpha = 0.7
        self.steering_filter = 0.0
        self.steering_filter_alpha = 0.8
        self.scan_initialized = False
        self.scan_count = 0

        # Control PID
        self.kp_angle = 15  # Ganancia proporcional para el ángulo
        self.ki_angle = 0.0  # Ganancia integral para el ángulo
        self.kd_angle = 0.1  # Ganancia derivativa para el ángulo

        self.kp_speed = 0.02  # Ganancia proporcional para la velocidad
        self.ki_speed = 0.0  # Ganancia integral para la velocidad
        self.kd_speed = 0.1  # Ganancia derivativa para la velocidad

        self.last_error_angle = 0.0
        self.integral_angle = 0.0
        self.last_error_speed = 0.0
        self.integral_speed = 0.0

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
        #Preprocesa los datos del LiDAR: reemplaza valores inválidos y aplica un filtro
        proc = np.array(ranges, dtype=np.float32)
        invalid_mask = (proc == 0) | np.isinf(proc) | np.isnan(proc)
        proc[invalid_mask] = self.max_lidar_range
        proc = np.clip(proc, 0.1, self.max_lidar_range)
        kernel = np.ones(3) / 3
        return np.convolve(proc, kernel, mode='same')

    def create_safety_bubble(self, ranges, closest_idx):
        #Crea la burbuja de seguridad que ayuda a evitar obstaculos dentro del rango definido
        bubble_ranges = np.copy(ranges)
        angle_increment = 2 * np.pi / len(ranges)
        bubble_radius_idx = int(self.bubble_radius_m / (ranges[closest_idx] * angle_increment))
        bubble_radius_idx = max(5, min(bubble_radius_idx, 20))
        start_idx = max(0, closest_idx - bubble_radius_idx)
        end_idx = min(len(ranges) - 1, closest_idx + bubble_radius_idx)
        bubble_ranges[start_idx:end_idx + 1] = 0.0
        return bubble_ranges

    def find_gaps(self, ranges):
        #Detecta los huecos o gaps que el carrito puede usar para avanzar
        free_mask = ranges >= self.min_clearance
        gaps = []
        start_idx = None
        for i, is_free in enumerate(free_mask):
            if is_free and start_idx is None:
                start_idx = i
            elif not is_free and start_idx is not None:
                gap_width_rad = (i - start_idx) * (2 * np.pi / len(ranges))
                gap_width_m = gap_width_rad * np.mean(ranges[start_idx:i])
                if gap_width_m >= self.gap_threshold:  # Filtramos por el umbral de gap
                    gaps.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None:
            gap_width_rad = (len(free_mask) - start_idx) * (2 * np.pi / len(ranges))
            gap_width_m = gap_width_rad * np.mean(ranges[start_idx:])
            if gap_width_m >= self.gap_threshold:  # Filtramos por el umbral de gap
                gaps.append((start_idx, len(free_mask) - 1))
        return gaps

    def select_best_gap(self, gaps, ranges):
        #Selecciona el mejor, el mas largo con espacio para cruzar, prioriza el gap mas largo
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
            width_score = min(gap_width / 50.0, 1.0)  # Aumentamos la importancia del ancho
            center_score = 1.0 - (center_distance / center_idx) * self.center_bias
            total_score = (distance_score * 0.2 + width_score * 0.6 + center_score * 0.2)  # Ajustamos peso
            if total_score > best_score:
                best_score = total_score
                best_gap = (start_idx, end_idx)
        return best_gap

    def get_target_point(self, start_idx, end_idx, ranges):
        #Define un punto siguiente para que el carrito avance
        if start_idx is None or end_idx is None:
            return None
        gap_ranges = ranges[start_idx:end_idx + 1]
        max_dist_idx = np.argmax(gap_ranges)
        target_idx = start_idx + max_dist_idx
        gap_center = (start_idx + end_idx) // 2
        target_idx = gap_center  # Centramos el robot en el gap más ancho
        return target_idx

    def calculate_pid(self, error, last_error, integral, kp, ki, kd):
        #Calcula el pid
        integral += error
        derivative = error - last_error
        output = kp * error + ki * integral + kd * derivative
        return output, integral

    def calculate_steering_angle(self, target_idx, scan_data):
        #Obtiene el steering angle para acercarse al punto objetivo definido
        if target_idx is None:
            return 0.0
        target_angle = scan_data.angle_min + target_idx * scan_data.angle_increment
        center_angle = scan_data.angle_min + (len(scan_data.ranges) // 2) * scan_data.angle_increment
        error_angle = target_angle - center_angle

        # Aumentamos la ganancia proporcional si se detecta un obstáculo cerca
        if abs(error_angle) > np.radians(10):  # Si el error angular es grande, aumenta el giro
            self.kp_angle = 1.5  # Aumentamos la ganancia proporcional para giros más rápidos
        else:
            self.kp_angle = 0.8  # Restablecemos la ganancia a su valor original si no hay obstáculo cercano

        # Calcular el control PID para el ángulo
        steering_angle, self.integral_angle = self.calculate_pid(error_angle, self.last_error_angle, self.integral_angle, self.kp_angle, self.ki_angle, self.kd_angle)
        self.last_error_angle = error_angle

        # Limitar el ángulo de dirección para evitar giros bruscos
        return np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

    def calculate_speed(self, ranges, steering_angle):
        #Adapta la velocidad adecuada del carrito
        center_idx = len(ranges) // 2
        front_range = int(np.radians(30) / (2 * np.pi / len(ranges)))
        front_distances = ranges[max(0, center_idx - front_range):min(len(ranges), center_idx + front_range)]
        min_front_distance = np.min(front_distances)
        avg_front_distance = np.mean(front_distances)

        # Ajuste de la velocidad con PID para evitar retrocesos
        speed_error = avg_front_distance - 2.0  # Queremos mantener al menos 2 metros de distancia
        speed, self.integral_speed = self.calculate_pid(speed_error, self.last_error_speed, self.integral_speed, self.kp_speed, self.ki_speed, self.kd_speed)
        self.last_error_speed = speed_error

        # Limitar la velocidad y asegurarse que no retroceda
        speed = max(self.min_speed, speed)  # Aseguramos que la velocidad no sea menor a un mínimo
        return np.clip(speed, self.min_speed, self.max_speed)

    def lidar_callback(self, data):
        #Llama la informacion del lidar
        self.scan_count += 1
        if self.scan_count < 5:
            return

        ranges = self.preprocess_lidar(data.ranges)
        closest_idx = np.argmin(ranges)
        closest_distance = ranges[closest_idx]

        # Crear burbuja de seguridad solo si es necesario
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
        #Llama la informacion de odometria del carrito
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Zona de inicio
        in_start_zone = (abs(x) < 2.0 and abs(y) < 2.0)

        #Contador de vueltas
        if in_start_zone and not self.prev_in_start_zone and self.has_left_start_zone:
            lap_end_time = time.time()
            lap_duration = lap_end_time - self.lap_start_time
            self.lap_times.append(lap_duration)
            self.lap_count += 1

            self.get_logger().info(f"🏁 Vuelta {self.lap_count} completada en {lap_duration:.2f} s.")
            self.lap_start_time = lap_end_time


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