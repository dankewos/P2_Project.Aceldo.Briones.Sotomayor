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

        # Parámetros AGRESIVOS para pasillos estrechos
        self.base_speed = 4.0  # Velocidad base más conservadora para control
        self.max_speed = 8.0   # Velocidad máxima reducida para mejor control
        self.min_speed = 1.0   # Velocidad mínima más baja
        self.max_lidar_range = 10.0
        self.bubble_radius_m = 0.3  # Radio muy pequeño - MUY AGRESIVO
        self.min_clearance = 0.6    # Clearance mínimo muy reducido - AGRESIVO
        self.max_steering_angle = np.radians(45)  # Ángulo máximo aumentado
        self.gap_threshold = 0.15   # Threshold muy pequeño para gaps estrechos
        self.center_bias = 0.1      # Bias mínimo - seguir el gap más grande
        
        # Filtros más responsivos
        self.speed_filter = self.base_speed
        self.speed_filter_alpha = 0.3  # MUY responsive
        self.steering_filter = 0.0
        self.steering_filter_alpha = 0.4  # MUY responsive
        
        # Control de escaneado
        self.scan_initialized = False
        self.scan_count = 0
        
        # Variables para evitar bucles
        self.stuck_counter = 0
        self.position_history = []
        self.last_positions = []
        self.stuck_threshold = 20  # Número de scans para detectar si está atascado
        self.escape_mode = False
        self.escape_direction = 1  # 1 para derecha, -1 para izquierda
        self.escape_counter = 0
        
        # Variables para detección de progreso
        self.prev_position = None
        self.movement_threshold = 0.1  # Movimiento mínimo para no considerarse atascado
        
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

        self.get_logger().info("AGGRESSIVE ReactiveFollowGap initialized for narrow passages.")

    def preprocess_lidar(self, ranges):
        """Preprocesamiento mínimo para mantener detalles en pasillos estrechos"""
        proc = np.array(ranges, dtype=np.float32)
        
        # Manejo de valores inválidos
        invalid_mask = (proc == 0) | np.isinf(proc) | np.isnan(proc)
        proc[invalid_mask] = self.max_lidar_range
        proc = np.clip(proc, 0.05, self.max_lidar_range)  # Permitir lecturas muy cercanas
        
        # Filtrado MUY suave para preservar detalles
        kernel = np.array([0.1, 0.8, 0.1])  # Filtro muy suave
        proc = np.convolve(proc, kernel, mode='same')
        
        return proc

    def create_safety_bubble(self, ranges, closest_idx):
        """Burbuja de seguridad MÍNIMA - solo para obstáculos muy cercanos"""
        bubble_ranges = np.copy(ranges)
        
        if closest_idx is None or closest_idx >= len(ranges):
            return bubble_ranges
            
        closest_distance = ranges[closest_idx]
        
        # Solo crear burbuja si está MUY cerca
        if closest_distance > 0.4:
            return bubble_ranges
        
        # Burbuja muy pequeña
        angle_increment = 2 * np.pi / len(ranges)
        bubble_radius_idx = max(1, int(self.bubble_radius_m / (closest_distance * angle_increment + 0.01)))
        bubble_radius_idx = min(bubble_radius_idx, 5)  # Máximo 5 puntos
        
        start_idx = max(0, closest_idx - bubble_radius_idx)
        end_idx = min(len(ranges) - 1, closest_idx + bubble_radius_idx)
        
        # Burbuja sin degradado - directa
        bubble_ranges[start_idx:end_idx + 1] = 0.0
                
        return bubble_ranges

    def find_gaps(self, ranges):
        """Detección de gaps ULTRA AGRESIVA"""
        # Clearance muy pequeño
        free_mask = ranges >= self.min_clearance
        gaps = []
        start_idx = None
        
        for i, is_free in enumerate(free_mask):
            if is_free and start_idx is None:
                start_idx = i
            elif not is_free and start_idx is not None:
                # Cualquier gap, por pequeño que sea
                gaps.append((start_idx, i - 1))
                start_idx = None
                
        # Manejar gap que termina al final del array
        if start_idx is not None:
            gaps.append((start_idx, len(free_mask) - 1))
                
        return gaps

    def select_best_gap(self, ranges):
        """Selección de gap SIMPLIFICADA - el más grande y más lejano"""
        gaps = self.find_gaps(ranges)
        
        if not gaps:
            return None
            
        best_gap = None
        best_score = -1
        
        for start_idx, end_idx in gaps:
            gap_width = end_idx - start_idx + 1
            gap_ranges = ranges[start_idx:end_idx + 1]
            avg_distance = np.mean(gap_ranges)
            
            # Score simple: tamaño del gap * distancia promedio
            score = gap_width * avg_distance
            
            if score > best_score:
                best_score = score
                best_gap = (start_idx, end_idx)
                
        return best_gap

    def get_target_point(self, start_idx, end_idx, ranges):
        """Punto objetivo en el CENTRO del gap más profundo"""
        if start_idx is None or end_idx is None:
            return None
            
        gap_ranges = ranges[start_idx:end_idx + 1]
        
        # Encontrar el punto más lejano en el gap
        max_dist_idx = np.argmax(gap_ranges)
        target_idx = start_idx + max_dist_idx
        
        return target_idx

    def calculate_steering_angle(self, target_idx, scan_data):
        """Cálculo de ángulo DIRECTO y AGRESIVO"""
        if target_idx is None:
            return 0.0
            
        target_angle = scan_data.angle_min + target_idx * scan_data.angle_increment
        center_angle = scan_data.angle_min + (len(scan_data.ranges) // 2) * scan_data.angle_increment
        
        # Ángulo directo sin mucho filtrado
        steering_angle = (target_angle - center_angle) * 1.2  # Factor agresivo
        
        # Filtrado mínimo
        steering_angle = (self.steering_filter_alpha * self.steering_filter +
                          (1 - self.steering_filter_alpha) * steering_angle)
        self.steering_filter = steering_angle
        
        return np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

    def check_if_stuck(self, current_position):
        """Detecta si el vehículo está atascado en un bucle"""
        if current_position is None:
            return False
            
        # Agregar posición actual al historial
        self.position_history.append(current_position)
        
        # Mantener solo las últimas posiciones
        if len(self.position_history) > self.stuck_threshold:
            self.position_history.pop(0)
            
        # Si no tenemos suficiente historial, no está atascado
        if len(self.position_history) < self.stuck_threshold:
            return False
            
        # Calcular la varianza de posiciones
        positions = np.array(self.position_history)
        x_variance = np.var(positions[:, 0])
        y_variance = np.var(positions[:, 1])
        
        # Si la varianza es muy pequeña, está atascado
        total_variance = x_variance + y_variance
        stuck = total_variance < 0.5  # Threshold para considerar que está atascado
        
        if stuck:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            
        return self.stuck_counter > 5

    def escape_behavior(self, ranges):
        """Comportamiento de escape cuando está atascado"""
        center_idx = len(ranges) // 2
        
        # Analizar espacios a los lados
        left_quarter = ranges[:center_idx//2]
        right_quarter = ranges[-center_idx//2:]
        
        left_space = np.mean(left_quarter[left_quarter > 0.1])
        right_space = np.mean(right_quarter[right_quarter > 0.1])
        
        # Elegir la dirección con más espacio
        if left_space > right_space:
            self.escape_direction = 1  # Izquierda
            steering_angle = np.radians(35)
        else:
            self.escape_direction = -1  # Derecha  
            steering_angle = np.radians(-35)
            
        # Velocidad baja pero constante para el escape
        speed = 2.0
        
        self.get_logger().info(f"ESCAPE MODE: Going {'LEFT' if self.escape_direction == 1 else 'RIGHT'}")
        
        return speed, steering_angle

    def calculate_speed(self, ranges, steering_angle):
        """Cálculo de velocidad AGRESIVO para pasillos estrechos"""
        center_idx = len(ranges) // 2
        front_range = int(np.radians(20) / (2 * np.pi / len(ranges)))
        
        front_distances = ranges[max(0, center_idx - front_range):
                               min(len(ranges), center_idx + front_range)]
        
        min_front = np.min(front_distances)
        avg_front = np.mean(front_distances)
        
        # Velocidad agresiva basada en distancia frontal
        if min_front < 0.3:
            target_speed = 0.5  # Muy cerca - casi parar
        elif min_front < 0.6:
            target_speed = 1.5  # Cerca - velocidad baja
        elif min_front < 1.2:
            target_speed = 2.5  # Media distancia
        elif avg_front > 2.0:
            target_speed = self.max_speed  # Camino libre - acelerar
        else:
            target_speed = self.base_speed
        
        # Reducir velocidad menos por ángulo de dirección
        angle_penalty = (abs(steering_angle) / self.max_steering_angle) * 0.2
        target_speed *= (1.0 - angle_penalty)
        
        target_speed = np.clip(target_speed, self.min_speed, self.max_speed)
        
        # Filtrado muy responsive
        self.speed_filter = (self.speed_filter_alpha * self.speed_filter +
                             (1 - self.speed_filter_alpha) * target_speed)
        
        return self.speed_filter

    def lidar_callback(self, data):
        """Callback principal con manejo de situaciones de atasco"""
        self.scan_count += 1
        if self.scan_count < 2:  # Inicialización rápida
            return

        ranges = self.preprocess_lidar(data.ranges)
        
        # Preparar mensaje de drive
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        
        # Verificar si está en modo escape
        if self.escape_mode:
            speed, steering_angle = self.escape_behavior(ranges)
            self.escape_counter += 1
            
            # Salir del modo escape después de un tiempo
            if self.escape_counter > 30:  # ~3 segundos
                self.escape_mode = False
                self.escape_counter = 0
                self.stuck_counter = 0
                self.position_history.clear()
                self.get_logger().info("Exiting ESCAPE MODE")
                
        else:
            # Comportamiento normal
            closest_idx = np.argmin(ranges)
            closest_distance = ranges[closest_idx]

            # Burbuja de seguridad mínima
            if closest_distance < 0.3:
                bubble_ranges = self.create_safety_bubble(ranges, closest_idx)
            else:
                bubble_ranges = ranges

            # Encontrar el mejor gap
            best_gap = self.select_best_gap(bubble_ranges)

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
                # No hay gaps - buscar el lado con más espacio
                center_idx = len(ranges) // 2
                left_space = np.mean(ranges[:center_idx])
                right_space = np.mean(ranges[center_idx:])
                
                speed = 1.5
                if left_space > right_space:
                    steering_angle = np.radians(25)
                else:
                    steering_angle = np.radians(-25)
                    
                self.get_logger().info(f"No gaps found. Turning {'LEFT' if left_space > right_space else 'RIGHT'}")

        # Verificar frenado de emergencia
        center_idx = len(ranges) // 2
        emergency_range = int(np.radians(15) / (2 * np.pi / len(ranges)))
        emergency_distances = ranges[max(0, center_idx - emergency_range):
                                   min(len(ranges), center_idx + emergency_range)]
        
        if np.min(emergency_distances) < 0.2:  # Obstáculo muy muy cerca
            speed = 0.0
            self.get_logger().warn("EMERGENCY BRAKE!")

        drive_msg.drive.speed = float(speed)
        drive_msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(drive_msg)

    def odom_callback(self, msg):
        """Callback de odometría con detección de atasco"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        current_position = (x, y)
        
        # Verificar si está atascado
        if self.check_if_stuck(current_position) and not self.escape_mode:
            self.escape_mode = True
            self.escape_counter = 0
            self.get_logger().warn("STUCK DETECTED! Activating ESCAPE MODE")

        # Control de vueltas (zona de inicio ajustada)
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