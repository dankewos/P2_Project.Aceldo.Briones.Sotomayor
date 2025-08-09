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

        # Parámetros optimizados del algoritmo
        self.base_speed = 6.0  # Velocidad base aumentada
        self.max_speed = 12.0  # Velocidad máxima aumentada para rectas
        self.min_speed = 2.0
        self.max_lidar_range = 10.0
        self.bubble_radius_m = 0.8  # Radio de burbuja reducido para ser más agresivo
        self.min_clearance = 1.2  # Clearance mínimo reducido
        self.max_steering_angle = np.radians(35)  # Ángulo máximo aumentado
        self.gap_threshold = 0.4  # Threshold reducido para detectar gaps más pequeños
        self.center_bias = 0.3  # Bias hacia el centro reducido para ser más flexible
        
        # Filtros adaptativos
        self.speed_filter = self.base_speed
        self.speed_filter_alpha = 0.6  # Más responsive
        self.steering_filter = 0.0
        self.steering_filter_alpha = 0.7  # Más responsive
        
        # Control de escaneado
        self.scan_initialized = False
        self.scan_count = 0
        
        # Variables para detección de curvas
        self.prev_steering = 0.0
        self.steering_history = []
        self.speed_history = []
        self.history_length = 10
        
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

        self.get_logger().info("Optimized ReactiveFollowGap node initialized.")

    def preprocess_lidar(self, ranges):
        """Preprocesamiento mejorado del LiDAR con filtrado adaptativo"""
        proc = np.array(ranges, dtype=np.float32)
        
        # Manejo de valores inválidos
        invalid_mask = (proc == 0) | np.isinf(proc) | np.isnan(proc)
        proc[invalid_mask] = self.max_lidar_range
        proc = np.clip(proc, 0.1, self.max_lidar_range)
        
        # Filtro de mediana para eliminar ruido de spikes
        from scipy.signal import medfilt
        proc = medfilt(proc, kernel_size=3)
        
        # Suavizado gaussiano adaptativo
        kernel = np.array([0.2, 0.6, 0.2])  # Filtro más agresivo
        proc = np.convolve(proc, kernel, mode='same')
        
        return proc

    def create_safety_bubble(self, ranges, closest_idx):
        """Burbuja de seguridad adaptativa basada en velocidad y distancia"""
        bubble_ranges = np.copy(ranges)
        
        if closest_idx is None or closest_idx >= len(ranges):
            return bubble_ranges
            
        closest_distance = ranges[closest_idx]
        
        # Burbuja adaptativa: más grande a mayor velocidad
        current_speed = getattr(self, 'current_speed', self.base_speed)
        speed_factor = min(current_speed / self.base_speed, 2.0)
        
        # Radio de burbuja adaptativo
        adaptive_radius = self.bubble_radius_m * speed_factor
        
        angle_increment = 2 * np.pi / len(ranges)
        bubble_radius_idx = int(adaptive_radius / (closest_distance * angle_increment + 0.01))
        bubble_radius_idx = max(3, min(bubble_radius_idx, 25))
        
        start_idx = max(0, closest_idx - bubble_radius_idx)
        end_idx = min(len(ranges) - 1, closest_idx + bubble_radius_idx)
        
        # Aplicar burbuja con degradado suave
        for i in range(start_idx, end_idx + 1):
            distance_from_center = abs(i - closest_idx)
            if distance_from_center <= bubble_radius_idx // 2:
                bubble_ranges[i] = 0.0
            else:
                # Degradado suave en los bordes
                fade_factor = (bubble_radius_idx - distance_from_center) / (bubble_radius_idx // 2)
                bubble_ranges[i] = min(bubble_ranges[i], closest_distance * fade_factor)
                
        return bubble_ranges

    def find_gaps(self, ranges):
        """Detección de gaps mejorada con análisis de conectividad"""
        # Usar clearance adaptativo basado en velocidad
        current_speed = getattr(self, 'current_speed', self.base_speed)
        adaptive_clearance = self.min_clearance * (1 + current_speed / self.max_speed)
        
        free_mask = ranges >= adaptive_clearance
        gaps = []
        start_idx = None
        
        for i, is_free in enumerate(free_mask):
            if is_free and start_idx is None:
                start_idx = i
            elif not is_free and start_idx is not None:
                gap_width_rad = (i - start_idx) * (2 * np.pi / len(ranges))
                gap_width_m = gap_width_rad * np.mean(ranges[start_idx:i])
                
                # Threshold adaptativo para gaps
                adaptive_threshold = self.gap_threshold * (1 - current_speed / self.max_speed * 0.3)
                
                if gap_width_m >= adaptive_threshold:
                    gaps.append((start_idx, i - 1))
                start_idx = None
                
        # Manejar gap que termina al final del array
        if start_idx is not None:
            gap_width_rad = (len(free_mask) - start_idx) * (2 * np.pi / len(ranges))
            gap_width_m = gap_width_rad * np.mean(ranges[start_idx:])
            adaptive_threshold = self.gap_threshold * (1 - current_speed / self.max_speed * 0.3)
            
            if gap_width_m >= adaptive_threshold:
                gaps.append((start_idx, len(free_mask) - 1))
                
        return gaps

    def select_best_gap(self, gaps, ranges):
        """Selección mejorada de gap con múltiples criterios"""
        if not gaps:
            return None
            
        best_gap = None
        best_score = -1
        center_idx = len(ranges) // 2
        
        for start_idx, end_idx in gaps:
            gap_ranges = ranges[start_idx:end_idx + 1]
            
            # Métricas del gap
            avg_distance = np.mean(gap_ranges)
            max_distance = np.max(gap_ranges)
            min_distance = np.min(gap_ranges)
            gap_width = end_idx - start_idx
            gap_center = (start_idx + end_idx) // 2
            center_distance = abs(gap_center - center_idx)
            
            # Scores normalizados
            distance_score = min(avg_distance / 6.0, 1.0)
            width_score = min(gap_width / 60.0, 1.0)
            consistency_score = min_distance / (avg_distance + 0.1)  # Consistencia del gap
            
            # Bias hacia el centro adaptativo
            adaptive_center_bias = self.center_bias * (1 - abs(self.steering_filter) / self.max_steering_angle)
            center_score = 1.0 - (center_distance / center_idx) * adaptive_center_bias
            
            # Score de profundidad (preferir gaps profundos)
            depth_score = min(max_distance / 8.0, 1.0)
            
            # Score total con pesos optimizados
            total_score = (distance_score * 0.3 + 
                          width_score * 0.25 + 
                          center_score * 0.2 + 
                          consistency_score * 0.15 + 
                          depth_score * 0.1)
            
            if total_score > best_score:
                best_score = total_score
                best_gap = (start_idx, end_idx)
                
        return best_gap

    def get_target_point(self, start_idx, end_idx, ranges):
        """Selección de punto objetivo optimizada"""
        if start_idx is None or end_idx is None:
            return None
            
        gap_ranges = ranges[start_idx:end_idx + 1]
        
        # Encontrar el punto más lejano
        max_dist_idx = np.argmax(gap_ranges)
        farthest_point = start_idx + max_dist_idx
        
        # Centro geométrico del gap
        gap_center = (start_idx + end_idx) // 2
        
        # Punto objetivo adaptativo basado en el ancho del gap
        gap_width = end_idx - start_idx
        
        if gap_width > 50:  # Gap muy ancho - ir hacia el punto más lejano
            target_idx = int(0.8 * farthest_point + 0.2 * gap_center)
        elif gap_width > 20:  # Gap medio - balancear
            target_idx = int(0.6 * farthest_point + 0.4 * gap_center)
        else:  # Gap estrecho - priorizar centro
            target_idx = int(0.4 * farthest_point + 0.6 * gap_center)
        
        return target_idx

    def calculate_steering_angle(self, target_idx, scan_data):
        """Cálculo de ángulo de dirección mejorado"""
        if target_idx is None:
            return 0.0
            
        target_angle = scan_data.angle_min + target_idx * scan_data.angle_increment
        center_angle = scan_data.angle_min + (len(scan_data.ranges) // 2) * scan_data.angle_increment
        
        # Ángulo de dirección base
        raw_steering = target_angle - center_angle
        
        # Factor de corrección basado en velocidad
        current_speed = getattr(self, 'current_speed', self.base_speed)
        speed_factor = 1.0 + (current_speed - self.base_speed) / self.max_speed * 0.3
        
        steering_angle = raw_steering * 0.9 * speed_factor
        
        # Filtro de steering más inteligente
        steering_diff = abs(steering_angle - self.steering_filter)
        if steering_diff > np.radians(15):  # Cambio brusco - usar filtrado más conservador
            alpha = 0.5
        else:  # Cambio suave - permitir más respuesta
            alpha = self.steering_filter_alpha
            
        steering_angle = alpha * self.steering_filter + (1 - alpha) * steering_angle
        self.steering_filter = steering_angle
        
        return np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

    def detect_track_section(self, ranges, steering_angle):
        """Detecta el tipo de sección de la pista"""
        center_idx = len(ranges) // 2
        front_range = int(np.radians(45) / (2 * np.pi / len(ranges)))
        
        # Analizar distancias frontales
        front_distances = ranges[max(0, center_idx - front_range):
                                min(len(ranges), center_idx + front_range)]
        avg_front = np.mean(front_distances)
        
        # Analizar distancias laterales
        left_range = ranges[:center_idx//2]
        right_range = ranges[-center_idx//2:]
        avg_left = np.mean(left_range)
        avg_right = np.mean(right_range)
        
        # Historial de steering para detectar curvas
        self.steering_history.append(abs(steering_angle))
        if len(self.steering_history) > self.history_length:
            self.steering_history.pop(0)
            
        avg_steering = np.mean(self.steering_history) if self.steering_history else 0
        
        # Clasificación de sección
        if avg_front > 6.0 and avg_steering < np.radians(10):
            return "straight"  # Recta larga
        elif avg_steering > np.radians(20):
            return "sharp_turn"  # Curva cerrada
        elif min(avg_left, avg_right) < 2.0:
            return "narrow"  # Sección estrecha
        else:
            return "normal"  # Sección normal

    def calculate_speed(self, ranges, steering_angle):
        """Cálculo de velocidad adaptativo mejorado"""
        center_idx = len(ranges) // 2
        
        # Analizar diferentes regiones
        front_range = int(np.radians(30) / (2 * np.pi / len(ranges)))
        wide_front_range = int(np.radians(60) / (2 * np.pi / len(ranges)))
        
        front_distances = ranges[max(0, center_idx - front_range):
                               min(len(ranges), center_idx + front_range)]
        wide_front_distances = ranges[max(0, center_idx - wide_front_range):
                                    min(len(ranges), center_idx + wide_front_range)]
        
        min_front = np.min(front_distances)
        avg_front = np.mean(front_distances)
        avg_wide_front = np.mean(wide_front_distances)
        
        # Detectar tipo de sección
        section_type = self.detect_track_section(ranges, steering_angle)
        
        # Factor base de velocidad según la sección
        if section_type == "straight":
            base_factor = 1.4  # Acelerar en rectas
        elif section_type == "sharp_turn":
            base_factor = 0.6  # Reducir en curvas cerradas
        elif section_type == "narrow":
            base_factor = 0.8  # Cuidado en secciones estrechas
        else:
            base_factor = 1.0
        
        # Ajustes por distancia frontal
        if min_front < 0.8:
            distance_factor = 0.3  # Muy cerca - frenar fuerte
        elif min_front < 1.5:
            distance_factor = 0.5  # Cerca - reducir velocidad
        elif avg_front > 5.0 and avg_wide_front > 4.0:
            distance_factor = 1.3  # Camino libre - acelerar
        elif avg_front > 3.0:
            distance_factor = 1.1  # Camino relativamente libre
        else:
            distance_factor = 0.9
        
        # Factor por ángulo de dirección (menos penalización)
        angle_factor = 1.0 - (abs(steering_angle) / self.max_steering_angle) * 0.3
        
        # Calcular velocidad objetivo
        target_speed = self.base_speed * base_factor * distance_factor * angle_factor
        target_speed = np.clip(target_speed, self.min_speed, self.max_speed)
        
        # Filtrado adaptativo de velocidad
        speed_diff = abs(target_speed - self.speed_filter)
        if speed_diff > 2.0:  # Cambio grande - filtrar más
            alpha = 0.4
        else:  # Cambio pequeño - ser más responsive
            alpha = self.speed_filter_alpha
            
        self.speed_filter = alpha * self.speed_filter + (1 - alpha) * target_speed
        self.current_speed = self.speed_filter  # Para usar en otras funciones
        
        return self.speed_filter

    def emergency_brake_check(self, ranges):
        """Verificación de frenado de emergencia"""
        center_idx = len(ranges) // 2
        emergency_range = int(np.radians(20) / (2 * np.pi / len(ranges)))
        
        emergency_distances = ranges[max(0, center_idx - emergency_range):
                                   min(len(ranges), center_idx + emergency_range)]
        
        min_emergency = np.min(emergency_distances)
        
        if min_emergency < 0.5:  # Obstáculo muy cerca
            return True, 0.5  # Frenar inmediatamente
        elif min_emergency < 1.0:
            return True, self.min_speed * 0.8  # Reducir velocidad drásticamente
        
        return False, None

    def lidar_callback(self, data):
        """Callback principal optimizado"""
        self.scan_count += 1
        if self.scan_count < 3:  # Reducir tiempo de inicialización
            return

        # Preprocesamiento
        ranges = self.preprocess_lidar(data.ranges)
        
        # Verificar frenado de emergencia
        emergency, emergency_speed = self.emergency_brake_check(ranges)
        
        if emergency:
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = "base_link"
            drive_msg.drive.speed = float(emergency_speed)
            drive_msg.drive.steering_angle = float(self.steering_filter * 0.5)  # Mantener dirección suave
            self.drive_pub.publish(drive_msg)
            return
        
        # Procesamiento normal
        closest_idx = np.argmin(ranges)
        closest_distance = ranges[closest_idx]

        # Crear burbuja de seguridad si es necesario
        if closest_distance < 2.0:
            bubble_ranges = self.create_safety_bubble(ranges, closest_idx)
        else:
            bubble_ranges = ranges

        # Encontrar gaps y seleccionar el mejor
        gaps = self.find_gaps(bubble_ranges)
        best_gap = self.select_best_gap(gaps, ranges)

        # Preparar mensaje
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
                steering_angle = self.steering_filter * 0.8
        else:
            # No hay gaps disponibles - comportamiento conservador
            speed = self.min_speed * 0.8
            # Girar hacia el lado con más espacio
            center_idx = len(ranges) // 2
            left_space = np.mean(ranges[:center_idx])
            right_space = np.mean(ranges[center_idx:])
            
            if left_space > right_space:
                steering_angle = np.radians(15)
            else:
                steering_angle = np.radians(-15)

        drive_msg.drive.speed = float(speed)
        drive_msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(drive_msg)

    def odom_callback(self, msg):
        """Callback de odometría para conteo de vueltas"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Zona de inicio (ajustar según la pista)
        in_start_zone = (abs(x) < 2.0 and abs(y) < 2.0)

        if in_start_zone and not self.prev_in_start_zone and self.has_left_start_zone:
            lap_end_time = time.time()
            lap_duration = lap_end_time - self.lap_start_time
            self.lap_times.append(lap_duration)
            self.lap_count += 1

            self.get_logger().info(f"🏁 Vuelta {self.lap_count} completada en {lap_duration:.2f} s.")
            
            if self.lap_times:
                best_time = min(self.lap_times)
                avg_time = np.mean(self.lap_times)
                self.get_logger().info(f"Mejor tiempo: {best_time:.2f}s | Promedio: {avg_time:.2f}s")
            
            self.lap_start_time = lap_end_time

            if self.lap_count >= 10:
                self.get_logger().info(f"¡{self.lap_count} vueltas completadas!")
                best_time = min(self.lap_times)
                self.get_logger().info(f"Mejor tiempo de vuelta: {best_time:.2f} segundos")

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