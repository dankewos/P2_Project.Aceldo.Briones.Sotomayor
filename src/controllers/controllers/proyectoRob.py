import rclpy
from rclpy.node import Node
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String
from nav_msgs.msg import Odometry

class FollowGapCarrera(Node):
    def __init__(self):
        super().__init__('follow_gap_carrera_node')
        
        self.VELOCIDAD_MAXIMA = 8.2
        self.VELOCIDAD_MINIMA = 3.5
        self.FACTOR_REDUCCION_CURVA = 1.2 
        self.DISTANCIA_MAX_LIDAR = 3.5
        self.RADIO_BURBUJA = 0.25
        self.ANGULO_MAX_DIRECCION = np.radians(20)
        self.PUNTO_LEJANO_BIAS = 0.8
        self.SUAVIZADO_DIRECCION = 0.9 
        self.ultimo_angulo_direccion = 0.0

        self.FINISH_LINE_X = 0.00
        self.FINISH_LINE_Y = 0.00
        self.FINISH_LINE_Y_RANGE = 5.0
        self.MIN_DISTANCE_TO_ARM = 0.2

        self.last_pose = None
        self.can_count_lap = True
        self.race_started = True

        self.lap_publisher = self.create_publisher(String, '/lap_trigger', 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        msg_out = String()
        msg_out.data = "Nueva vuelta"
        self.lap_publisher.publish(msg_out)

        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        

    def odom_callback(self, msg: Odometry):
        current_pose = msg.pose.pose

        if self.last_pose is None:
            self.last_pose = current_pose
            return

        last_x = self.last_pose.position.x
        current_x = current_pose.position.x
        current_y = current_pose.position.y

        crossed_finish_line = (last_x < self.FINISH_LINE_X and current_x >= self.FINISH_LINE_X)
        on_finish_straight = abs(current_y - self.FINISH_LINE_Y) < self.FINISH_LINE_Y_RANGE
        
        if crossed_finish_line and on_finish_straight and self.can_count_lap:
            msg_out = String()
            msg_out.data = "Nueva vuelta"
            self.lap_publisher.publish(msg_out)

            self.can_count_lap = False

        if not self.can_count_lap:
            dist_from_finish = math.sqrt((current_pose.position.x - self.FINISH_LINE_X)**2)
            if dist_from_finish > self.MIN_DISTANCE_TO_ARM:
                self.can_count_lap = True

        self.last_pose = current_pose


    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[np.isinf(ranges) | np.isnan(ranges)] = self.DISTANCIA_MAX_LIDAR
        ranges = np.clip(ranges, 0.0, self.DISTANCIA_MAX_LIDAR)
        
        idx_closest = np.argmin(ranges)
        dist_closest = ranges[idx_closest]
        
        if dist_closest > 0.1:
            angle_span = 2 * np.arctan2(self.RADIO_BURBUJA, dist_closest)
            bubble_radius_idx = int(angle_span / msg.angle_increment)
        else:
            bubble_radius_idx = 30
        start_idx = max(0, idx_closest - bubble_radius_idx)
        end_idx = min(len(ranges) - 1, idx_closest + bubble_radius_idx)
        ranges[start_idx : end_idx + 1] = 0.0

        max_len = 0
        best_start_idx = 0
        best_end_idx = 0
        current_len = 0
        current_start_idx = 0
        for i in range(len(ranges)):
            if ranges[i] > 0.1:
                if current_len == 0:
                    current_start_idx = i
                current_len += 1
            else:
                if current_len > max_len:
                    max_len = current_len
                    best_start_idx = current_start_idx
                    best_end_idx = i - 1
                current_len = 0
        if current_len > max_len:
            max_len = current_len
            best_start_idx = current_start_idx
            best_end_idx = len(ranges) - 1
            
        drive_msg = AckermannDriveStamped()
        
        if max_len > 0:
            gap_ranges = ranges[best_start_idx : best_end_idx + 1]
            idx_punto_lejano_relativo = np.argmax(gap_ranges)
            idx_punto_lejano_absoluto = best_start_idx + idx_punto_lejano_relativo
            idx_centro_gap = (best_start_idx + best_end_idx) // 2
            target_idx = int(self.PUNTO_LEJANO_BIAS * idx_punto_lejano_absoluto + (1 - self.PUNTO_LEJANO_BIAS) * idx_centro_gap)
            target_angle = msg.angle_min + target_idx * msg.angle_increment
            
            steering_angle = self.SUAVIZADO_DIRECCION * self.ultimo_angulo_direccion + (1 - self.SUAVIZADO_DIRECCION) * target_angle
            self.ultimo_angulo_direccion = steering_angle
            steering_angle = np.clip(steering_angle, -self.ANGULO_MAX_DIRECCION, self.ANGULO_MAX_DIRECCION)
            
            severidad_curva = abs(target_angle) / self.ANGULO_MAX_DIRECCION
            
            severidad_curva = np.clip(severidad_curva, 0.0, 1.0)
            
            factor_reduccion = (1.0 - severidad_curva) ** self.FACTOR_REDUCCION_CURVA
            speed = self.VELOCIDAD_MINIMA + (self.VELOCIDAD_MAXIMA - self.VELOCIDAD_MINIMA) * factor_reduccion
            
            drive_msg.drive.steering_angle = float(steering_angle)
            drive_msg.drive.speed = float(speed)
        else:
            drive_msg.drive.steering_angle = 0.0
            drive_msg.drive.speed = 0.0
            self.ultimo_angulo_direccion = 0.0
        
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FollowGapCarrera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()