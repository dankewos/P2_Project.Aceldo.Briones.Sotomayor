#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Quaternion
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime
import math

def yaw_from_quat(q: Quaternion) -> float:
    # Convierte quaternion a yaw (Z) en radianes
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

class F1Logger(Node):
    def __init__(self):
        super().__init__('f1_logger')

        # Suscripciones
        self.sub_odom  = self.create_subscription(Odometry, '/ego_racecar/odom',  self.on_odom, 10)
        self.sub_drive = self.create_subscription(AckermannDriveStamped, '/drive', self.on_drive, 10)


        # Buffers
        self.t        = []
        self.x        = []
        self.y        = []
        self.yaw      = []
        self.v_linear = []
        self.steer    = []

        self.last_drive_msg = None
        self.start_time = self.get_clock().now()

        # Parámetro opcional para nombre de salida
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.out_csv = f'f1tenth_run_{stamp}.csv'
        self.get_logger().info('F1Logger listo. Grabando /odom y /drive...  (Ctrl+C para terminar)')

    def now_secs(self):
        return (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

    def on_odom(self, msg: Odometry):
        t = self.now_secs()
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q)

        # Velocidad lineal aproximada desde odom (vector en base local)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        v = math.hypot(vx, vy)

        # Último steering recibido (si hay)
        steer = self.last_drive_msg.drive.steering_angle if self.last_drive_msg else float('nan')

        # Guardar
        self.t.append(t)
        self.x.append(p.x)
        self.y.append(p.y)
        self.yaw.append(yaw)
        self.v_linear.append(v)
        self.steer.append(steer)

    def on_drive(self, msg: AckermannDriveStamped):
        self.last_drive_msg = msg

    def destroy_node(self):
        # Guardar CSV
        try:
            with open(self.out_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['t[s]','x[m]','y[m]','yaw[rad]','v[m/s]','steering[rad]'])
                for i in range(len(self.t)):
                    w.writerow([self.t[i], self.x[i], self.y[i], self.yaw[i], self.v_linear[i], self.steer[i]])
            self.get_logger().info(f'CSV guardado: {os.path.abspath(self.out_csv)}')
        except Exception as e:
            self.get_logger().error(f'No se pudo guardar CSV: {e}')

        # Graficar
        if len(self.t) > 5:
            try:
                fig1 = plt.figure(figsize=(6,6))
                plt.plot(self.x, self.y)
                plt.axis('equal')
                plt.xlabel('x [m]'); plt.ylabel('y [m]')
                plt.title('Trayectoria (x,y)')

                fig2 = plt.figure(figsize=(7,4))
                plt.plot(self.t, self.steer)
                plt.xlabel('tiempo [s]'); plt.ylabel('steering [rad]')
                plt.title('Ángulo de dirección vs tiempo')

                fig3 = plt.figure(figsize=(7,4))
                plt.plot(self.t, self.v_linear)
                plt.xlabel('tiempo [s]'); plt.ylabel('velocidad [m/s]')
                plt.title('Velocidad lineal vs tiempo')

                plt.tight_layout()
                plt.show()
            except Exception as e:
                self.get_logger().error(f'No se pudo graficar: {e}')

        super().destroy_node()

def main():
    rclpy.init()
    node = F1Logger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
