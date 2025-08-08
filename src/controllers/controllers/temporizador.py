import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String


class LapTracker(Node):
    def __init__(self):
        super().__init__('lap_time_node')

        # Parámetro para número total de vueltas (por defecto 10)
        self.declare_parameter('total_laps', 10)
        self.total_laps = int(self.get_parameter('total_laps').value)

        # Estado
        self._laps_done = 0
        self._best_lap_secs = float('inf')
        self._last_stamp = None
        self._clock = self.get_clock()

        # Suscripción al trigger de vuelta
        qos = QoSProfile(depth=10)
        self._sub = self.create_subscription(
            String, '/lap_trigger', self._on_trigger, qos
        )

        self.get_logger().info('LapTracker listo. Esperando primer trigger...')

    def _on_trigger(self, _msg: String):
        now = self._clock.now()

        # Primer trigger: inicia cronometraje
        if self._last_stamp is None:
            self._last_stamp = now
            self.get_logger().info('🏁 Carrera iniciada')
            return

        # Duración de la vuelta
        dt = now - self._last_stamp
        lap_secs = dt.nanoseconds / 1e9
        self._laps_done += 1

        # ¿Mejor vuelta?
        new_record = lap_secs < self._best_lap_secs
        if new_record:
            self._best_lap_secs = lap_secs

        tag = 'MEJOR VUELTA' if new_record else ''
        self.get_logger().info(
            f'Vuelta {self._laps_done} completada en {lap_secs:.4f}s {tag}'
        )

        # ¿Alcanzó el total?
        if self._laps_done == self.total_laps:
            self.get_logger().info(
                f'✅ {self.total_laps} vueltas completadas | '
                f'Mejor vuelta: {self._best_lap_secs:.2f}s'
            )

        # Preparar siguiente medición
        self._last_stamp = now


def main(args=None):
    rclpy.init(args=args)
    node = LapTracker()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
