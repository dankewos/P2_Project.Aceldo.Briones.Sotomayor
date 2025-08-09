import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LapTimeLogger(Node):
    def __init__(self):
        super().__init__('timer_node')
        
        self.lap_count = 0
        self.best_lap_time_s = 9999.0
        self.last_trigger_time = None
        self.clock = self.get_clock()

        self.subscription = self.create_subscription(String,'/lap_trigger',self.lap_callback,10)
        

    def lap_callback(self, msg: String):
        current_time = self.clock.now()

        if self.last_trigger_time is None:
            self.last_trigger_time = current_time
            self.get_logger().info("Carrera iniciada")
            return

        lap_duration = current_time - self.last_trigger_time
        lap_time_s = lap_duration.nanoseconds / 1e9
        
        self.lap_count += 1

        if lap_time_s < self.best_lap_time_s:
            self.best_lap_time_s = lap_time_s
            best_lap_str = f"Vuelta completada en menor tiempo"
        else:
            best_lap_str = ""
        
        self.get_logger().info(f"Vuelta {self.lap_count} completada en" f": {lap_time_s:.4f}s | {best_lap_str}")
        
        if self.lap_count ==10:
            self.get_logger().info(f"5 vueltas completadas. Tiempo mejor vuelta:{self.best_lap_time_s:.2f}s")

        self.last_trigger_time = current_time

def main(args=None):
    rclpy.init(args=args)
    lap_logger_node = LapTimeLogger()
    rclpy.spin(lap_logger_node)
    lap_logger_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()