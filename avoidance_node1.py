#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import threading
import sys
import select
import tty
import termios
import traceback
import signal
import math
import json
import cv2
from cv_bridge import CvBridge
import openai
import base64
import os

# Si no configuras la variable de entorno OPENAI_API_KEY, puedes ponerla aquí:
# openai.api_key = "TU_CLAVE_AQUI"

# Crear el cliente OpenAI con la clave del entorno
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None))

class ObstacleAvoidanceNode:
    def __init__(self):
        rospy.init_node('obstacle_avoidance_node', anonymous=False)
        rospy.loginfo("Inicializando ObstacleAvoidanceNode con análisis de imagen y API de OpenAI...")

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)

        self.laser_data = None
        self.current_heading = 0.0
        self.current_image = None

        self.rate = rospy.Rate(10)  # 10 Hz

        self.paused = False
        self.running = True

        self.forward_speed = 0.3
        self.max_turn_speed = 0.5
        self.safety_distance = 0.3

        self.desired_distance_left = 0.6
        self.desired_distance_right = 0.6

        self.Kp = 0.8

        self.min_valid_range = 0.12
        self.max_valid_range = 3.5

        self.lock = threading.Lock()

        self.state = 'MOVE_FORWARD'

        self.input_thread = threading.Thread(target=self.keyboard_listener)
        self.input_thread.daemon = True
        self.input_thread.start()

        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)

        self.bridge = CvBridge()

        # Aquí se almacenará la acción resultante del API
        self.api_response = {"actions": []}
        self.last_image_processed = None

        # Historia de las últimas tres imágenes y sus respuestas
        # Cada elemento será un dict: {"image_base64": str, "response": dict}
        self.history = []

    def laser_callback(self, data):
        with self.lock:
            self.laser_data = data

    def odom_callback(self, data):
        orientation_q = data.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.current_heading = yaw

    def image_callback(self, img_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            self.current_image = cv_image
        except Exception as e:
            rospy.logerr("Error al convertir la imagen: %s", e)

    def keyboard_listener(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self.running and not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == 'q':
                        self.paused = not self.paused
                        rospy.loginfo("Robot %s.", "pausado" if self.paused else "reanudado")
                    elif key == 'x':
                        rospy.loginfo("Saliendo del programa.")
                        self.running = False
                        rospy.signal_shutdown("El usuario solicitó la terminación.")
                        break
        except Exception as e:
            rospy.logerr("Error en keyboard_listener: %s", e)
            traceback.print_exc()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def control_loop(self):
        while not rospy.is_shutdown() and self.running:
            if self.paused:
                twist = Twist()
                self.cmd_vel_pub.publish(twist)
                rospy.loginfo_throttle(1, "Robot está pausado. Presiona 'q' para reanudar.")
                self.rate.sleep()
                continue

            if self.laser_data is None or self.current_image is None:
                rospy.loginfo_throttle(2, "Esperando a recibir datos de LIDAR o la primera imagen...")
                self.rate.sleep()
                continue

            # Obtener distancias del LIDAR en los ángulos de interés
            ranges = np.array(self.laser_data.ranges)
            angles = np.linspace(self.laser_data.angle_min, self.laser_data.angle_max, len(ranges))
            distancia_0 = round(self.get_distance_at_angle(ranges, angles, 0), 3)
            distancia_15 = round(self.get_distance_at_angle(ranges, angles, 15), 3)
            distancia_neg_15 = round(self.get_distance_at_angle(ranges, angles, 345), 3)
            distancia_45 = round(self.get_distance_at_angle(ranges, angles, 45), 3)
            distancia_neg_45 = round(self.get_distance_at_angle(ranges, angles, 315), 3)
            distancia_90 = round(self.get_distance_at_angle(ranges, angles, 90), 3)
            distancia_neg_90 = round(self.get_distance_at_angle(ranges, angles, 270), 3)
            # Crear contexto para el prompt con las distancias obtenidas
            system_context = (
                f"Eres un TurtleBot con dimensiones cuadradas de 12 x 12 cm y una altura aproximada de 22 cm, Además, viajas a una velocidad de 0.22 metros/segundos. Estás en una pista de carreras y se te va a compartir datos sensoriales de profundidad provenientes de un Lidar. Basándote en los datos, debes tomar la siguiente acción para evitar colisiones y avanzar de manera segura:"
                f"""
                 Acciones:
                 1. Avanzar hacia delante: X metros.
                 2. Girar a la derecha: X grados.
                 3. Girar a la izquierda: X grados."""
                f"""
                Datos del Lidar:
                - Distancia a 0 grados: {distancia_0} metros
                - Distancia a 15 grados a la derecha: {distancia_15} metros
                - Distancia a 45 grados a la derecha: {distancia_45} metros
                - Distancia a 90 grados a la derecha: {distancia_90} metros
                - Distancia a 345 grados a la izquierda: {distancia_neg_15} metros
                - Distancia a 315 grados a la izquierda: {distancia_neg_45} metros
                - Distancia a 270 grados a la izquierda: {distancia_neg_90} metros
                """
                f"""
                Instrucciones para la acción:
                Formato del output: Devuelve solo una acción en formato JSON con la siguiente estructura:
                {
                    "action" : "<action>",
                    "angle" : <angle>,
                    "distance" : <distance>,
                    "duration": <duration>
                }
                Ejemplo:
                {
                  "action": "FORWARD",
                  "angle": 0,
                  "distance": 1.0,
                  "duration": 5
                }
                {
                  "action": "TURN LEFT",
                  "angle": 30,
                  "distance": 0.0,
                  "duration": 2
                }
                {
                  "action": "TURN RIGHT",
                  "angle": 30,
                  "distance": 0.0,
                  "duration": 2
                } 
                """
                f"""
                Recuerda:
                - Solo debes retonar una acción
                - Asegúrate de que la acción sea segura y evite colisiones
                - Si hay suficiente espacio para avanzar hacia adelante, hazlo.
                """
                f"""
                Ejemplo de datos:
                Datos del lidar de un Turtlebot3:
                - Distancia a 0 grados: 1.5 metros
                - Distancia a 15 grados a la derecha: 1.15 metros
                - Distancia a 45 grados a la derecha: 0.6 metros
                - Distancia a 90 grados a la derecha: 0.4 metros
                - Distancia a 345 grados a la izquierda: 0.92 metros
                - Distancia a 315 grados a la izquierda: 0.45 metros
                - Distancia a 270 grados a la izquierda: 0.2 metros
                Razonamiento:
                Si el robot quiere avanzar 1.3 metro hacia adelante:
                - La distancia a 0 grados es 1.5 metros, que es suficiente para avanzar 1.3 metro.
                - Las distancias a 15 grados (1.15 metros), 45 grados (0.6 metros) y 345 grados (0.92 metros) también son razonablemente grandes, lo que indica que hay espacio suficiente para el movimiento.
                En este caso, el robot puede avanzar 1.3 metro hacia adelante sin chocar la pared que se encuentra a unos 0.2 metros hacia adelante.
                Output:
                {
                  "action": "FORWARD",
                  "angle": 0,
                  "distance": 1.0,
                  "duration": 5
                }
                """
            )

            if self.current_image is not None:
                if self.last_image_processed is not self.current_image:
                    rospy.loginfo("Enviando imagen al API para su análisis...")
                    try:
                        self.api_response = self.analyze_image_with_api(self.current_image, system_context)
                    except Exception as e:
                        rospy.logerr("Error inesperado en analyze_image_with_api: %s", e)
                        traceback.print_exc()
                        self.api_response = {"actions": []}

                    self.last_image_processed = self.current_image
                    rospy.loginfo("Respuesta recibida del API: %s", json.dumps(self.api_response))
            else:
                self.api_response = {"actions": []}
                rospy.loginfo_throttle(2, "Esperando a recibir la primera imagen...")

            # Si no hay acciones o la lista está vacía, detener
            if not self.api_response.get("action"):
                twist = Twist()
                self.cmd_vel_pub.publish(twist)
                self.rate.sleep()
                continue

            # Tomar la acción (única) de la lista
            movement_type = self.api_response.get("action", "stop")
            distance = self.api_response.get("distance", 0.0)
            angle = self.api_response.get("angle", 0.0)
            duration = self.api_response.get("duration", 1)

            rospy.logdebug("Acción a ejecutar: %s", movement_type)

            twist = Twist()
            if movement_type == "FORWARD":
                twist.linear.x = min(self.forward_speed, distance)
                twist.angular.z = 0.0
            elif movement_type == "RIGHT":
                twist.linear.x = 0.0
                twist.angular.z = -min(self.max_turn_speed, self.max_turn_speed * (angle / 90.0))
            elif movement_type == "LEFT":
                twist.linear.x = 0.0
                twist.angular.z = min(self.max_turn_speed, self.max_turn_speed * (angle / 90.0))
            else:
                # stop
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            end_time = rospy.Time.now() + rospy.Duration(duration)
            while rospy.Time.now() < end_time and not rospy.is_shutdown() and self.running:
                self.cmd_vel_pub.publish(twist)
                self.rate.sleep()

            # Después de ejecutar, eliminar la acción
            self.api_response["actions"].pop(0)

        self.shutdown()

    def analyze_image_with_api(self, cv_image, prompt_context):
        # Codificar la imagen a base64 según la documentación
        _, buffer = cv2.imencode('.jpg', cv_image)
        img_bytes = buffer.tobytes()
        base64_image = base64.b64encode(img_bytes).decode('utf-8')

        # Mensaje del sistema
        messages = [
            {"role": "system", "content": prompt_context},
            {"role": "user", "content": "Imagen proporcionada como base64:", "image_base64": base64_image}
        ]

        rospy.loginfo("Esperando respuesta del API de OpenAI...")

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0
            )

            if not response.choices or len(response.choices) == 0:
                rospy.logerr("La respuesta del API no contiene 'choices'. Respuesta completa: %s", response)
                return {"actions": []}

            response_text = response.choices[0].message.content.strip()

            # Intentar parsear el JSON
            try:
                instructions = json.loads(response_text)
                # Verificar que haya solo una acción
                actions = instructions.get("actions", [])
                if len(actions) != 1:
                    rospy.logwarn("El API devolvió más de una acción, solo se utilizará la primera.")
                    instructions["actions"] = [actions[0]] if actions else []
            except json.JSONDecodeError as jde:
                rospy.logerr("Error al decodificar el JSON. Texto de respuesta: %s", response_text)
                rospy.logerr("JSONDecodeError: %s", jde)
                instructions = {"actions": []}

            # Agregar la entrada actual a la historia
            self.history.append({
                "image_base64": base64_image,
                "response": instructions
            })
            # Si hay más de 3, eliminar la más antigua
            if len(self.history) > 3:
                self.history.pop(0)

            return instructions

        except Exception as e:
            rospy.logerr("Error al llamar al API de OpenAI: %s", e)
            traceback.print_exc()
            return {"actions": []}

    def get_distance_at_angle(self, ranges, angles, target_angle):
        angle_range = 5
        target_rad = np.radians(target_angle)
        indices = np.where((angles > target_rad - np.radians(angle_range)) & (angles < target_rad + np.radians(angle_range)))[0]
        distances = ranges[indices] if len(indices) > 0 else np.array([self.max_valid_range])
        return np.mean(distances)

    def find_free_direction(self, ranges, angles):
        free_mask = ranges > self.safety_distance
        free_angles = angles[free_mask]
        free_ranges = ranges[free_mask]

        forward_mask = np.abs(np.degrees(free_angles)) <= 90
        forward_angles = free_angles[forward_mask]
        forward_ranges = free_ranges[forward_mask]

        if len(forward_angles) > 0:
            max_distance_index = np.argmax(forward_ranges)
            target_angle = np.degrees(forward_angles[max_distance_index])
            return target_angle
        else:
            return None

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def shutdown_handler(self, signum, frame):
        rospy.loginfo("Señal recibida: %s. Cerrando nodo...", signum)
        self.shutdown()

    def shutdown(self):
        self.running = False
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(1)
        rospy.signal_shutdown("Nodo cerrado de manera segura.")


if __name__ == '__main__':
    try:
        node = ObstacleAvoidanceNode()
        node.control_loop()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.shutdown()

