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

            if self.current_image is not None:
                if self.last_image_processed is not self.current_image:
                    rospy.loginfo("Enviando imagen al API para su análisis...")
                    try:
                        self.api_response = self.analyze_image_with_api(self.current_image)
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
            if not self.api_response.get("actions"):
                twist = Twist()
                self.cmd_vel_pub.publish(twist)
                self.rate.sleep()
                continue

            # Tomar la acción (única) de la lista
            action = self.api_response["actions"][0]
            movement_type = action.get("movement_type", "stop")
            distance = action.get("distance", 0.0)
            angle = action.get("angle", 0.0)
            duration = action.get("duration", 1)

            rospy.logdebug("Acción a ejecutar: %s", action)

            twist = Twist()
            if movement_type == "forward":
                twist.linear.x = self.forward_speed
                twist.angular.z = 0.0
            elif movement_type == "turn_left":
                twist.linear.x = 0.0
                twist.angular.z = min(self.max_turn_speed, self.max_turn_speed * (angle/90.0))
            elif movement_type == "turn_right":
                twist.linear.x = 0.0
                twist.angular.z = -min(self.max_turn_speed, self.max_turn_speed * (angle/90.0))
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

    def analyze_image_with_api(self, cv_image):
        # Codificar la imagen a base64 según la documentación
        _, buffer = cv2.imencode('.jpg', cv_image)
        img_bytes = buffer.tobytes()
        base64_image = base64.b64encode(img_bytes).decode('utf-8')

        # Mensaje del sistema, exigir una sola acción
        # Ahora indicamos que tenemos en cuenta las 3 últimas iteraciones
        system_message = (
            "Eres un asistente para un TurtleBot3 en una pista de carreras. "
            "Debes analizar la imagen y generar instrucciones en JSON, sin Markdown. "
            "El TurtleBot3 debe siempre intentar avanzar, salvo que haya un obstáculo cercano. "
            "Debes devolver sólo una acción en el campo 'actions' (una sola), por ejemplo:\n"
            "Formato del JSON:\n"
            "{\n"
            "  \"actions\": [\n"
            "    {\"movement_type\": \"forward\", \"distance\": <num>, \"duration\": <num>},\n"
            "    {\"movement_type\": \"turn_left\", \"angle\": <num>, \"duration\": <num>},\n"
            "  ]\n"
            "}\n"
            "Nada más que este JSON con una sola acción. RECUERDA DEBES AVANZAR SIEMPRE PORQUE ES UNA CARRERA PARAR ES SOLO CUANDO LA CAMARA NO VE NADA PORQUE ESTA PEGADO A UNA PARED. AVANZA SIEMPRE NO DES VUELTAS A CADA RATO"
            "Tienes en cuenta las últimas 3 imágenes previas con sus acciones resultantes. "
            "Si ha avanzado varias veces seguidas, el camino delante está libre. "
            "Usa la información histórica para tomar la mejor decisión. SI VES TODO OSCURO O TODO CLARO ES PORQUE TE HAZ ACERCADO MUCHO A UNA PARED LO QUE DEBES HACER ES GIRARA 180 GRADOS"
        )

        # Construir mensajes adicionales con el historial (hasta 3 últimas iteraciones)
        # Cada elemento de self.history es {"image_base64":..., "response":...}
        history_messages = []
        for i, entry in enumerate(self.history[-3:], start=1):
            # Podríamos proporcionar un pequeño contexto al modelo
            history_msg = (
                f"Historial {i}:\n"
                f"Imagen previa (base64 no mostrada por brevedad).\n"
                f"Respuesta anterior del API: {json.dumps(entry['response'])}\n"
            )
            # Lo agregamos como un mensaje assistant para dar contexto
            history_messages.append(
                {"role": "assistant", "content": history_msg}
            )

        user_message = [
            {
                "type": "text",
                "text": "Analiza la imagen y proporciona el JSON con una sola acción basándote también en el historial. RECUERDA ES UNA PISTA DE CARRERAS DE CAJAS DE CARTON TU PUEDES CREER QUE ESTAS CERCA PERO NO ES PORQUE LA CAMARA REDUCE AVANZA NOMAS ASI CREAS QEU CHIQUES SOO CUANDO NO VEAS ABSOLUTAMENTE NADA AHI RECIEN GIRA SINO AVANZA"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]

        rospy.loginfo("Esperando respuesta del API de OpenAI...")

        try:
            messages = [
                {"role": "system", "content": system_message},
            ]

            # Agregar el historial como contexto previo
            messages.extend(history_messages)

            # Finalmente, el mensaje del usuario con la imagen actual
            messages.append({"role": "user", "content": user_message})

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

