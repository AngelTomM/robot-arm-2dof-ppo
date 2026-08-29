"""
Entorno de Simulación 2D para Brazo Robótico de 2 Grados de Libertad (2 DOF)
Compatible con Gymnasium y Stable-Baselines3.

Cinemática Directa:
  Base en (0, 0)
  Codo (x1, y1) = (L1 * cos(theta1), L1 * sin(theta1))
  Efector (x, y) = (x1 + L2 * cos(theta1 + theta2), y1 + L2 * sin(theta1 + theta2))
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame


class Arm2DEnv(gym.Env):
    """
    Entorno Gymnasium personalizado para un brazo planar 2 DOF.
    
    Metadatos:
      - render_modes: Modos de renderizado soportados ('human', 'rgb_array').
      - render_fps: Cuadros por segundo para el renderizado.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60
    }

    def __init__(self, render_mode=None):
        super().__init__()

        # --- 1. Parámetros Físicos y Geométricos del Brazo ---
        self.l1 = 1.0  # Longitud del primer eslabón (metros/unidades arbitrarias)
        self.l2 = 1.0  # Longitud del segundo eslabón
        self.max_reach = self.l1 + self.l2  # Alcance máximo = 2.0
        self.min_reach = abs(self.l1 - self.l2) + 0.1  # Alcance mínimo para evitar singularidad

        self.dt = 0.05  # Paso de tiempo de la simulación física (segundos)
        self.max_speed = 2.0  # Velocidad angular máxima (rad/s)
        self.target_radius = 0.08  # Radio de tolerancia de éxito alrededor del objetivo

        # --- 2. Estado Interno ---
        self.theta1 = 0.0  # Ángulo de la articulación 1 (rad)
        self.theta2 = 0.0  # Ángulo relativo de la articulación 2 (rad)
        self.dtheta1 = 0.0  # Velocidad angular 1
        self.dtheta2 = 0.0  # Velocidad angular 2
        self.target = np.zeros(2, dtype=np.float32)  # Posición [x, y] del objetivo
        self.step_count = 0
        self.max_steps = 200  # Límite de pasos por episodio

        # --- 3. Espacio de Acciones (Action Space) ---
        # Acción continua: [velocidad_angular_1, velocidad_angular_2] normalizada entre [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # --- 4. Espacio de Observaciones (Observation Space) ---
        # Representación continua para evitar discontinuidades angulares en [-pi, pi]:
        # [cos(th1), sin(th1), cos(th2), sin(th2), dth1, dth2, target_x, target_y, dist_x, dist_y]
        # Total: 10 valores
        obs_high = np.array([
            1.0, 1.0, 1.0, 1.0,  # sin/cos
            self.max_speed, self.max_speed,  # velocidades
            self.max_reach, self.max_reach,  # target_x, target_y
            2 * self.max_reach, 2 * self.max_reach  # distancias relativas
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )

        # --- 5. Configuración de Renderizado (Pygame) ---
        self.render_mode = render_mode
        self.screen_dim = 600
        self.screen = None
        self.clock = None
        self.scale = (self.screen_dim / 2) / (self.max_reach * 1.15)  # Factor de escala pixel/metro
        self.origin = np.array([self.screen_dim // 2, self.screen_dim // 2])  # Centro en píxeles

    def _forward_kinematics(self, th1, th2):
        """
        Paso 2: Cinemática Directa
        Calcula las coordenadas (x, y) de la articulación intermedia (codo) y del efector final.
        """
        # Posición del codo (articulación 1)
        x1 = self.l1 * math.cos(th1)
        y1 = self.l1 * math.sin(th1)

        # Posición del efector final (articulación 2)
        # Nota: el ángulo del segundo eslabón respecto al plano global es (th1 + th2)
        x2 = x1 + self.l2 * math.cos(th1 + th2)
        y2 = y1 + self.l2 * math.sin(th1 + th2)

        return np.array([x1, y1], dtype=np.float32), np.array([x2, y2], dtype=np.float32)

    def _get_obs(self):
        """
        Construye el vector de observación actual.
        """
        _, end_effector = self._forward_kinematics(self.theta1, self.theta2)
        dist_vec = self.target - end_effector

        obs = np.array([
            math.cos(self.theta1),
            math.sin(self.theta1),
            math.cos(self.theta2),
            math.sin(self.theta2),
            self.dtheta1,
            self.dtheta2,
            self.target[0],
            self.target[1],
            dist_vec[0],
            dist_vec[1]
        ], dtype=np.float32)

        return obs

    def reset(self, seed=None, options=None):
        """
        Método OBLIGATORIO de Gymnasium:
        Reinicia el estado del entorno al inicio de un nuevo episodio.
        Retorna: tuple(observation, info)
        """
        super().reset(seed=seed)

        # 1. Ángulos iniciales aleatorios en [-pi, pi]
        self.theta1 = float(self.np_random.uniform(-math.pi, math.pi))
        self.theta2 = float(self.np_random.uniform(-math.pi, math.pi))
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0
        self.step_count = 0

        # 2. Generar objetivo aleatorio alcanzable en coordenadas polares
        target_r = float(self.np_random.uniform(self.min_reach, self.max_reach * 0.95))
        target_angle = float(self.np_random.uniform(-math.pi, math.pi))
        self.target = np.array([
            target_r * math.cos(target_angle),
            target_r * math.sin(target_angle)
        ], dtype=np.float32)

        obs = self._get_obs()
        info = {"is_success": False, "distance": float(np.linalg.norm(self.target - self._forward_kinematics(self.theta1, self.theta2)[1]))}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        """
        Método OBLIGATORIO de Gymnasium:
        Aplica la acción del agente, actualiza la física y calcula la recompensa.
        Retorna: tuple(observation, reward, terminated, truncated, info)
        """
        self.step_count += 1

        # 1. Escalar acción normalizada [-1, 1] a velocidades angulares
        action = np.clip(action, -1.0, 1.0)
        self.dtheta1 = float(action[0] * self.max_speed)
        self.dtheta2 = float(action[1] * self.max_speed)

        # 2. Integración Euler para actualizar ángulos
        self.theta1 += self.dtheta1 * self.dt
        self.theta2 += self.dtheta2 * self.dt

        # Normalizar ángulos al rango [-pi, pi]
        self.theta1 = (self.theta1 + math.pi) % (2 * math.pi) - math.pi
        self.theta2 = (self.theta2 + math.pi) % (2 * math.pi) - math.pi

        # 3. Cinemática directa y distancia al objetivo
        _, end_effector = self._forward_kinematics(self.theta1, self.theta2)
        distance = float(np.linalg.norm(self.target - end_effector))

        # 4. Función de Recompensa (Reward Function)
        # - Recompensa densa por proximidad (penalización proporcional a la distancia)
        # - Penalización suave por esfuerzo de control (magnitud de la acción)
        reward = -distance - 0.01 * float(np.sum(np.square(action)))

        # Bono si alcanza el objetivo con precisión
        is_success = distance < self.target_radius
        if is_success:
            reward += 5.0  # Gran bonificación al tocar la meta

        # 5. Condiciones de finalización del episodio
        terminated = is_success  # Termina con éxito
        truncated = self.step_count >= self.max_steps  # Termina por tiempo límite

        obs = self._get_obs()
        info = {
            "is_success": is_success,
            "distance": distance,
            "step_count": self.step_count
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Método OBLIGATORIO de Gymnasium:
        Dibuja el estado del robot en pantalla o genera un array RGB.
        """
        if self.render_mode is None:
            return

        # Inicialización perezosa de Pygame
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
                pygame.display.set_caption("2 DOF Robot Arm - PPO Simulation")
            else:  # rgb_array
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
            self.clock = pygame.time.Clock()

        # Fondo oscuro elegante
        self.screen.fill((20, 24, 34))

        # Dibujar área de alcance máximo
        pygame.draw.circle(
            self.screen,
            (35, 42, 58),
            self.origin,
            int(self.max_reach * self.scale),
            1
        )

        # Convertir coordenadas métricas a píxeles de pantalla
        def to_screen(pt):
            # Invertir eje Y porque en pantalla crece hacia abajo
            px = int(self.origin[0] + pt[0] * self.scale)
            py = int(self.origin[1] - pt[1] * self.scale)
            return (px, py)

        p_base = self.origin
        elbow, end_effector = self._forward_kinematics(self.theta1, self.theta2)
        p_elbow = to_screen(elbow)
        p_ee = to_screen(end_effector)
        p_target = to_screen(self.target)

        # 1. Dibujar Objetivo (Target) con halo y radio de tolerancia
        target_color = (46, 204, 113)  # Verde esmeralda
        pygame.draw.circle(self.screen, (46, 204, 113, 50), p_target, int(self.target_radius * self.scale), 1)
        pygame.draw.circle(self.screen, target_color, p_target, 8)

        # 2. Dibujar Eslabón 1
        pygame.draw.line(self.screen, (52, 152, 219), p_base, p_elbow, 8)  # Azul

        # 3. Dibujar Eslabón 2
        pygame.draw.line(self.screen, (241, 196, 15), p_elbow, p_ee, 6)  # Amarillo/Dorado

        # 4. Dibujar Articulaciones
        pygame.draw.circle(self.screen, (236, 240, 241), p_base, 10)   # Base
        pygame.draw.circle(self.screen, (236, 240, 241), p_elbow, 7)  # Codo
        pygame.draw.circle(self.screen, (231, 76, 60), p_ee, 6)       # Efector final (Rojo)

        if self.render_mode == "human":
            # Procesar eventos de Pygame para mantener la ventana responsiva
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def set_target(self, x, y):
        """Permite fijar manualmente el target (útil para interacción con mouse)."""
        dist = math.sqrt(x**2 + y**2)
        if dist > self.max_reach:
            scale = (self.max_reach * 0.95) / dist
            x *= scale
            y *= scale
        self.target = np.array([x, y], dtype=np.float32)

    def close(self):
        """
        Método OBLIGATORIO de Gymnasium:
        Libera recursos gráficos.
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
