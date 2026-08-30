"""
Entorno de Simulación 2D para Brazo Robótico de 2 Grados de Libertad (2 DOF)
Compatible con Gymnasium, Stable-Baselines3 y configurable por YAML / Diccionario.

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
    Entorno Gymnasium personalizado para un brazo planar 2 DOF configurable.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60
    }

    def __init__(self, render_mode=None, config=None):
        super().__init__()

        # --- Cargar configuración o usar valores por defecto ---
        cfg = config or {}
        env_cfg = cfg.get("env", {})
        reward_params = env_cfg.get("reward_params", {})

        # 1. Parámetros Físicos y Geométricos
        self.l1 = float(env_cfg.get("l1", 1.0))
        self.l2 = float(env_cfg.get("l2", 1.0))
        self.max_reach = self.l1 + self.l2
        self.min_reach = abs(self.l1 - self.l2) + 0.1

        self.dt = float(env_cfg.get("dt", 0.05))
        self.max_speed = float(env_cfg.get("max_speed", 2.0))
        self.target_radius = float(env_cfg.get("target_radius", 0.08))
        self.max_steps = int(env_cfg.get("max_steps", 200))

        # 2. Configuración del Tipo de Recompensa (Reward Shaping)
        self.reward_type = env_cfg.get("reward_type", "distance_only")
        self.dist_coef = float(reward_params.get("distance_penalty_coef", 1.0))
        self.action_coef = float(reward_params.get("action_penalty_coef", 0.01))
        self.success_bonus = float(reward_params.get("success_bonus", 5.0))
        self.near_threshold = float(reward_params.get("near_zone_threshold", 0.25))
        self.vel_near_penalty = float(reward_params.get("velocity_penalty_near", 0.1))

        # 3. Estado Interno
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0
        self.target = np.zeros(2, dtype=np.float32)
        self.step_count = 0

        # 4. Espacio de Acciones (Action Space) [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # 5. Espacio de Observaciones (Observation Space) - 10 dimensiones
        obs_high = np.array([
            1.0, 1.0, 1.0, 1.0,               # sin/cos
            self.max_speed, self.max_speed,   # velocidades
            self.max_reach, self.max_reach,   # target_x, target_y
            2 * self.max_reach, 2 * self.max_reach  # distancias relativas
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )

        # 6. Renderizado (Pygame)
        self.render_mode = render_mode
        self.screen_dim = 600
        self.screen = None
        self.clock = None
        self.scale = (self.screen_dim / 2) / (self.max_reach * 1.15)
        self.origin = np.array([self.screen_dim // 2, self.screen_dim // 2])

    def _forward_kinematics(self, th1, th2):
        """Calcula cinemática directa: posición de codo y efector final."""
        x1 = self.l1 * math.cos(th1)
        y1 = self.l1 * math.sin(th1)
        x2 = x1 + self.l2 * math.cos(th1 + th2)
        y2 = y1 + self.l2 * math.sin(th1 + th2)
        return np.array([x1, y1], dtype=np.float32), np.array([x2, y2], dtype=np.float32)

    def _get_obs(self):
        """Construye el vector de observación actual."""
        _, end_effector = self._forward_kinematics(self.theta1, self.theta2)
        dist_vec = self.target - end_effector

        return np.array([
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

    def reset(self, seed=None, options=None):
        """Reinicia el estado del entorno."""
        super().reset(seed=seed)

        self.theta1 = float(self.np_random.uniform(-math.pi, math.pi))
        self.theta2 = float(self.np_random.uniform(-math.pi, math.pi))
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0
        self.step_count = 0

        target_r = float(self.np_random.uniform(self.min_reach, self.max_reach * 0.95))
        target_angle = float(self.np_random.uniform(-math.pi, math.pi))
        self.target = np.array([
            target_r * math.cos(target_angle),
            target_r * math.sin(target_angle)
        ], dtype=np.float32)

        obs = self._get_obs()
        info = {
            "is_success": False,
            "distance": float(np.linalg.norm(self.target - self._forward_kinematics(self.theta1, self.theta2)[1]))
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        """Aplica acción, calcula dinámica y evalúa la recompensa según la estrategia configurada."""
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)
        self.dtheta1 = float(action[0] * self.max_speed)
        self.dtheta2 = float(action[1] * self.max_speed)

        # Integración Euler
        self.theta1 += self.dtheta1 * self.dt
        self.theta2 += self.dtheta2 * self.dt
        self.theta1 = (self.theta1 + math.pi) % (2 * math.pi) - math.pi
        self.theta2 = (self.theta2 + math.pi) % (2 * math.pi) - math.pi

        _, end_effector = self._forward_kinematics(self.theta1, self.theta2)
        distance = float(np.linalg.norm(self.target - end_effector))
        is_success = distance < self.target_radius

        # --- Cálculo Modular de la Recompensa ---
        if self.reward_type == "smooth_precision":
            # Penalización base por distancia y acción
            reward = -self.dist_coef * distance - self.action_coef * float(np.sum(np.square(action)))
            # Recompensa por proximidad suave y penalización por exceso de velocidad al aproximarse
            if distance < self.near_threshold:
                reward += 1.0 - (distance / self.near_threshold)
                reward -= self.vel_near_penalty * (abs(self.dtheta1) + abs(self.dtheta2))
            if is_success:
                reward += self.success_bonus

        elif self.reward_type == "energy_efficient":
            reward = -self.dist_coef * distance - 0.1 * float(np.sum(np.square(action)))
            if is_success:
                reward += self.success_bonus

        else:  # "distance_only" (Por defecto)
            reward = -self.dist_coef * distance - self.action_coef * float(np.sum(np.square(action)))
            if is_success:
                reward += self.success_bonus

        terminated = is_success
        truncated = self.step_count >= self.max_steps

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
        """Renderizado en tiempo real con Pygame."""
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
                pygame.display.set_caption(f"2 DOF Robot Arm - Mode: {self.reward_type}")
            else:
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
            self.clock = pygame.time.Clock()

        self.screen.fill((20, 24, 34))

        # Dibujar área de alcance
        pygame.draw.circle(
            self.screen,
            (35, 42, 58),
            self.origin,
            int(self.max_reach * self.scale),
            1
        )

        def to_screen(pt):
            return (
                int(self.origin[0] + pt[0] * self.scale),
                int(self.origin[1] - pt[1] * self.scale)
            )

        p_base = self.origin
        elbow, end_effector = self._forward_kinematics(self.theta1, self.theta2)
        p_elbow = to_screen(elbow)
        p_ee = to_screen(end_effector)
        p_target = to_screen(self.target)

        # 1. Target
        pygame.draw.circle(self.screen, (46, 204, 113, 50), p_target, int(self.target_radius * self.scale), 1)
        pygame.draw.circle(self.screen, (46, 204, 113), p_target, 8)

        # 2. Eslabones
        pygame.draw.line(self.screen, (52, 152, 219), p_base, p_elbow, 8)
        pygame.draw.line(self.screen, (241, 196, 15), p_elbow, p_ee, 6)

        # 3. Articulaciones
        pygame.draw.circle(self.screen, (236, 240, 241), p_base, 10)
        pygame.draw.circle(self.screen, (236, 240, 241), p_elbow, 7)
        pygame.draw.circle(self.screen, (231, 76, 60), p_ee, 6)

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def set_target(self, x, y):
        dist = math.sqrt(x**2 + y**2)
        if dist > self.max_reach:
            scale = (self.max_reach * 0.95) / dist
            x *= scale
            y *= scale
        self.target = np.array([x, y], dtype=np.float32)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
