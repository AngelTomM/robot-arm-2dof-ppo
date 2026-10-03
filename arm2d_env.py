"""
Entorno de Simulación 2D para Brazo Robótico de 2 Grados de Libertad (2 DOF)
Compatible con Gymnasium, Stable-Baselines3 y configurable por YAML / Diccionario.

Fase A: Preservación de comportamiento histórico y soporte para modo 'hold',
descomposición de recompensas y métricas operativas avanzadas.

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
    Entorno Gymnasium configurable para un brazo planar 2 DOF.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60
    }

    def __init__(self, render_mode=None, config=None):
        super().__init__()

        # --- 1. Cargar configuración o usar valores por defecto ---
        cfg = config or {}
        env_cfg = cfg.get("env", {})
        reward_params = env_cfg.get("reward_params", {})

        # Parámetros Físicos y Geométricos
        self.l1 = float(env_cfg.get("l1", 1.0))
        self.l2 = float(env_cfg.get("l2", 1.0))
        self.max_reach = self.l1 + self.l2
        self.min_reach = abs(self.l1 - self.l2) + 0.1

        self.dt = float(env_cfg.get("dt", 0.05))
        self.max_speed = float(env_cfg.get("max_speed", 2.0))
        self.target_radius = float(env_cfg.get("target_radius", 0.08))
        self.max_steps = int(env_cfg.get("max_steps", 200))

        # Flags de Modo de Éxito y Recompensa (Fase A)
        # success_mode: "touch" (histórico) | "hold"
        self.success_mode = str(env_cfg.get("success_mode", "touch"))
        # Parámetros de Hold
        self.hold_steps = int(env_cfg.get("hold_steps", 20))
        self.velocity_tolerance = float(env_cfg.get("velocity_tolerance", 0.1))

        # Tipo de Recompensa
        # "distance_only", "smooth_precision", "energy_efficient", "historical", "x5_near", "x5_vel", "x6_smooth"
        self.reward_type = str(env_cfg.get("reward_type", "distance_only"))
        
        # Parámetros de Recompensa
        self.dist_coef = float(reward_params.get("distance_penalty_coef", 1.0))
        self.action_coef = float(reward_params.get("action_penalty_coef", 0.01))
        self.success_bonus = float(reward_params.get("success_bonus", 5.0))
        self.near_threshold = float(reward_params.get("near_zone_threshold", 0.25))
        self.vel_near_penalty = float(reward_params.get("velocity_penalty_near", 0.1))
        
        # Parámetros Nuevos de Fase A (X5, X6)
        self.w_v = float(reward_params.get("w_v", 0.0))
        self.sigma_v = float(reward_params.get("sigma_v", 0.15))
        self.w_a = float(reward_params.get("w_a", 0.0))
        self.r_near_mode = str(reward_params.get("r_near_mode", "historical")) # "historical", "none", "non_positive"

        # --- 2. Espacio de Acciones [-1.0, 1.0] ---
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # --- 3. Espacio de Observaciones (10 dimensiones) ---
        obs_high = np.array([
            1.0, 1.0, 1.0, 1.0,               # cos1, sin1, cos2, sin2
            self.max_speed, self.max_speed,   # velocidades dtheta1, dtheta2
            self.max_reach, self.max_reach,   # target_x, target_y
            2 * self.max_reach, 2 * self.max_reach  # dist_x, dist_y
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )

        # --- 4. Variables de Estado Interno ---
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0
        self.target = np.zeros(2, dtype=np.float32)
        self.step_count = 0

        # Variables de Métricas y Hold de Fase A
        self.hold_counter = 0
        self.hold_counter_max = 0
        self.prev_action = None
        self.has_prev_action = False
        
        self.first_entry_step = None
        self.joint_speed_at_first_entry = None
        self.ee_speed_at_first_entry = None
        self.was_inside_target = False
        self.n_exits = 0
        self.n_cross = 0
        self.prev_error_vec = None
        self.time_inside_near = 0

        # --- 5. Renderizado (Pygame) ---
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

    def get_jacobian(self, th1, th2):
        """Calcula la matriz Jacobiana analítica del efector final."""
        j11 = -self.l1 * math.sin(th1) - self.l2 * math.sin(th1 + th2)
        j12 = -self.l2 * math.sin(th1 + th2)
        j21 = self.l1 * math.cos(th1) + self.l2 * math.cos(th1 + th2)
        j22 = self.l2 * math.cos(th1 + th2)
        return np.array([[j11, j12], [j21, j22]], dtype=np.float32)

    def _get_obs(self):
        """Construye el vector de observación actual (10 dimensiones)."""
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

        # Reiniciar contadores y métricas de Fase A
        self.hold_counter = 0
        self.hold_counter_max = 0
        self.prev_action = None
        self.has_prev_action = False

        self.first_entry_step = None
        self.joint_speed_at_first_entry = None
        self.ee_speed_at_first_entry = None
        self.was_inside_target = False
        self.n_exits = 0
        self.n_cross = 0
        self.prev_error_vec = None
        self.time_inside_near = 0

        _, end_effector = self._forward_kinematics(self.theta1, self.theta2)
        distance = float(np.linalg.norm(self.target - end_effector))

        obs = self._get_obs()
        info = {
            "is_success": False,
            "distance": distance,
            "hold_counter": 0,
            "hold_counter_max": 0
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    def set_state_and_target(self, theta1, theta2, target_x, target_y):
        """Permite fijar estados y objetivos específicos (para tests y datasets)."""
        self.theta1 = float((theta1 + math.pi) % (2 * math.pi) - math.pi)
        self.theta2 = float((theta2 + math.pi) % (2 * math.pi) - math.pi)
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0
        self.target = np.array([target_x, target_y], dtype=np.float32)
        self.step_count = 0
        self.hold_counter = 0
        self.hold_counter_max = 0
        self.prev_action = None
        self.has_prev_action = False
        self.first_entry_step = None
        self.joint_speed_at_first_entry = None
        self.ee_speed_at_first_entry = None
        self.was_inside_target = False
        self.n_exits = 0
        self.n_cross = 0
        self.prev_error_vec = None
        self.time_inside_near = 0
        return self._get_obs()

    def step(self, action):
        """Aplica acción, calcula dinámica y evalúa recompensa y permanencia."""
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)
        self.dtheta1 = float(action[0] * self.max_speed)
        self.dtheta2 = float(action[1] * self.max_speed)

        # 1. Integración Euler (mismo orden histórico exacto)
        self.theta1 += self.dtheta1 * self.dt
        self.theta2 += self.dtheta2 * self.dt
        self.theta1 = (self.theta1 + math.pi) % (2 * math.pi) - math.pi
        self.theta2 = (self.theta2 + math.pi) % (2 * math.pi) - math.pi

        # 2. Cinemática Directa y Distancia
        _, end_effector = self._forward_kinematics(self.theta1, self.theta2)
        error_vec = self.target - end_effector
        distance = float(np.linalg.norm(error_vec))

        # 3. Velocidades
        joint_speed = float(math.sqrt(self.dtheta1**2 + self.dtheta2**2))
        jac = self.get_jacobian(self.theta1, self.theta2)
        ee_vel = jac @ np.array([self.dtheta1, self.dtheta2], dtype=np.float32)
        ee_speed = float(np.linalg.norm(ee_vel))

        # 4. Evaluación de Condición de Éxito
        inside_target = (distance < self.target_radius)
        slow_enough = (joint_speed < self.velocity_tolerance)

        if self.success_mode == "hold":
            if inside_target and slow_enough:
                self.hold_counter += 1
            else:
                self.hold_counter = 0
            
            if self.hold_counter > self.hold_counter_max:
                self.hold_counter_max = self.hold_counter
            
            is_success = (self.hold_counter >= self.hold_steps)
        else:
            # Modo histórico por contacto ("touch")
            is_success = inside_target

        # 5. Métricas Operativas de Fase A
        # (a) Primera entrada en target
        if inside_target and (self.first_entry_step is None):
            self.first_entry_step = self.step_count
            self.joint_speed_at_first_entry = joint_speed
            self.ee_speed_at_first_entry = ee_speed

        # (b) Salidas tras primera entrada
        if self.first_entry_step is not None:
            if self.was_inside_target and not inside_target:
                self.n_exits += 1
        self.was_inside_target = inside_target

        # (c) Tiempo en región cercana (d < 0.30)
        if distance < 0.30:
            self.time_inside_near += 1

        # (d) Cruces de signo de vector de error en d < 0.30
        if distance < 0.30 and (self.prev_error_vec is not None):
            if float(np.dot(error_vec, self.prev_error_vec)) < 0.0:
                self.n_cross += 1
        self.prev_error_vec = error_vec

        # (e) Variación de acción (Convención Sección 14)
        if self.has_prev_action:
            delta_action = action - self.prev_action
            delta_action_norm = float(np.linalg.norm(delta_action))
            delta_action_sq = float(np.sum(np.square(delta_action)))
        else:
            delta_action = np.zeros_like(action)
            delta_action_norm = None # No definido en paso 0
            delta_action_sq = 0.0

        # --- 6. Cálculo Descompuesto de Recompensa ---
        reward_dist = -self.dist_coef * distance
        reward_action = -self.action_coef * float(np.sum(np.square(action)))
        reward_near = 0.0
        reward_vel = 0.0
        reward_smooth = 0.0
        reward_success = 0.0

        # Cálculo de términos según reward_type
        if self.reward_type in ["smooth_precision", "historical", "x5_near", "x5_vel", "x6_smooth"]:
            if distance < self.near_threshold:
                if self.r_near_mode == "none" or self.reward_type == "x5_near":
                    # X5-near diagnóstico: sin término positivo
                    reward_near = -self.vel_near_penalty * (abs(self.dtheta1) + abs(self.dtheta2))
                elif self.r_near_mode == "non_positive":
                    reward_near = min(0.0, 1.0 - (distance / self.near_threshold)) - self.vel_near_penalty * (abs(self.dtheta1) + abs(self.dtheta2))
                else:
                    # Histórico
                    reward_near = (1.0 - (distance / self.near_threshold)) - self.vel_near_penalty * (abs(self.dtheta1) + abs(self.dtheta2))

        elif self.reward_type == "energy_efficient":
            reward_action = -0.1 * float(np.sum(np.square(action)))

        # Penalización Gaussiana de Velocidad (X5)
        if self.w_v > 0.0:
            gauss_factor = math.exp(-(distance**2) / (2.0 * (self.sigma_v**2)))
            reward_vel = -self.w_v * gauss_factor * (joint_speed**2)

        # Penalización de Suavidad (X6)
        if self.w_a > 0.0 and self.has_prev_action:
            reward_smooth = -self.w_a * delta_action_sq

        # Bonificación de Éxito (otorgada al completar is_success)
        if is_success:
            reward_success = self.success_bonus

        total_reward = reward_dist + reward_action + reward_near + reward_vel + reward_smooth + reward_success

        # Actualizar acción previa para el siguiente paso
        self.prev_action = np.copy(action)
        self.has_prev_action = True

        terminated = bool(is_success)
        truncated = bool(self.step_count >= self.max_steps and not terminated)

        obs = self._get_obs()
        info = {
            "is_success": is_success,
            "distance": distance,
            "step_count": self.step_count,
            "hold_counter": self.hold_counter,
            "hold_counter_max": self.hold_counter_max,
            "joint_speed": joint_speed,
            "ee_speed": ee_speed,
            "first_entry_step": self.first_entry_step,
            "joint_speed_at_first_entry": self.joint_speed_at_first_entry,
            "ee_speed_at_first_entry": self.ee_speed_at_first_entry,
            "n_exits": self.n_exits,
            "n_cross": self.n_cross,
            "time_inside_near": self.time_inside_near,
            "delta_action_norm": delta_action_norm,
            "reward_dist": reward_dist,
            "reward_action": reward_action,
            "reward_near": reward_near,
            "reward_vel": reward_vel,
            "reward_smooth": reward_smooth,
            "reward_success": reward_success,
            "total_reward": total_reward
        }

        if self.render_mode == "human":
            self.render()

        return obs, total_reward, terminated, truncated, info

    def render(self):
        """Renderizado en tiempo real con Pygame."""
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
                pygame.display.set_caption(f"2 DOF Robot Arm - Mode: {self.reward_type} ({self.success_mode})")
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
