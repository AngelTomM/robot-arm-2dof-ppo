"""
Entorno de Simulación 2D para Brazo Robótico de 2 Grados de Libertad (2 DOF)
Fase D: Proceso de Decisión de Markov (MDP) Continuo con Regulación Lyapunov / LQR.
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
    Entorno Gymnasium para un brazo planar 2 DOF bajo formulación MDP continua.
    Recompensa principal: lyapunov_lqr (función cuadrática continua sin horizonte discontinuo).
    Recompensa secundaria aislada: distance_only (-distance).
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

        # Límites físicos estrictos en articulación 2 (prevención de auto-colisión con la base)
        self.joint2_limit_deg = float(env_cfg.get("joint2_limit_deg", 170.0))
        self.joint2_max = math.radians(self.joint2_limit_deg)
        self.joint2_min = -self.joint2_max

        # Radio mínimo alcanzable según cinemática física: r_min = sqrt(l1^2 + l2^2 + 2*l1*l2*cos(joint2_max))
        min_reach_geom = math.sqrt(self.l1**2 + self.l2**2 + 2 * self.l1 * self.l2 * math.cos(self.joint2_max))
        self.min_reach = float(env_cfg.get("min_reach", min_reach_geom))

        self.dt = float(env_cfg.get("dt", 0.05))
        self.max_speed = float(env_cfg.get("max_speed", 2.0))
        self.target_radius = float(env_cfg.get("target_radius", 0.06))
        self.max_steps = int(env_cfg.get("max_steps", 200))

        # Validación estricta de Modo de Éxito (solo 'none' en Fase D)
        raw_success_mode = env_cfg.get("success_mode", "none")
        if raw_success_mode is None or str(raw_success_mode).strip().lower() == "none":
            self.success_mode = "none"
        else:
            raise ValueError(
                f"success_mode='{raw_success_mode}' no es válido en Fase D. "
                "Solo se admite 'none' (MDP puro sin horizonte discontinuo por hold/touch)."
            )

        # Validación estricta de Tipo de Recompensa
        self.reward_type = str(env_cfg.get("reward_type", "lyapunov_lqr")).strip().lower()
        if self.reward_type not in ["lyapunov_lqr", "distance_only"]:
            raise ValueError(
                f"reward_type='{self.reward_type}' no está soportado en Fase D. "
                "Solo se admiten 'lyapunov_lqr' (principal) y 'distance_only' (secundaria)."
            )

        # Parámetros de Recompensa Lyapunov / LQR
        self.w_pos = float(reward_params.get("w_pos", env_cfg.get("w_pos", 1.0)))
        self.w_vel = float(reward_params.get("w_vel", env_cfg.get("w_vel", 0.05)))
        self.w_ctrl = float(reward_params.get("w_ctrl", env_cfg.get("w_ctrl", 0.01)))

        # --- 2. Espacio de Acciones [-1.0, 1.0] ---
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # --- 3. Espacio de Observaciones (10 dimensiones cinemáticas intactas) ---
        obs_high = np.array([
            1.0, 1.0, 1.0, 1.0,                     # cos1, sin1, cos2, sin2
            self.max_speed, self.max_speed,         # velocidades dtheta1, dtheta2
            self.max_reach, self.max_reach,         # target_x, target_y
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
        """Construye el vector de observación actual (10 dimensiones cinemáticas puras)."""
        x_ee = self.l1 * math.cos(self.theta1) + self.l2 * math.cos(self.theta1 + self.theta2)
        y_ee = self.l1 * math.sin(self.theta1) + self.l2 * math.sin(self.theta1 + self.theta2)
        dist_vec = self.target - np.array([x_ee, y_ee], dtype=np.float32)

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
        """Reinicia el estado del entorno de forma determinista o aleatoria."""
        super().reset(seed=seed)

        self.theta1 = float(self.np_random.uniform(-math.pi, math.pi))
        self.theta2 = float(self.np_random.uniform(self.joint2_min, self.joint2_max))
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0
        self.step_count = 0

        target_r = float(self.np_random.uniform(self.min_reach, self.max_reach * 0.95))
        target_angle = float(self.np_random.uniform(-math.pi, math.pi))
        self.target = np.array([
            target_r * math.cos(target_angle),
            target_r * math.sin(target_angle)
        ], dtype=np.float32)

        _, end_effector = self._forward_kinematics(self.theta1, self.theta2)
        distance = float(np.linalg.norm(self.target - end_effector))

        obs = self._get_obs()
        info = {
            "distance": distance,
            "joint_speed": 0.0,
            "step_count": 0,
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    def set_state_and_target(self, theta1, theta2, target_x, target_y):
        """Permite fijar estados y objetivos específicos (para tests y datasets reproducibles)."""
        self.theta1 = float((theta1 + math.pi) % (2 * math.pi) - math.pi)
        if theta2 >= self.joint2_max:
            self.theta2 = float(self.joint2_max)
        elif theta2 <= self.joint2_min:
            self.theta2 = float(self.joint2_min)
        else:
            self.theta2 = float(theta2)
        self.dtheta1 = 0.0
        self.dtheta2 = 0.0
        self.target = np.array([target_x, target_y], dtype=np.float32)
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        """Aplica acción, calcula dinámica continua y evalúa recompensa MDP."""
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)
        self.dtheta1 = float(action[0] * self.max_speed)
        self.dtheta2 = float(action[1] * self.max_speed)

        # 1. Integración Euler y dinámica de topes mecánicos duros (Hard Stops)
        self.theta1 = float((self.theta1 + self.dtheta1 * self.dt + math.pi) % (2 * math.pi) - math.pi)

        new_theta2 = self.theta2 + self.dtheta2 * self.dt
        if new_theta2 > self.joint2_max:
            self.theta2 = self.joint2_max
            self.dtheta2 = 0.0
        elif new_theta2 < self.joint2_min:
            self.theta2 = self.joint2_min
            self.dtheta2 = 0.0
        else:
            self.theta2 = float(new_theta2)

        # 2. Cinemática Directa y Distancia
        _, end_effector = self._forward_kinematics(self.theta1, self.theta2)
        error_vec = self.target - end_effector
        distance = float(np.linalg.norm(error_vec))

        # 3. Velocidad Articular (norma euclidiana)
        joint_speed = float(np.linalg.norm([self.dtheta1, self.dtheta2]))

        # 4. Cálculo Descompuesto de Recompensa (solo 2 opciones permitidas)
        if self.reward_type == "lyapunov_lqr":
            reward_dist = -self.w_pos * (distance ** 2)
            reward_vel = -self.w_vel * (joint_speed ** 2)
            reward_action = -self.w_ctrl * float(np.sum(np.square(action)))
        elif self.reward_type == "distance_only":
            reward_dist = -distance
            reward_vel = 0.0
            reward_action = 0.0
        else:
            raise ValueError(f"reward_type='{self.reward_type}' no soportado.")

        total_reward = float(reward_dist + reward_vel + reward_action)

        # 5. Horizonte fijo: terminated siempre False en Fase D (corta por horizonte)
        terminated = False
        truncated = bool(self.step_count >= self.max_steps)

        obs = self._get_obs()
        info = {
            "distance": distance,
            "joint_speed": joint_speed,
            "action_norm_sq": float(np.sum(np.square(action))),
            "reward_dist": float(reward_dist),
            "reward_vel": float(reward_vel),
            "reward_action": float(reward_action),
            "step_count": self.step_count,
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

        # Dibujar área de alcance máximo y zona de auto-colisión (radio mínimo)
        pygame.draw.circle(
            self.screen,
            (35, 42, 58),
            self.origin,
            int(self.max_reach * self.scale),
            1
        )
        pygame.draw.circle(
            self.screen,
            (55, 30, 35),
            self.origin,
            int(self.min_reach * self.scale),
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
        elif dist < self.min_reach:
            if dist > 1e-6:
                scale = self.min_reach / dist
                x *= scale
                y *= scale
            else:
                x = self.min_reach
                y = 0.0
        self.target = np.array([x, y], dtype=np.float32)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
