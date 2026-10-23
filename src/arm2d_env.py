"""
Entorno de Simulación Dinámica MuJoCo para Manipulador Planar 2-DOF (Fase D).
Basado en Gymnasium 'Reacher-v5' bajo formulación MDP continua con recompensa cuadrática LQR.

Cinemática y Dinámica:
  Base fija en (0, 0, 0.01)
  Eslabón 1: longitud L1 = 0.10 m
  Eslabón 2 + Fingertip: longitud L2 = 0.11 m
  Acción: Torque articular normalizado en [-1.0, 1.0] (N·m)
"""
import math
from typing import Optional, Dict, Any, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from .config import load_env_config


class Arm2DEnv(gym.Env):
    """
    Adaptador canónico del entorno MuJoCo Reacher-v5 de Gymnasium.
    Implementa un Proceso de Decisión de Markov (MDP) continuo de horizonte fijo
    con función de costo/recompensa cuadrática canónica (Lyapunov / LQR):
        r_t = - (w_pos * ||p_ee - p_tgt||^2 + w_ctrl * ||a_t||^2 + w_vel * ||qvel||^2)
    """
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        # 1. Cargar configuración unificada (SSOT)
        if config is not None:
            cfg = config
        else:
            try:
                cfg = load_env_config()
            except Exception:
                cfg = {}

        env_cfg = cfg.get("env", {})
        reward_params = env_cfg.get("reward_params", {})

        # Parámetros geométricos y dinámicos
        self.l1 = float(env_cfg.get("l1", 0.10))
        self.l2 = float(env_cfg.get("l2", 0.11))
        self.target_radius = float(env_cfg.get("target_radius", 0.02))
        self.max_steps = int(env_cfg.get("max_steps", 200))
        self.dt = float(env_cfg.get("dt", 0.01))
        self.frame_skip = int(env_cfg.get("frame_skip", 2))
        self.vel_tol = 0.10  # rad/s

        # Ponderaciones del funcional de costo/recompensa
        self.w_pos = float(reward_params.get("w_pos", 1.0))
        self.w_ctrl = float(reward_params.get("w_ctrl", 0.01))
        self.w_vel = float(reward_params.get("w_vel", 0.05))

        self.render_mode = render_mode

        # 2. Instanciación del entorno Reacher-v5 de Gymnasium
        self.reacher_env = gym.make(
            "Reacher-v5",
            max_episode_steps=self.max_steps,
            render_mode=render_mode,
        )

        self.action_space = self.reacher_env.action_space
        self.observation_space = self.reacher_env.observation_space

        self.current_step = 0
        self._last_action = np.zeros(2, dtype=np.float32)

    @property
    def model(self):
        """Acceso al modelo cinemático/dinámico nativo de MuJoCo."""
        return self.reacher_env.unwrapped.model

    @property
    def data(self):
        """Acceso al estado de simulación (qpos, qvel, geom_xpos) de MuJoCo."""
        return self.reacher_env.unwrapped.data

    @property
    def unwrapped(self):
        """Acceso al entorno base sin wrappers."""
        return self.reacher_env.unwrapped

    def _get_obs(self) -> np.ndarray:
        """Extrae el vector de observación canónico de 10 dimensiones de Reacher-v5."""
        return self.reacher_env.unwrapped._get_obs()

    def _forward_kinematics(self, theta1: float, theta2: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula la cinemática directa planar:
            p_elbow = [l1 * cos(th1), l1 * sin(th1)]
            p_ee = [p_elbow_x + l2 * cos(th1 + th2), p_elbow_y + l2 * sin(th1 + th2)]
        """
        th1 = float(theta1)
        th2 = float(theta2)
        elbow_x = self.l1 * math.cos(th1)
        elbow_y = self.l1 * math.sin(th1)
        ee_x = elbow_x + self.l2 * math.cos(th1 + th2)
        ee_y = elbow_y + self.l2 * math.sin(th1 + th2)
        return np.array([elbow_x, elbow_y], dtype=np.float64), np.array([ee_x, ee_y], dtype=np.float64)

    def _compute_jacobian(self, theta1: float, theta2: float) -> np.ndarray:
        """Calcula analíticamente la matriz Jacobiana cinemática 2x2 para el efector final."""
        th1 = float(theta1)
        th2 = float(theta2)
        s1 = math.sin(th1)
        c1 = math.cos(th1)
        s12 = math.sin(th1 + th2)
        c12 = math.cos(th1 + th2)

        j11 = -self.l1 * s1 - self.l2 * s12
        j12 = -self.l2 * s12
        j21 = self.l1 * c1 + self.l2 * c12
        j22 = self.l2 * c12
        return np.array([[j11, j12], [j21, j22]], dtype=np.float64)

    def set_target(self, target_x: float, target_y: float) -> None:
        """Actualiza las coordenadas de la esfera objetivo en la simulación MuJoCo."""
        d = self.reacher_env.unwrapped.data
        m = self.reacher_env.unwrapped.model
        d.qpos[2] = float(target_x)
        d.qpos[3] = float(target_y)
        d.qvel[2] = 0.0
        d.qvel[3] = 0.0
        mujoco.mj_forward(m, d)

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Establece directamente el vector articular y de velocidad en el motor de física."""
        self.reacher_env.unwrapped.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64))

    def set_state_and_target(self, theta1: float, theta2: float, target_x: float, target_y: float) -> np.ndarray:
        """Reinicia el entorno fijando la postura articular y la posición cartesiana del objetivo."""
        obs, _ = self.reset(options={
            "theta1": theta1,
            "theta2": theta2,
            "target_x": target_x,
            "target_y": target_y,
        })
        return obs

    @property
    def theta1(self) -> float:
        """Ángulo actual de la primera articulación (rad)."""
        return float(self.data.qpos[0])

    @property
    def theta2(self) -> float:
        """Ángulo actual de la segunda articulación (rad)."""
        return float(self.data.qpos[1])

    @property
    def target(self) -> np.ndarray:
        """Posición cartesiana (x, y) del objetivo."""
        return self.data.qpos[2:4].copy()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reinicia el entorno y opcionalmente impone estados iniciales deterministas:
        options = {'theta1': float, 'theta2': float, 'target_x': float, 'target_y': float}
        """
        super().reset(seed=seed)
        self.current_step = 0
        self._last_action = np.zeros(2, dtype=np.float32)

        obs, info = self.reacher_env.reset(seed=seed)

        if options is not None:
            qpos = self.data.qpos.copy()
            qvel = np.zeros(4, dtype=np.float64)

            if "theta1" in options:
                qpos[0] = float(options["theta1"])
            if "theta2" in options:
                qpos[1] = float(options["theta2"])
            if "target_x" in options:
                qpos[2] = float(options["target_x"])
            if "target_y" in options:
                qpos[3] = float(options["target_y"])

            if "qpos" in options:
                qpos = np.asarray(options["qpos"], dtype=np.float64)
            if "qvel" in options:
                qvel = np.asarray(options["qvel"], dtype=np.float64)

            self.set_state(qpos, qvel)
            obs = self._get_obs()

        # Telemetría inicial
        p_ee = self.reacher_env.unwrapped.get_body_com("fingertip")[:2]
        p_tgt = self.reacher_env.unwrapped.get_body_com("target")[:2]
        dist = float(np.linalg.norm(p_ee - p_tgt))

        info.update({
            "distance": dist,
            "joint_speed": float(np.linalg.norm(self.data.qvel[:2])),
            "target_x": float(p_tgt[0]),
            "target_y": float(p_tgt[1]),
            "ee_x": float(p_ee[0]),
            "ee_y": float(p_ee[1]),
            "theta1": float(self.data.qpos[0]),
            "theta2": float(self.data.qpos[1]),
            "dtheta1": float(self.data.qvel[0]),
            "dtheta2": float(self.data.qvel[1]),
            "step_count": 0,
        })
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Ejecuta un paso de simulación dinámica aplicando torque en las articulaciones.
        Calcula la recompensa cuadrática canónica LQR parametrizada desde environment.yaml.
        """
        act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.current_step += 1

        # Avance dinámico en MuJoCo
        obs, _, _, _, _ = self.reacher_env.step(act)

        # Extracción cartesiana y articular desde datos de simulación
        p_ee = self.reacher_env.unwrapped.get_body_com("fingertip")[:2]
        p_tgt = self.reacher_env.unwrapped.get_body_com("target")[:2]
        dist_vec = p_ee - p_tgt
        dist = float(np.linalg.norm(dist_vec))
        pos_error_sq = float(np.dot(dist_vec, dist_vec))

        arm_qvel = self.data.qvel[:2]
        vel_sq = float(np.dot(arm_qvel, arm_qvel))
        joint_speed = float(np.linalg.norm(arm_qvel))

        ctrl_effort_sq = float(np.dot(act, act))
        act_delta = act - self._last_action
        act_delta_sq = float(np.dot(act_delta, act_delta))
        self._last_action = act.copy()

        # Recompensa continua cuadrática (Lyapunov / LQR)
        reward = -(self.w_pos * pos_error_sq + self.w_ctrl * ctrl_effort_sq + self.w_vel * vel_sq)

        terminated = False
        truncated = self.current_step >= self.max_steps

        is_success = (dist <= self.target_radius) and (joint_speed <= self.vel_tol)

        info = {
            "distance": dist,
            "pos_error_sq": pos_error_sq,
            "ctrl_effort_sq": ctrl_effort_sq,
            "vel_sq": vel_sq,
            "act_delta_sq": act_delta_sq,
            "joint_speed": joint_speed,
            "target_x": float(p_tgt[0]),
            "target_y": float(p_tgt[1]),
            "ee_x": float(p_ee[0]),
            "ee_y": float(p_ee[1]),
            "theta1": float(self.data.qpos[0]),
            "theta2": float(self.data.qpos[1]),
            "dtheta1": float(arm_qvel[0]),
            "dtheta2": float(arm_qvel[1]),
            "is_success": is_success,
            "step_count": self.current_step,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Renderizado nativo de MuJoCo (OpenGL interactivo o buffer RGB)."""
        return self.reacher_env.render()

    def close(self):
        """Libera los recursos del visualizador y motor físico."""
        if hasattr(self, "reacher_env") and self.reacher_env is not None:
            self.reacher_env.close()
