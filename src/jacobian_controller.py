"""
Controlador Analítico Cartesiano PD con Transposición del Jacobiano (C0).
Formulado para control por torque continuo en el manipulador planar 2-DOF (MuJoCo Reacher-v5).

Ecuación de control por torque:
    tau = J(q)^T * (Kp * e_pos) - Kd * qvel
    action = clip(tau, -1.0, 1.0)
"""
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import math
import numpy as np
import yaml

from .config import load_env_config


class JacobianDampedController:
    """
    Controlador cartesiano analítico PD en espacio operacional con transposición del Jacobiano.
    Genera torques normalizados en [-1.0, 1.0] compatibles con MuJoCo Reacher-v5.
    """
    def __init__(self, kp: float = 50.0, kd: float = 1.5,
                 l1: float = 0.10, l2: float = 0.11,
                 lambda_damping: float = 0.01,
                 env_config: Optional[Union[Dict[str, Any], str, Path]] = None):
        self.kp = float(kp)
        self.kd = float(kd)
        self.lambda_damping = float(lambda_damping)

        # Si se suministra env_config, sincronizar parámetros geométricos
        if env_config is not None:
            resolved_cfg = self._parse_config(env_config)
            env_sec = resolved_cfg.get("env", resolved_cfg)
            self.l1 = float(env_sec.get("l1", l1))
            self.l2 = float(env_sec.get("l2", l2))
        else:
            self.l1 = float(l1)
            self.l2 = float(l2)

    @staticmethod
    def _parse_config(cfg: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
        """Carga y parsea la configuración del entorno."""
        if isinstance(cfg, (str, Path)):
            p = Path(cfg)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            return {}
        elif isinstance(cfg, dict):
            return cfg
        return {}

    @classmethod
    def from_config(cls, env_config: Optional[Union[Dict[str, Any], str, Path]] = None,
                    kp: float = 50.0, kd: float = 1.5) -> "JacobianDampedController":
        """Instancia el controlador a partir del archivo o diccionario unificado del entorno."""
        if env_config is None:
            try:
                env_config = load_env_config()
            except Exception:
                env_config = None
        return cls(kp=kp, kd=kd, env_config=env_config)

    def _compute_jacobian(self, th1: float, th2: float) -> np.ndarray:
        """Calcula analíticamente la matriz Jacobiana 2x2 para el efector final."""
        s1 = math.sin(th1)
        c1 = math.cos(th1)
        s12 = math.sin(th1 + th2)
        c12 = math.cos(th1 + th2)

        j11 = -self.l1 * s1 - self.l2 * s12
        j12 = -self.l2 * s12
        j21 = self.l1 * c1 + self.l2 * c12
        j22 = self.l2 * c12
        return np.array([[j11, j12], [j21, j22]], dtype=np.float64)

    def predict(self, observation_or_env: Any, state: Any = None, deterministic: bool = True) -> Tuple[np.ndarray, Any]:
        """
        Interfaz compatible con Stable-Baselines3 y evaluador modular.
        Retorna (action, state).
        """
        if hasattr(observation_or_env, "_get_obs"):
            obs = observation_or_env._get_obs()
        else:
            obs = np.asarray(observation_or_env, dtype=np.float64)

        if obs.ndim > 1:
            # Procesamiento por lotes si corresponde
            actions = [self._predict_single(o) for o in obs]
            return np.array(actions, dtype=np.float32), state

        action = self._predict_single(obs)
        return action.astype(np.float32), state

    def _predict_single(self, obs: np.ndarray) -> np.ndarray:
        """
        Calcula el torque a partir del vector de observación de 10 dimensiones de Reacher-v5:
            obs[0]: cos(th1), obs[1]: cos(th2)
            obs[2]: sin(th1), obs[3]: sin(th2)
            obs[4]: target_x, obs[5]: target_y
            obs[6]: dtheta1,  obs[7]: dtheta2
            obs[8]: fingertip_x - target_x
            obs[9]: fingertip_y - target_y
        """
        # 1. Error de posición cartesiana e_pos = p_target - p_ee
        e_pos = -obs[8:10]

        # 2. Configuración articular y velocidades
        th1 = math.atan2(float(obs[2]), float(obs[0]))
        th2 = math.atan2(float(obs[3]), float(obs[1]))
        qvel = obs[6:8]

        # 3. Jacobiano cinemático en el efector final
        J = self._compute_jacobian(th1, th2)

        # 4. Torque operacional mediante transposición del Jacobiano y amortiguamiento derivativo
        tau = J.T @ (self.kp * e_pos) - self.kd * qvel

        # 5. Saturación al espacio de acción normalizado [-1.0, 1.0]
        action = np.clip(tau, -1.0, 1.0)
        return action

    def get_action(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Método directo de inferencia para bucles de control convencionales."""
        action, _ = self.predict(obs)
        return action


# Alias semántico formal
JacobianTransposePDController = JacobianDampedController
