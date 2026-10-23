"""
Controlador Cinemático Proporcional con Pseudoinversa Amortiguada del Jacobiano (C0) (Sección 10).
"""
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import yaml


class JacobianDampedController:
    """
    Controlador Cinemático Clásico sin Aprendizaje.
    Implementa:
      e = p_tgt - p_ee
      v_ref = Kp * e
      J_pinv = J^T @ inv(J @ J^T + lambda_damping^2 * I)
      theta_dot_ref = J_pinv @ v_ref
      action = clip(theta_dot_ref / vmax, -1.0, 1.0)
    """
    def __init__(self, kp: float = 1.0, lambda_damping: float = 0.05,
                 max_speed: Optional[float] = None,
                 env_config: Optional[Union[Dict[str, Any], str, Path]] = None):
        self.kp = float(kp)
        self.lambda_damping = float(lambda_damping)

        # Si se suministra env_config, extraer max_speed del entorno
        if env_config is not None:
            resolved_speed = self._extract_max_speed(env_config)
            self.max_speed = resolved_speed if resolved_speed is not None else (float(max_speed) if max_speed is not None else 1.5)
        else:
            self.max_speed = float(max_speed) if max_speed is not None else None

    @staticmethod
    def _extract_max_speed(cfg: Union[Dict[str, Any], str, Path]) -> Optional[float]:
        """Extrae la velocidad articular máxima a partir de un diccionario o archivo YAML."""
        if isinstance(cfg, (str, Path)):
            p = Path(cfg)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    cfg_dict = yaml.safe_load(f) or {}
            else:
                return None
        elif isinstance(cfg, dict):
            cfg_dict = cfg
        else:
            return None

        env_section = cfg_dict.get("env", cfg_dict)
        if isinstance(env_section, dict) and "max_speed" in env_section:
            return float(env_section["max_speed"])
        return None

    @classmethod
    def from_config(cls, env_config: Optional[Union[Dict[str, Any], str, Path]] = None,
                    kp: float = 3.0, lambda_damping: float = 0.01) -> "JacobianDampedController":
        """
        Instancia el controlador a partir del archivo o diccionario unificado del entorno.
        Si env_config es None, busca 'configs/environment.yaml' de manera predeterminada.
        """
        if env_config is None:
            repo_root = Path(__file__).resolve().parents[2]
            default_env = repo_root / "configs" / "environment.yaml"
            env_config = default_env if default_env.exists() else None

        return cls(kp=kp, lambda_damping=lambda_damping, env_config=env_config)

    def predict(self, env) -> np.ndarray:
        """Calcula la acción a partir del estado actual del entorno."""
        # 1. Posición del efector final y error cartesiano
        _, end_effector = env._forward_kinematics(env.theta1, env.theta2)
        error = env.target - end_effector

        # 2. Velocidad cartesiana de referencia
        v_ref = self.kp * error

        # 3. Jacobiano analítico
        J = env.get_jacobian(env.theta1, env.theta2)  # Shape (2, 2)

        # 4. Pseudoinversa Amortiguada (Damped Least Squares)
        lambda_sq = self.lambda_damping ** 2
        damped_matrix = J @ J.T + lambda_sq * np.eye(2, dtype=np.float32)
        inv_damped = np.linalg.inv(damped_matrix)
        J_damped_pinv = J.T @ inv_damped

        # 5. Velocidad articular requerida
        theta_dot_ref = J_damped_pinv @ v_ref

        # 6. Acción normalizada en [-1.0, 1.0] respetando la velocidad máxima del entorno
        vmax = getattr(env, "max_speed", self.max_speed) or (self.max_speed if self.max_speed is not None else 1.5)
        action = np.clip(theta_dot_ref / vmax, -1.0, 1.0)
        return action.astype(np.float32)
