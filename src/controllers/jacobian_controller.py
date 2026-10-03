"""
Controlador Cinemático Proporcional con Pseudoinversa Amortiguada del Jacobiano (C0) (Sección 10).
"""
import math
import numpy as np

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
    def __init__(self, kp: float = 1.0, lambda_damping: float = 0.05, max_speed: float = 2.0):
        self.kp = float(kp)
        self.lambda_damping = float(lambda_damping)
        self.max_speed = float(max_speed)

    def predict(self, env) -> np.ndarray:
        """Calcula la acción a partir del estado actual del entorno."""
        # 1. Posición del efector final y error cartesiano
        _, end_effector = env._forward_kinematics(env.theta1, env.theta2)
        error = env.target - end_effector

        # 2. Velocidad cartesiana de referencia
        v_ref = self.kp * error

        # 3. Jacobiano analítico
        J = env.get_jacobian(env.theta1, env.theta2) # Shape (2, 2)

        # 4. Pseudoinversa Amortiguada (Damped Least Squares)
        lambda_sq = self.lambda_damping ** 2
        damped_matrix = J @ J.T + lambda_sq * np.eye(2, dtype=np.float32)
        inv_damped = np.linalg.inv(damped_matrix)
        J_damped_pinv = J.T @ inv_damped

        # 5. Velocidad articular requerida
        theta_dot_ref = J_damped_pinv @ v_ref

        # 6. Acción normalizada en [-1.0, 1.0]
        action = np.clip(theta_dot_ref / self.max_speed, -1.0, 1.0)
        return action.astype(np.float32)
