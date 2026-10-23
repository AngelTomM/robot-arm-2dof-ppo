"""
Pruebas Unitarias de Cinemática Directa, Jacobiano y Coherencia de Estados (Sección 15).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import unittest
import numpy as np
from arm2d_env import Arm2DEnv


class TestKinematics(unittest.TestCase):
    def test_forward_kinematics_known_points(self):
        env = Arm2DEnv()
        
        # Caso 1: Brazo completamente extendido a lo largo de eje X (theta1=0, theta2=0)
        elbow, ee = env._forward_kinematics(0.0, 0.0)
        np.testing.assert_allclose(elbow, [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(ee, [2.0, 0.0], atol=1e-6)
        
        # Caso 2: Brazo apuntando arriba a lo largo de eje Y (theta1=pi/2, theta2=0)
        elbow, ee = env._forward_kinematics(math.pi / 2, 0.0)
        np.testing.assert_allclose(elbow, [0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(ee, [0.0, 2.0], atol=1e-6)
        
        # Caso 3: Codo a 90 grados (theta1=0, theta2=pi/2) -> elbow=(1,0), ee=(1,1)
        elbow, ee = env._forward_kinematics(0.0, math.pi / 2)
        np.testing.assert_allclose(elbow, [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(ee, [1.0, 1.0], atol=1e-6)
        
        # Caso 4: Doblado sobre sí mismo (theta1=0, theta2=pi) -> elbow=(1,0), ee=(0,0)
        elbow, ee = env._forward_kinematics(0.0, math.pi)
        np.testing.assert_allclose(elbow, [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(ee, [0.0, 0.0], atol=1e-6)

    def test_jacobian_analytical_vs_numerical(self):
        env = Arm2DEnv()
        angles = [
            (0.2, 0.5),
            (-1.2, 0.8),
            (2.1, -1.5),
            (0.0, math.pi / 4)
        ]
        eps = 1e-6
        
        def fk_64(th1, th2):
            x1 = env.l1 * math.cos(th1)
            y1 = env.l1 * math.sin(th1)
            x2 = x1 + env.l2 * math.cos(th1 + th2)
            y2 = y1 + env.l2 * math.sin(th1 + th2)
            return np.array([x2, y2], dtype=np.float64)
        
        for th1, th2 in angles:
            J_analytic = env.get_jacobian(th1, th2)
            
            ee_plus1 = fk_64(th1 + eps, th2)
            ee_minus1 = fk_64(th1 - eps, th2)
            d_ee_dth1 = (ee_plus1 - ee_minus1) / (2.0 * eps)
            
            ee_plus2 = fk_64(th1, th2 + eps)
            ee_minus2 = fk_64(th1, th2 - eps)
            d_ee_dth2 = (ee_plus2 - ee_minus2) / (2.0 * eps)
            
            J_numeric = np.column_stack([d_ee_dth1, d_ee_dth2]).astype(np.float32)
            np.testing.assert_allclose(J_analytic, J_numeric, rtol=1e-4, atol=1e-5)

    def test_observation_state_consistency(self):
        env = Arm2DEnv()
        obs, info = env.reset(seed=123)
        
        # Formato de obs: [cos(th1), sin(th1), cos(th2), sin(th2), dth1, dth2, tgt_x, tgt_y, dx, dy]
        self.assertEqual(len(obs), 10)
        np.testing.assert_allclose(obs[0], math.cos(env.theta1), atol=1e-6)
        np.testing.assert_allclose(obs[1], math.sin(env.theta1), atol=1e-6)
        np.testing.assert_allclose(obs[2], math.cos(env.theta2), atol=1e-6)
        np.testing.assert_allclose(obs[3], math.sin(env.theta2), atol=1e-6)
        np.testing.assert_allclose(obs[4], env.dtheta1, atol=1e-6)
        np.testing.assert_allclose(obs[5], env.dtheta2, atol=1e-6)
        np.testing.assert_allclose(obs[6:8], env.target, atol=1e-6)
        
        _, ee = env._forward_kinematics(env.theta1, env.theta2)
        dist_vec = env.target - ee
        np.testing.assert_allclose(obs[8:10], dist_vec, atol=1e-6)


class TestJointLimits(unittest.TestCase):
    def test_joint2_reset_within_limits(self):
        """Verifica que en 500 llamadas a reset(), theta2 respete estrictamente [-170°, 170°]."""
        env = Arm2DEnv()
        limit_rad = math.radians(170)
        for seed in range(500):
            env.reset(seed=seed)
            self.assertGreaterEqual(env.theta2, -limit_rad - 1e-6)
            self.assertLessEqual(env.theta2, limit_rad + 1e-6)
            target_dist = np.linalg.norm(env.target)
            self.assertGreaterEqual(target_dist, env.min_reach - 1e-6)

    def test_joint2_hard_stops_and_velocity_halt(self):
        """Verifica que acciones hacia el tope detengan theta2 y anulen dtheta2, y que la acción contraria despegue."""
        env = Arm2DEnv()
        limit_rad = math.radians(170)
        env.reset(seed=42)

        # Colocar el brazo muy cerca del tope positivo (+169°)
        near_limit = math.radians(169)
        env.set_state_and_target(0.0, near_limit, 1.0, 1.0)

        # Ordenar velocidad positiva máxima hacia el tope (action[1] = 1.0)
        action_forward = np.array([0.0, 1.0], dtype=np.float32)
        env.step(action_forward)

        # Debe chocar contra el límite exacto y anular su velocidad articular efectiva
        self.assertAlmostEqual(env.theta2, limit_rad, places=5)
        self.assertEqual(env.dtheta2, 0.0)

        # Intentar seguir empujando hacia adelante: debe permanecer en el tope
        env.step(action_forward)
        self.assertAlmostEqual(env.theta2, limit_rad, places=5)
        self.assertEqual(env.dtheta2, 0.0)

        # Ordenar acción en reversa (action[1] = -1.0): debe despegar de inmediato
        action_reverse = np.array([0.0, -1.0], dtype=np.float32)
        env.step(action_reverse)
        self.assertLess(env.theta2, limit_rad)
        self.assertLess(env.dtheta2, 0.0)

    def test_set_state_clips_joint2(self):
        """Verifica que set_state_and_target recorte theta2 fuera de rango al límite físico."""
        env = Arm2DEnv()
        limit_rad = math.radians(170)

        # Intentar fijar theta2 = pi (doblado completo en colisión)
        env.set_state_and_target(0.0, math.pi, 1.0, 1.0)
        self.assertAlmostEqual(env.theta2, limit_rad, places=5)

        # Intentar fijar theta2 = -pi
        env.set_state_and_target(0.0, -math.pi, 1.0, 1.0)
        self.assertAlmostEqual(env.theta2, -limit_rad, places=5)

    def test_minimum_reach_geometry(self):
        """Verifica que la distancia del efector a la base en cualquier estado admisible sea >= r_min."""
        env = Arm2DEnv()
        expected_min_r = 2.0 * math.sin(math.radians(5))
        self.assertAlmostEqual(env.min_reach, expected_min_r, places=5)

        for th2_deg in range(-170, 171, 10):
            th2 = math.radians(th2_deg)
            for th1_deg in range(-180, 180, 45):
                th1 = math.radians(th1_deg)
                _, ee = env._forward_kinematics(th1, th2)
                dist_base = float(np.linalg.norm(ee))
                self.assertGreaterEqual(dist_base, expected_min_r - 1e-5)


if __name__ == "__main__":
    unittest.main()
