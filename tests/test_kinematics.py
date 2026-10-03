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


if __name__ == "__main__":
    unittest.main()
