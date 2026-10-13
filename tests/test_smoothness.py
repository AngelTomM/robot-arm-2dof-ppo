"""
Pruebas Unitarias de la Convención de Suavidad y Acción Previa (Secciones 14.1 y 14.2).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np
from arm2d_env import Arm2DEnv


class TestSmoothness(unittest.TestCase):
    def test_smoothness_convention(self):
        cfg = {
            "env": {
                "reward_type": "x6_smooth",
                "reward_params": {
                    "w_a": 0.5
                }
            }
        }
        env = Arm2DEnv(config=cfg)
        env.reset(seed=42)
        
        # 1. Paso 0: Primera acción no debe generar penalización de suavidad
        action_1 = np.array([0.8, -0.4], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action_1)
        
        self.assertEqual(info["reward_smooth"], 0.0)
        self.assertIsNone(info["delta_action_norm"])
        
        # 2. Paso 1: Dos acciones iguales consecutivas deben dar penalización nula
        obs, reward, terminated, truncated, info = env.step(action_1)
        self.assertEqual(info["reward_smooth"], 0.0)
        self.assertEqual(info["delta_action_norm"], 0.0)
        
        # 3. Paso 2: Acción diferente debe dar penalización proporcional a ||delta_a||^2
        action_2 = np.array([0.4, 0.0], dtype=np.float32)
        diff = action_2 - action_1 # [-0.4, 0.4] -> sum(diff^2) = 0.32
        expected_penalty = -0.5 * 0.32 # -0.16
        
        obs, reward, terminated, truncated, info = env.step(action_2)
        np.testing.assert_allclose(info["reward_smooth"], expected_penalty, atol=1e-5)
        np.testing.assert_allclose(info["delta_action_norm"], np.linalg.norm(diff), atol=1e-5)
        
        # 4. Reset: Indicador se reinicia y no hay penalización entre episodios
        env.reset(seed=99)
        self.assertFalse(env.has_prev_action)
        self.assertIsNone(env.prev_action)
        
        action_3 = np.array([-0.5, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action_3)
        self.assertEqual(info["reward_smooth"], 0.0)
        self.assertIsNone(info["delta_action_norm"])


if __name__ == "__main__":
    unittest.main()
