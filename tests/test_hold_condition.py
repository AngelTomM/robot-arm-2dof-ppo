"""
Pruebas Unitarias de la Condición de Permanencia (Hold at Target, Sección 5 y 15).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np
from arm2d_env import Arm2DEnv


class TestHoldCondition(unittest.TestCase):
    def test_high_speed_entry_no_success(self):
        """Entrar al radio con velocidad alta NO debe dar éxito en modo hold."""
        cfg = {
            "env": {
                "success_mode": "hold",
                "hold_steps": 5,
                "velocity_tolerance": 0.1,
                "target_radius": 0.08,
                "max_speed": 2.0
            }
        }
        env = Arm2DEnv(config=cfg)
        env.reset(seed=1)
        
        _, ee = env._forward_kinematics(0.0, 0.0)
        env.set_state_and_target(0.0, 0.0, ee[0], ee[1])
        
        # Aplicar acción máxima (velocidad = 2.0 > 0.1)
        action = np.array([1.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        self.assertEqual(info["hold_counter"], 0)
        self.assertFalse(terminated)
        self.assertFalse(info["is_success"])

    def test_hold_consecutive_steps_success(self):
        """Completar todos los pasos consecutivos requeridos debe dar éxito."""
        cfg = {
            "env": {
                "success_mode": "hold",
                "hold_steps": 5,
                "velocity_tolerance": 0.2,
                "target_radius": 0.1,
                "max_speed": 2.0,
                "reward_params": {"success_bonus": 15.0}
            }
        }
        env = Arm2DEnv(config=cfg)
        env.reset(seed=1)
        
        _, ee = env._forward_kinematics(0.5, -0.5)
        env.set_state_and_target(0.5, -0.5, ee[0], ee[1])
        
        # Aplicar acción de velocidad cero durante 4 pasos
        action_zero = np.array([0.0, 0.0], dtype=np.float32)
        for step in range(4):
            obs, reward, terminated, truncated, info = env.step(action_zero)
            self.assertEqual(info["hold_counter"], step + 1)
            self.assertFalse(terminated)
            self.assertEqual(info["reward_success"], 0.0)
            
        # Paso 5: Debe completar hold_steps = 5 y otorgar el bonus exactamente una vez
        obs, reward, terminated, truncated, info = env.step(action_zero)
        self.assertEqual(info["hold_counter"], 5)
        self.assertTrue(terminated)
        self.assertTrue(info["is_success"])
        self.assertEqual(info["reward_success"], 15.0)

    def test_leaving_target_resets_counter(self):
        """Salir del radio objetivo debe reiniciar el contador de permanencia a cero."""
        cfg = {
            "env": {
                "success_mode": "hold",
                "hold_steps": 10,
                "velocity_tolerance": 0.5,
                "target_radius": 0.1,
                "max_speed": 2.0
            }
        }
        env = Arm2DEnv(config=cfg)
        env.reset(seed=1)
        
        _, ee = env._forward_kinematics(0.0, 0.0)
        env.set_state_and_target(0.0, 0.0, ee[0], ee[1])
        
        # 2 pasos dentro y quietos
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        self.assertEqual(info["hold_counter"], 2)
        
        # Moverse lejos del target
        env.step(np.array([1.0, 1.0], dtype=np.float32))
        _, _, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
        self.assertEqual(info["hold_counter"], 0)

    def test_truncation_without_success(self):
        """Si no se completa la permanencia tras max_steps, el episodio se trunca y terminated es False."""
        cfg = {
            "env": {
                "success_mode": "hold",
                "hold_steps": 20,
                "max_steps": 10
            }
        }
        env = Arm2DEnv(config=cfg)
        env.reset(seed=1)
        
        for step in range(9):
            obs, reward, terminated, truncated, info = env.step(np.array([0.5, -0.5], dtype=np.float32))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            
        # Paso 10: Truncamiento
        obs, reward, terminated, truncated, info = env.step(np.array([0.5, -0.5], dtype=np.float32))
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertFalse(info["is_success"])

    def test_progressive_hold_configurations(self):
        """Verifica que las variantes progresivas lean exactamente sus parámetros."""
        variants = [
            ("relaxed", 5, 0.3),
            ("intermediate", 10, 0.2),
            ("strict", 20, 0.1)
        ]
        for name, steps, vel_tol in variants:
            cfg = {
                "env": {
                    "success_mode": "hold",
                    "hold_steps": steps,
                    "velocity_tolerance": vel_tol
                }
            }
            env = Arm2DEnv(config=cfg)
            self.assertEqual(env.hold_steps, steps)
            self.assertAlmostEqual(env.velocity_tolerance, vel_tol, places=5)


if __name__ == "__main__":
    unittest.main()
