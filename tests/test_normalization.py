"""
Pruebas Unitarias de Normalización y Entorno Fase D (MDP LQR Continuo).
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import unittest
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
from src import (
    Arm2DEnv,
    compute_stabilization_metrics,
    compute_lqr_canonical_cost,
    TARGET_RADIUS_EVAL,
    VEL_TOL_EVAL,
    W_POS_EVAL,
    W_VEL_EVAL,
    W_CTRL_EVAL,
)


class TestNormalization(unittest.TestCase):
    def test_monitor_inner_wrapper_raw_rewards(self):
        """Verifica que Monitor registre la recompensa física sin normalizar en Fase D."""
        cfg = {
            "env": {
                "success_mode": "none",
                "reward_type": "distance_only",
                "target_radius": 0.08
            }
        }

        def make_env():
            return Monitor(Arm2DEnv(config=cfg))

        vec_env = DummyVecEnv([make_env])
        norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        obs = norm_env.reset()
        for _ in range(10):
            action = [norm_env.action_space.sample()]
            next_obs, reward, dones, infos = norm_env.step(action)
            self.assertTrue(np.isfinite(reward[0]))

        norm_env.close()

    def test_vecnormalize_save_and_load(self):
        """Verifica que las estadísticas de normalización se guarden y restauren correctamente."""
        tmp_stats_path = str(REPO_ROOT / "tests" / "temp_vecnormalize_stats.pkl")

        def make_env():
            return Monitor(Arm2DEnv(config={"env": {"success_mode": "none"}}))

        vec_env = DummyVecEnv([make_env])
        norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

        norm_env.reset()
        for _ in range(50):
            norm_env.step([norm_env.action_space.sample()])

        obs_mean_orig = np.copy(norm_env.obs_rms.mean)
        obs_var_orig = np.copy(norm_env.obs_rms.var)

        norm_env.save(tmp_stats_path)
        norm_env.close()

        eval_vec_env = DummyVecEnv([make_env])
        loaded_eval_env = VecNormalize.load(tmp_stats_path, eval_vec_env)
        loaded_eval_env.training = False
        loaded_eval_env.norm_reward = False

        np.testing.assert_allclose(loaded_eval_env.obs_rms.mean, obs_mean_orig, atol=1e-6)
        np.testing.assert_allclose(loaded_eval_env.obs_rms.var, obs_var_orig, atol=1e-6)
        self.assertFalse(loaded_eval_env.training)
        self.assertFalse(loaded_eval_env.norm_reward)

        loaded_eval_env.close()
        if os.path.exists(tmp_stats_path):
            os.remove(tmp_stats_path)

    def test_sync_envs_normalization(self):
        """Verifica la sincronización de estadísticas entre entrenamiento y evaluación."""
        def make_env():
            return Monitor(Arm2DEnv(config={"env": {"success_mode": "none"}}))

        train_env = VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=False)
        eval_env = VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=False, training=False)

        train_env.reset()
        for _ in range(30):
            train_env.step([train_env.action_space.sample()])

        sync_envs_normalization(train_env, eval_env)
        eval_env.training = False

        np.testing.assert_allclose(train_env.obs_rms.mean, eval_env.obs_rms.mean, atol=1e-6)
        np.testing.assert_allclose(train_env.obs_rms.var, eval_env.obs_rms.var, atol=1e-6)

        train_env.close()
        eval_env.close()


class TestPhaseDEnv(unittest.TestCase):
    def test_lyapunov_lqr_exact_calculation(self):
        """Verifica el cálculo matemático exacto de lyapunov_lqr."""
        env = Arm2DEnv(config={
            "env": {
                "reward_type": "lyapunov_lqr",
                "success_mode": "none",
                "w_pos": 1.0,
                "w_vel": 0.05,
                "w_ctrl": 0.01,
            }
        })
        env.reset(seed=123)
        action = np.array([0.5, -0.2], dtype=np.float32)
        _, reward, terminated, truncated, info = env.step(action)

        dist = info["distance"]
        speed = info["joint_speed"]
        expected_dist = -1.0 * (dist ** 2)
        expected_vel = -0.05 * (speed ** 2)
        expected_ctrl = -0.01 * float(np.sum(np.square(action)))
        expected_total = expected_dist + expected_vel + expected_ctrl

        self.assertAlmostEqual(info["reward_dist"], expected_dist, places=9)
        self.assertAlmostEqual(info["reward_vel"], expected_vel, places=9)
        self.assertAlmostEqual(info["reward_action"], expected_ctrl, places=9)
        self.assertAlmostEqual(reward, expected_total, places=9)
        self.assertFalse(terminated)
        env.close()

    def test_distance_only_exact_calculation(self):
        """Verifica que distance_only sea exactamente -distance sin vel ni ctrl."""
        env = Arm2DEnv(config={
            "env": {
                "reward_type": "distance_only",
                "success_mode": "none",
            }
        })
        env.reset(seed=123)
        action = np.array([0.8, -0.4], dtype=np.float32)
        _, reward, terminated, truncated, info = env.step(action)

        self.assertAlmostEqual(reward, -info["distance"], places=9)
        self.assertEqual(info["reward_vel"], 0.0)
        self.assertEqual(info["reward_action"], 0.0)
        self.assertFalse(terminated)
        env.close()

    def test_rejection_of_obsolete_modes_and_rewards(self):
        """Verifica que success_mode != 'none' y reward_types obsoletos lancen ValueError."""
        with self.assertRaises(ValueError):
            Arm2DEnv(config={"env": {"success_mode": "hold"}})

        with self.assertRaises(ValueError):
            Arm2DEnv(config={"env": {"success_mode": "touch"}})

        for obsolete_rew in ["smooth_precision", "historical", "x5_near", "x5_vel", "x6_smooth", "energy_efficient"]:
            with self.assertRaises(ValueError):
                Arm2DEnv(config={"env": {"reward_type": obsolete_rew, "success_mode": "none"}})

    def test_terminated_always_false_and_truncates_at_max_steps(self):
        """Verifica que terminated siempre sea False y solo corte por truncamiento en max_steps=200."""
        env = Arm2DEnv(config={"env": {"max_steps": 200, "success_mode": "none"}})
        env.reset(seed=42)

        for s in range(1, 201):
            action = np.array([0.0, 0.0], dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            self.assertFalse(terminated, f"terminated no debe ser True en paso {s}")
            if s < 200:
                self.assertFalse(truncated, f"truncated no debe ser True antes de max_steps (paso {s})")
            else:
                self.assertTrue(truncated, "truncated debe ser True exactamente en paso 200")

        env.close()


class TestPhaseDEvaluationMetrics(unittest.TestCase):
    def test_option_2_stabilization(self):
        """Verifica el cálculo de la Opción 2 en los últimos k=20 pasos."""
        # Caso de éxito: distancias <= 0.06m y velocidades <= 0.10rad/s
        distances_ok = [0.2] * 180 + [0.04] * 20
        speeds_ok = [0.5] * 180 + [0.08] * 20
        stab_ok = compute_stabilization_metrics(distances_ok, speeds_ok, k=20)
        self.assertTrue(stab_ok["is_success"])
        self.assertAlmostEqual(stab_ok["mean_dist_last_k"], 0.04)
        self.assertAlmostEqual(stab_ok["mean_vel_last_k"], 0.08)

        # Caso de fallo por velocidad alta (llegada rápida sin frenado)
        distances_fast = [0.2] * 180 + [0.03] * 20
        speeds_fast = [0.5] * 180 + [0.25] * 20
        stab_fast = compute_stabilization_metrics(distances_fast, speeds_fast, k=20)
        self.assertFalse(stab_fast["is_success"])

        # Caso de fallo por distancia
        distances_far = [0.2] * 180 + [0.08] * 20
        speeds_far = [0.5] * 180 + [0.02] * 20
        stab_far = compute_stabilization_metrics(distances_far, speeds_far, k=20)
        self.assertFalse(stab_far["is_success"])

    def test_option_3_canonical_lqr_cost(self):
        """Verifica el cálculo canónico de J_LQR (Opción 3) con pesos fijos."""
        distances = [0.1, 0.05]
        speeds = [0.2, 0.1]
        actions = [np.array([0.5, 0.5]), np.array([0.2, 0.1])]

        # t=1: 1.0*(0.1^2) + 0.05*(0.2^2) + 0.01*(0.25+0.25) = 0.01 + 0.002 + 0.005 = 0.017
        # t=2: 1.0*(0.05^2) + 0.05*(0.1^2) + 0.01*(0.04+0.01) = 0.0025 + 0.0005 + 0.0005 = 0.0035
        # Total cost: -(0.017 + 0.0035) = -0.0205
        lqr = compute_lqr_canonical_cost(distances, speeds, actions)
        self.assertAlmostEqual(lqr["j_lqr"], -0.0205, places=6)
        self.assertAlmostEqual(lqr["j_pos"], 0.0125, places=6)
        self.assertAlmostEqual(lqr["j_vel"], 0.0025, places=6)
        self.assertAlmostEqual(lqr["j_ctrl"], 0.0055, places=6)


if __name__ == "__main__":
    unittest.main()

