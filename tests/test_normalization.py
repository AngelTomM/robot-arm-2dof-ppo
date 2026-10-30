"""
Pruebas Unitarias de Normalización, Entorno y Métricas LQR (MuJoCo Reacher-v5).
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
        """Verifica que Monitor registre la recompensa física sin normalizar en el entorno MuJoCo."""
        def make_env():
            return Monitor(Arm2DEnv())

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
            return Monitor(Arm2DEnv())

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
        """Verifica la sincronización de estadísticas entre entornos vectorizados."""
        def make_env():
            return Monitor(Arm2DEnv())

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
    def test_quadratic_lqr_exact_calculation(self):
        """Verifica el cálculo matemático exacto de la recompensa cuadrática continua."""
        env = Arm2DEnv()
        env.reset(seed=123)
        action = np.array([0.5, -0.2], dtype=np.float32)
        _, reward, terminated, truncated, info = env.step(action)

        pos_err_sq = info["pos_error_sq"]
        ctrl_eff_sq = info["ctrl_effort_sq"]
        vel_sq = info["vel_sq"]

        expected_reward = -(env.w_pos * pos_err_sq + env.w_ctrl * ctrl_eff_sq + env.w_vel * vel_sq)

        self.assertAlmostEqual(reward, expected_reward, places=5)
        self.assertFalse(terminated)
        env.close()

    def test_horizon_truncation_at_max_steps(self):
        """Verifica que el entorno corte exactamente en max_steps (horizonte fijo sin terminación prematura)."""
        env = Arm2DEnv(config={"env": {"max_steps": 200}})
        env.reset(seed=42)

        for s in range(1, 201):
            action = np.array([0.0, 0.0], dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            self.assertFalse(terminated, f"terminated no debe ser True en paso {s}")
            if s < 200:
                self.assertFalse(truncated, f"truncated no debe ser True antes del paso 200 (paso {s})")
            else:
                self.assertTrue(truncated, "truncated debe ser True exactamente en paso 200")

        env.close()


class TestPhaseDEvaluationMetrics(unittest.TestCase):
    def test_stabilization_metrics(self):
        """Verifica el cálculo de estabilización en los últimos k=20 pasos bajo radio de tolerancia 0.02 m."""
        # Caso de éxito: distancias <= 0.02m y velocidades <= 0.10rad/s
        distances_ok = [0.1] * 180 + [0.015] * 20
        speeds_ok = [0.5] * 180 + [0.05] * 20
        stab_ok = compute_stabilization_metrics(distances_ok, speeds_ok, k=20)
        self.assertTrue(stab_ok["is_success"])
        self.assertAlmostEqual(stab_ok["mean_dist_last_k"], 0.015)
        self.assertAlmostEqual(stab_ok["mean_vel_last_k"], 0.05)

        # Caso de fallo por velocidad residual elevada
        distances_fast = [0.1] * 180 + [0.01] * 20
        speeds_fast = [0.5] * 180 + [0.25] * 20
        stab_fast = compute_stabilization_metrics(distances_fast, speeds_fast, k=20)
        self.assertFalse(stab_fast["is_success"])

        # Caso de fallo por distancia residual
        distances_far = [0.1] * 180 + [0.05] * 20
        speeds_far = [0.5] * 180 + [0.02] * 20
        stab_far = compute_stabilization_metrics(distances_far, speeds_far, k=20)
        self.assertFalse(stab_far["is_success"])

    def test_canonical_lqr_cost(self):
        """Verifica el cálculo canónico de J_LQR con pesos normalizados."""
        distances = [0.05, 0.02]
        speeds = [0.1, 0.05]
        actions = [np.array([0.5, 0.5]), np.array([0.2, 0.1])]

        lqr = compute_lqr_canonical_cost(distances, speeds, actions)
        j_pos = 1.0 * (0.05**2 + 0.02**2)
        j_vel = 0.05 * (0.1**2 + 0.05**2)
        j_ctrl = 0.01 * (0.25 + 0.25 + 0.04 + 0.01)
        expected_total = -(j_pos + j_vel + j_ctrl)

        self.assertAlmostEqual(lqr["j_lqr"], expected_total, places=6)
        self.assertAlmostEqual(lqr["j_pos"], j_pos, places=6)
        self.assertAlmostEqual(lqr["j_vel"], j_vel, places=6)
        self.assertAlmostEqual(lqr["j_ctrl"], j_ctrl, places=6)


if __name__ == "__main__":
    unittest.main()
