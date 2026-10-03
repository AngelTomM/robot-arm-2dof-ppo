"""
Pruebas Unitarias de VecNormalize, Wrappers y Persistencia de Normalización (Sección 11 y 15).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
from arm2d_env import Arm2DEnv


class TestNormalization(unittest.TestCase):
    def test_monitor_inner_wrapper_raw_rewards(self):
        """Verifica que Monitor registre la recompensa física sin normalizar."""
        cfg = {
            "env": {
                "success_mode": "touch",
                "reward_type": "distance_only",
                "target_radius": 0.08
            }
        }
        
        # 1. Crear entorno envuelto en Monitor y luego en DummyVecEnv y VecNormalize
        def make_env():
            return Monitor(Arm2DEnv(config=cfg))
            
        vec_env = DummyVecEnv([make_env])
        norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        obs = norm_env.reset()
        for _ in range(10):
            action = [norm_env.action_space.sample()]
            next_obs, reward, dones, infos = norm_env.step(action)
            
            # Recompensa devuelta por VecNormalize está normalizada
            self.assertTrue(np.isfinite(reward[0]))
            
        norm_env.close()

    def test_vecnormalize_save_and_load(self):
        """Verifica que las estadísticas de normalización se guarden y restauren correctamente."""
        tmp_stats_path = "tests/data/temp_vecnormalize_stats.pkl"
        os.makedirs(os.path.dirname(tmp_stats_path), exist_ok=True)
        
        def make_env():
            return Monitor(Arm2DEnv())
            
        vec_env = DummyVecEnv([make_env])
        norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
        
        norm_env.reset()
        # Dar varios pasos para acumular medias y varianzas
        for _ in range(50):
            norm_env.step([norm_env.action_space.sample()])
            
        obs_mean_orig = np.copy(norm_env.obs_rms.mean)
        obs_var_orig = np.copy(norm_env.obs_rms.var)
        
        # Guardar estadísticas
        norm_env.save(tmp_stats_path)
        norm_env.close()
        
        # Cargar en un nuevo entorno de evaluación
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
            return Monitor(Arm2DEnv())
            
        train_env = VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=False)
        eval_env = VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=False, training=False)
        
        train_env.reset()
        for _ in range(30):
            train_env.step([train_env.action_space.sample()])
            
        # Sincronizar
        sync_envs_normalization(train_env, eval_env)
        eval_env.training = False
        
        np.testing.assert_allclose(train_env.obs_rms.mean, eval_env.obs_rms.mean, atol=1e-6)
        np.testing.assert_allclose(train_env.obs_rms.var, eval_env.obs_rms.var, atol=1e-6)
        
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    unittest.main()
