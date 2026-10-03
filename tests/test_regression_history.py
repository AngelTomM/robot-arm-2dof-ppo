"""
Test de Regresión Histórica (Sección 4.3).
Compara la salida del entorno modificado en modo histórico contra las referencias grabadas.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import yaml
import unittest
import numpy as np
from arm2d_env import Arm2DEnv


class TestHistoricalRegression(unittest.TestCase):
    def test_historical_regression(self):
        ref_file = os.path.join(os.path.dirname(__file__), "data", "historical_regression_reference.json")
        self.assertTrue(os.path.exists(ref_file), f"No se encontró el archivo de referencia {ref_file}")
        
        with open(ref_file, "r", encoding="utf-8") as f:
            records = json.load(f)
            
        test_configs = {
            "default": None,
            "exp1_baseline": "configs/exp1_baseline.yaml",
            "exp2_smooth_precision": "configs/exp2_smooth_precision.yaml"
        }
        
        for cfg_name, cfg_path in test_configs.items():
            self.assertIn(cfg_name, records, f"Config {cfg_name} no está en las referencias")
            if cfg_path:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
            else:
                cfg = None
            
            for seed_str, ep_data in records[cfg_name].items():
                seed = int(seed_str)
                env = Arm2DEnv(config=cfg)
                obs, info = env.reset(seed=seed)
                
                # Comparar observación inicial
                np.testing.assert_allclose(obs, np.array(ep_data["initial_obs"]), rtol=1e-5, atol=1e-6,
                                           err_msg=f"Discrepancia en obs inicial {cfg_name} seed {seed}")
                
                for step_data in ep_data["steps"]:
                    action = np.array(step_data["action"], dtype=np.float32)
                    next_obs, reward, terminated, truncated, step_info = env.step(action)
                    
                    # Tolerancias numéricas explícitas
                    np.testing.assert_allclose(next_obs, np.array(step_data["obs"]), rtol=1e-4, atol=1e-5,
                                               err_msg=f"Discrepancia en obs paso {step_data['step_idx']}")
                    np.testing.assert_allclose(reward, step_data["reward"], rtol=1e-4, atol=1e-5,
                                               err_msg=f"Discrepancia en reward paso {step_data['step_idx']}")
                    
                    # Coincidencia booleana exacta
                    self.assertEqual(terminated, step_data["terminated"], f"Discrepancia en terminated paso {step_data['step_idx']}")
                    self.assertEqual(truncated, step_data["truncated"], f"Discrepancia en truncated paso {step_data['step_idx']}")
                    
                    np.testing.assert_allclose(env.theta1, step_data["theta1"], rtol=1e-4, atol=1e-5)
                    np.testing.assert_allclose(env.theta2, step_data["theta2"], rtol=1e-4, atol=1e-5)
                    np.testing.assert_allclose(env.dtheta1, step_data["dtheta1"], rtol=1e-4, atol=1e-5)
                    np.testing.assert_allclose(env.dtheta2, step_data["dtheta2"], rtol=1e-4, atol=1e-5)
                    
                env.close()


if __name__ == "__main__":
    unittest.main()
