"""
Pruebas Unitarias para el Desacoplamiento de Configuraciones y Controlador Jacobiano C0.
"""
import sys
from pathlib import Path
import unittest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import load_config, load_env_config, JacobianDampedController, Arm2DEnv


class TestConfigDecoupling(unittest.TestCase):
    def test_load_env_config_structure(self):
        """Verifica que load_env_config cargue todos los parámetros invariantes del entorno."""
        env_dict = load_env_config()
        self.assertIn("env", env_dict)
        env = env_dict["env"]

        # Parámetros físicos e invariantes de comparación
        self.assertEqual(env["l1"], 1.0)
        self.assertEqual(env["l2"], 1.0)
        self.assertEqual(env["joint2_limit_deg"], 170.0)
        self.assertEqual(env["max_speed"], 1.5)
        self.assertEqual(env["target_radius"], 0.06)
        self.assertEqual(env["max_steps"], 200)
        self.assertEqual(env["dt"], 0.05)
        self.assertEqual(env["success_mode"], "none")
        self.assertEqual(env["reward_type"], "lyapunov_lqr")
        self.assertIn("reward_params", env)

    def test_load_agent_configs_fusion(self):
        """Verifica que los YAML de los agentes fusionen automáticamente el entorno común."""
        agent_configs = [
            "configs/phase_D/prueba/PPO_1.yaml",
            "configs/phase_D/prueba/SAC_1.yaml",
            "configs/phase_D/smoke/PPO_smoke.yaml",
            "configs/phase_D/smoke/SAC_smoke.yaml",
            "configs/phase_D/prueba/C0.yaml"
        ]

        for rel_path in agent_configs:
            cfg = load_config(rel_path)
            self.assertIn("env", cfg, f"Falta sección 'env' fusionada en {rel_path}")
            self.assertEqual(cfg["env"]["joint2_limit_deg"], 170.0)
            self.assertEqual(cfg["env"]["max_speed"], 1.5)
            self.assertEqual(cfg["env"]["dt"], 0.05)

            # Instanciar el entorno Arm2DEnv para asegurar compatibilidad
            env = Arm2DEnv(config=cfg)
            self.assertEqual(env.joint2_limit_deg, 170.0)
            self.assertEqual(env.max_speed, 1.5)
            self.assertAlmostEqual(env.min_reach, 2.0 * np.sin(np.radians(5)), places=5)

    def test_jacobian_controller_env_coupling(self):
        """Verifica que JacobianDampedController comparta la configuración del entorno."""
        env_cfg = load_env_config()
        ctrl = JacobianDampedController(kp=3.0, lambda_damping=0.01, env_config=env_cfg)
        self.assertEqual(ctrl.max_speed, 1.5)

        env = Arm2DEnv(config=env_cfg)
        env.reset(seed=42)
        action = ctrl.predict(env)

        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))


if __name__ == "__main__":
    unittest.main()
