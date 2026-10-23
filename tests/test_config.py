"""
Pruebas Unitarias para el Desacoplamiento de Configuraciones y Controlador Jacobiano C0 (MuJoCo Reacher-v5).
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
        """Verifica que load_env_config cargue todos los parámetros invariantes de MuJoCo Reacher-v5."""
        env_dict = load_env_config()
        self.assertIn("env", env_dict)
        env = env_dict["env"]

        # Parámetros físicos, inerciales y temporales
        self.assertEqual(env["l1"], 0.1)
        self.assertEqual(env["l2"], 0.11)
        self.assertAlmostEqual(env["joint1_limit_deg"], 171.887, places=2)
        self.assertEqual(env["target_radius"], 0.02)
        self.assertEqual(env["max_steps"], 100)
        self.assertEqual(env["dt"], 0.01)

        self.assertEqual(env["frame_skip"], 2)
        self.assertEqual(env["torque_limit"], 1.0)
        self.assertEqual(env["gear"], 200.0)
        self.assertEqual(env["success_mode"], "none")
        self.assertEqual(env["reward_type"], "quadratic_lqr")
        self.assertIn("reward_params", env)

    def test_load_agent_configs_fusion(self):
        """Verifica que los YAML de los agentes fusionen automáticamente el entorno común MuJoCo."""
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
            self.assertEqual(cfg["env"]["l1"], 0.1)
            self.assertEqual(cfg["env"]["dt"], 0.01)
            self.assertEqual(cfg["env"]["frame_skip"], 2)

            # Instanciar el entorno Arm2DEnv para asegurar compatibilidad
            env = Arm2DEnv(config=cfg)
            self.assertEqual(env.l1, 0.1)
            self.assertEqual(env.l2, 0.11)
            self.assertEqual(env.action_space.shape, (2,))
            self.assertEqual(env.observation_space.shape, (10,))
            env.close()

    def test_jacobian_controller_env_coupling(self):
        """Verifica que JacobianDampedController opere con la configuración del entorno."""
        env_cfg = load_env_config()
        ctrl = JacobianDampedController.from_config(env_cfg, kp=50.0, kd=1.5)
        self.assertEqual(ctrl.l1, 0.1)
        self.assertEqual(ctrl.l2, 0.11)

        env = Arm2DEnv(config=env_cfg)
        obs, _ = env.reset(seed=42)
        action, _ = ctrl.predict(obs)

        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))
        env.close()


if __name__ == "__main__":
    unittest.main()
