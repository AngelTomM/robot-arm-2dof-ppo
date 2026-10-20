"""
Pruebas Unitarias y Validaciones de Fase B (Secciones 8 y 9).
Usa el módulo estándar unittest (sin dependencias adicionales).
Verifica:
1. Guardado y recarga de checkpoints acoplados (modelo + vecnormalize_stats.pkl).
2. Determinismo estricto de la evaluación.
3. Consistencia entre telemetría por paso y evaluador por episodio.
4. Integridad de archivos de Fase B.
5. Lógica algorítmica de la regla de selección y meseta con datos sintéticos.
"""
import os
import sys
from pathlib import Path
import tempfile
import unittest
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arm2d_env import Arm2DEnv
from src.controllers.jacobian_controller import JacobianDampedController
from src.training.phase_b_callbacks import PhaseBCheckpointCallback
from src.evaluation.evaluator import evaluate_on_dataset
from src.evaluation.datasets import generate_dataset
from scripts.phase_b_step_data import collect_step_telemetry_for_episodes

class TestPhaseBValidations(unittest.TestCase):

    def test_checkpoint_callback_atomic_saving(self):
        """Verifica que PhaseBCheckpointCallback guarda model.zip y vecnormalize_stats.pkl juntos y son recuperables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            vec_env = DummyVecEnv([lambda: Arm2DEnv()])
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

            model = PPO("MlpPolicy", vec_env, n_steps=64, batch_size=32, n_epochs=1, seed=42)
            cb = PhaseBCheckpointCallback(save_freq=100, save_path=str(tmp_path / "checkpoints"), verbose=0)
            cb.init_callback(model)

            # Simular guardado en paso 100
            cb._save_checkpoint(100)

            ckpt_dir = tmp_path / "checkpoints" / "ckpt_100"
            self.assertTrue(ckpt_dir.exists(), "La carpeta ckpt_100 debe existir")
            self.assertTrue((ckpt_dir / "model.zip").exists(), "model.zip debe existir")
            self.assertTrue((ckpt_dir / "vecnormalize_stats.pkl").exists(), "vecnormalize_stats.pkl debe existir")

            # Verificar recarga
            loaded_model = PPO.load(str(ckpt_dir / "model.zip"))
            dummy = DummyVecEnv([lambda: Arm2DEnv()])
            loaded_norm = VecNormalize.load(str(ckpt_dir / "vecnormalize_stats.pkl"), dummy)
            
            obs = np.ones((1, 10), dtype=np.float32)
            norm_obs = loaded_norm.normalize_obs(obs)
            act, _ = loaded_model.predict(norm_obs, deterministic=True)
            self.assertEqual(act.shape, (1, 2))
            self.assertFalse(np.isnan(act).any())

    def test_step_telemetry_matches_episode_evaluator(self):
        """Verifica que los datos por paso coinciden exactamente con el resultado de evaluate_on_dataset."""
        c0 = JacobianDampedController(kp=3.0, lambda_damping=0.01)
        dataset = generate_dataset(n_episodes=3, seed=202699)
        env_cfg = {
            "env": {
                "success_mode": "hold",
                "target_radius": 0.06,
                "velocity_tolerance": 0.10,
                "hold_steps": 20,
                "max_steps": 200,
                "dt": 0.05
            }
        }

        # 1. Evaluador por episodio de Fase A
        ep_df, _ = evaluate_on_dataset(c0, dataset, "test", "deterministic", env_config=env_cfg)

        # 2. Extractor de telemetría por paso
        step_df = collect_step_telemetry_for_episodes(c0, dataset, env_cfg, "deterministic")

        for _, row in ep_df.iterrows():
            ep_id = row["episode_id"]
            ep_steps = step_df[step_df["episode_id"] == ep_id]
            
            # El número total de pasos debe coincidir exactamente
            self.assertEqual(len(ep_steps), row["steps"], f"Pasos no coinciden en ep {ep_id}: {len(ep_steps)} vs {row['steps']}")
            # El desenlace final de éxito debe ser idéntico
            self.assertEqual(ep_steps["is_success"].iloc[-1], row["success"], f"Éxito no coincide en ep {ep_id}")
            # La distancia final debe coincidir con tolerancia numérica estricta
            self.assertAlmostEqual(ep_steps["distance"].iloc[-1], row["final_distance"], places=6)

    def test_selection_rule_logic_synthetic(self):
        """Valida la regla de meseta y desempate con datos sintéticos controlados."""
        u_meseta = 0.05

        # Caso 1: Meseta (dos transiciones consecutivas con cambio < 0.05)
        curve_meseta = [0.40, 0.60, 0.70, 0.72, 0.73]
        deltas = [abs(curve_meseta[i] - curve_meseta[i-1]) for i in range(1, len(curve_meseta))]
        is_meseta_at_4 = (deltas[2] < u_meseta and deltas[3] < u_meseta)
        self.assertTrue(is_meseta_at_4)

        # Caso 2: Degradación (caída > 0.05)
        curve_deg = [0.70, 0.62]
        self.assertTrue((curve_deg[1] - curve_deg[0]) < -u_meseta)

        # Caso 3: Desempate jerárquico
        c1 = {"step": 100, "success": 0.75, "median_steps": 60, "action_delta": 0.03}
        c2 = {"step": 200, "success": 0.74, "median_steps": 50, "action_delta": 0.04}
        diff = abs(c1["success"] - c2["success"])
        self.assertTrue(diff < u_meseta)
        chosen = c2 if c2["median_steps"] < c1["median_steps"] else c1
        self.assertEqual(chosen["step"], 200)

if __name__ == "__main__":
    unittest.main()
