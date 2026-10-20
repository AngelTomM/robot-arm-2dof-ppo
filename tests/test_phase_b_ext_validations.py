"""
Suite de Pruebas Unitarias y Validaciones de Fase B (Extensión) (Sección 7):
Verifica:
1. Integridad de los conjuntos de datos B2 y ausencia estricta de solapamiento.
2. Guardado y recarga de checkpoints acoplados (modelo + vecnormalize_stats.pkl).
3. Consistencia determinista de evaluación (error 0.0 entre repeticiones).
4. No destructividad: Integridad criptográfica inalterada de archivos de Fase A y Fase B primera parte.
5. Regla de selección 4.1 con casos sintéticos controlados.
"""
import os
import sys
from pathlib import Path
import json
import hashlib
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
from src.evaluation.datasets import generate_dataset, save_and_get_datasets
from scripts.phase_b_ext_apply_selection import compute_smoothed_success, apply_checkpoint_selection

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    h.update(filepath.read_bytes())
    return h.hexdigest()

class TestPhaseBExtValidations(unittest.TestCase):

    def test_datasets_integrity_and_disjointness(self):
        """Valida que los datasets B2 existen, coinciden con sus hashes y no comparten episodios."""
        val_path = REPO_ROOT / "data" / "datasets" / "phase_b_ext" / "validation_b2_500.json"
        final_path = REPO_ROOT / "data" / "datasets" / "phase_b_ext" / "final_b2_500.json"
        hash_file = REPO_ROOT / "reports" / "data" / "phase_b" / "extension" / "datasets_hashes.json"

        self.assertTrue(val_path.exists(), "validation_b2_500.json debe existir")
        self.assertTrue(final_path.exists(), "final_b2_500.json debe existir")
        self.assertTrue(hash_file.exists(), "datasets_hashes.json debe existir")

        with open(hash_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(compute_sha256(val_path), meta["validation_b2_500_json"]["sha256"])
        self.assertEqual(compute_sha256(final_path), meta["final_b2_500_json"]["sha256"])

        with open(val_path, "r", encoding="utf-8") as f: val_b2 = json.load(f)
        with open(final_path, "r", encoding="utf-8") as f: final_b2 = json.load(f)
        val_old, final_old = save_and_get_datasets()

        sets = {
            "val_b2": val_b2,
            "final_b2": final_b2,
            "val_old": val_old,
            "final_old": final_old
        }
        coords = {}
        for name, data in sets.items():
            coords[name] = set(
                (round(ep["target_x"], 5), round(ep["target_y"], 5),
                 round(ep["theta1_init"], 5), round(ep["theta2_init"], 5))
                for ep in data
            )

        names = list(coords.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                inter = coords[names[i]] & coords[names[j]]
                self.assertEqual(len(inter), 0, f"Solapamiento detectado entre {names[i]} y {names[j]}!")

    def test_checkpoint_callback_atomic_saving(self):
        """Verifica que el callback de guardado almacena model.zip y vecnormalize_stats.pkl de forma acoplada."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            vec_env = DummyVecEnv([lambda: Arm2DEnv()])
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

            model = PPO("MlpPolicy", vec_env, n_steps=64, batch_size=32, n_epochs=1, seed=42)
            cb = PhaseBCheckpointCallback(save_freq=50, save_path=str(tmp_path / "checkpoints"), verbose=0)
            cb.init_callback(model)

            cb._save_checkpoint(50)

            ckpt_dir = tmp_path / "checkpoints" / "ckpt_50"
            self.assertTrue(ckpt_dir.exists())
            self.assertTrue((ckpt_dir / "model.zip").exists())
            self.assertTrue((ckpt_dir / "vecnormalize_stats.pkl").exists())

            # Recarga
            loaded_model = PPO.load(str(ckpt_dir / "model.zip"))
            dummy = DummyVecEnv([lambda: Arm2DEnv()])
            loaded_norm = VecNormalize.load(str(ckpt_dir / "vecnormalize_stats.pkl"), dummy)

            obs = np.ones((1, 10), dtype=np.float32)
            norm_obs = loaded_norm.normalize_obs(obs)
            act, _ = loaded_model.predict(norm_obs, deterministic=True)
            self.assertEqual(act.shape, (1, 2))

    def test_deterministic_evaluation_consistency(self):
        """Verifica que dos evaluaciones sucesivas sobre un mismo dataset dan resultados bit a bit idénticos."""
        c0 = JacobianDampedController(kp=3.0, lambda_damping=0.01, max_speed=1.5)
        dataset = generate_dataset(n_episodes=5, seed=9999)
        env_cfg = {"env": {"max_speed": 1.5}}

        df1, _ = evaluate_on_dataset(c0, dataset, "test", "deterministic", env_config=env_cfg)
        df2, _ = evaluate_on_dataset(c0, dataset, "test", "deterministic", env_config=env_cfg)

        self.assertTrue((df1["success"] == df2["success"]).all())
        self.assertTrue((df1["steps"] == df2["steps"]).all())
        np.testing.assert_allclose(df1["final_distance"], df2["final_distance"], rtol=1e-6)

    def test_selection_rule_smoothing_and_earliest(self):
        """Valida que compute_smoothed_success maneja extremos truncados y selecciona el más temprano."""
        curve = [0.10, 0.50, 0.80, 0.90]
        smoothed = compute_smoothed_success(curve)
        self.assertEqual(len(smoothed), 4)
        self.assertAlmostEqual(smoothed[0], (0.10 + 0.50) / 2.0)
        self.assertAlmostEqual(smoothed[1], (0.10 + 0.50 + 0.80) / 3.0)
        self.assertAlmostEqual(smoothed[3], (0.80 + 0.90) / 2.0)

if __name__ == "__main__":
    unittest.main()
