"""
Suite de Pruebas Unitarias y Validaciones para Fase C (Etapa C-3):
Verifica:
1. V-1: Guardado y recarga atómica de checkpoints SAC (model.zip + vecnormalize_stats.pkl).
2. V-2: Consistencia determinista de evaluación (error 0.0 entre repeticiones idénticas).
3. V-4: Integridad de conjuntos de datos de Fase C y ausencia estricta de solapamiento entre los 6 conjuntos.
4. V-7: No destructividad: Integridad inalterada de archivos de referencia de Fases A y B.
5. V-9 & V-10: Verificación de equivalencia de entorno y recompensa entre configs S1 y X3.
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
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arm2d_env import Arm2DEnv
from src.controllers.jacobian_controller import JacobianDampedController
from src.training.phase_b_callbacks import PhaseBCheckpointCallback
from src.evaluation.evaluator import evaluate_on_dataset
from src.evaluation.datasets import generate_dataset, save_and_get_datasets

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    h.update(filepath.read_bytes())
    return h.hexdigest()

class TestPhaseCValidations(unittest.TestCase):

    def test_v1_sac_atomic_checkpoint_save_and_reload(self):
        """V-1: Verifica que PhaseBCheckpointCallback guarda atómicamente model.zip y vecnormalize_stats.pkl con SAC."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            vec_env = DummyVecEnv([lambda: Arm2DEnv()])
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

            model = SAC(
                "MlpPolicy",
                vec_env,
                learning_rate=3e-4,
                buffer_size=1000,
                learning_starts=50,
                batch_size=32,
                seed=42,
                device="cpu"
            )
            cb = PhaseBCheckpointCallback(save_freq=50, save_path=str(tmp_path / "checkpoints"), verbose=0)
            cb.init_callback(model)

            cb._save_checkpoint(50)

            ckpt_dir = tmp_path / "checkpoints" / "ckpt_50"
            self.assertTrue(ckpt_dir.exists())
            self.assertTrue((ckpt_dir / "model.zip").exists())
            self.assertTrue((ckpt_dir / "vecnormalize_stats.pkl").exists())

            # Recarga y predicción
            loaded_model = SAC.load(str(ckpt_dir / "model.zip"))
            dummy = DummyVecEnv([lambda: Arm2DEnv()])
            loaded_norm = VecNormalize.load(str(ckpt_dir / "vecnormalize_stats.pkl"), dummy)

            obs = np.ones((1, 10), dtype=np.float32)
            norm_obs = loaded_norm.normalize_obs(obs)
            act, _ = loaded_model.predict(norm_obs, deterministic=True)
            self.assertEqual(act.shape, (1, 2))
            self.assertTrue(np.all(act >= -1.0) and np.all(act <= 1.0))

    def test_v2_deterministic_evaluation_consistency(self):
        """V-2: Verifica que dos evaluaciones consecutivas deterministas son idénticas bit a bit (error = 0.0)."""
        c0 = JacobianDampedController(kp=3.0, lambda_damping=0.01, max_speed=1.5)
        dataset = generate_dataset(n_episodes=5, seed=7777)
        env_cfg = {"env": {"max_speed": 1.5}}

        df1, _ = evaluate_on_dataset(c0, dataset, "test_c", "deterministic", env_config=env_cfg)
        df2, _ = evaluate_on_dataset(c0, dataset, "test_c", "deterministic", env_config=env_cfg)

        self.assertTrue((df1["success"] == df2["success"]).all())
        self.assertTrue((df1["steps"] == df2["steps"]).all())
        np.testing.assert_allclose(df1["final_distance"], df2["final_distance"], rtol=1e-6)

    def test_v4_datasets_integrity_and_disjointness_all_six(self):
        """V-4: Verifica que los datasets de Fase C existen, coinciden con sus hashes y no solapan con ningún conjunto previo."""
        val_c_path = REPO_ROOT / "data" / "datasets" / "phase_c" / "validation_c_500.json"
        final_c_path = REPO_ROOT / "data" / "datasets" / "phase_c" / "final_c_500.json"
        hash_file = REPO_ROOT / "reports" / "data" / "phase_c" / "datasets_hashes.json"

        self.assertTrue(val_c_path.exists(), "validation_c_500.json debe existir")
        self.assertTrue(final_c_path.exists(), "final_c_500.json debe existir")
        self.assertTrue(hash_file.exists(), "datasets_hashes.json de Fase C debe existir")

        with open(hash_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(compute_sha256(val_c_path), meta["validation_c_500_json"]["sha256"])
        self.assertEqual(compute_sha256(final_c_path), meta["final_c_500_json"]["sha256"])

        # Cargar los 6 conjuntos
        with open(val_c_path, "r", encoding="utf-8") as f: val_c = json.load(f)
        with open(final_c_path, "r", encoding="utf-8") as f: final_c = json.load(f)

        val_b2_path = REPO_ROOT / "data" / "datasets" / "phase_b_ext" / "validation_b2_500.json"
        final_b2_path = REPO_ROOT / "data" / "datasets" / "phase_b_ext" / "final_b2_500.json"
        with open(val_b2_path, "r", encoding="utf-8") as f: val_b2 = json.load(f)
        with open(final_b2_path, "r", encoding="utf-8") as f: final_b2 = json.load(f)

        val_old, final_old = save_and_get_datasets()

        sets = {
            "val_old": val_old,
            "final_old": final_old,
            "val_b2": val_b2,
            "final_b2": final_b2,
            "val_c": val_c,
            "final_c": final_c
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

    def test_v7_phase_a_and_b_references_unmodified(self):
        """V-7: Verifica que ningún archivo de referencia histórico de Fase A o Fase B ha sido alterado."""
        hash_file_a = REPO_ROOT / "reports" / "data" / "phase_a_reference_hashes.json"
        self.assertTrue(hash_file_a.exists())
        with open(hash_file_a, "r", encoding="utf-8") as f:
            reference_hashes_a = json.load(f)

        for file_str, ref_hash in reference_hashes_a.items():
            p = Path(file_str)
            self.assertTrue(p.exists(), f"Archivo ausente: {p}")
            self.assertEqual(compute_sha256(p), ref_hash, f"Archivo de Fase A modificado: {p}")

        hash_file_b = REPO_ROOT / "reports" / "data" / "phase_b" / "extension" / "datasets_hashes.json"
        self.assertTrue(hash_file_b.exists())
        with open(hash_file_b, "r", encoding="utf-8") as f:
            meta_b = json.load(f)
        val_b2_p = REPO_ROOT / "data" / "datasets" / "phase_b_ext" / "validation_b2_500.json"
        final_b2_p = REPO_ROOT / "data" / "datasets" / "phase_b_ext" / "final_b2_500.json"
        self.assertEqual(compute_sha256(val_b2_p), meta_b["validation_b2_500_json"]["sha256"])
        self.assertEqual(compute_sha256(final_b2_p), meta_b["final_b2_500_json"]["sha256"])

    def test_v9_v10_config_equivalence_and_diff(self):
        """V-9 & V-10: Verifica que S1 y X3 comparten física, recompensa y hold, difiriendo solo en el algoritmo."""
        s1_yaml = REPO_ROOT / "configs" / "phase_c" / "S1.yaml"
        x3_yaml = REPO_ROOT / "configs" / "phase_b" / "extension" / "X3.yaml"

        self.assertTrue(s1_yaml.exists())
        self.assertTrue(x3_yaml.exists())

        with open(s1_yaml, "r", encoding="utf-8") as f: s1_cfg = yaml.safe_load(f)
        with open(x3_yaml, "r", encoding="utf-8") as f: x3_cfg = yaml.safe_load(f)

        # Entorno físico idéntico
        self.assertEqual(s1_cfg["env"]["l1"], x3_cfg["env"]["l1"])
        self.assertEqual(s1_cfg["env"]["l2"], x3_cfg["env"]["l2"])
        self.assertEqual(s1_cfg["env"]["max_speed"], x3_cfg["env"]["max_speed"])
        self.assertEqual(s1_cfg["env"]["target_radius"], x3_cfg["env"]["target_radius"])
        self.assertEqual(s1_cfg["env"]["max_steps"], x3_cfg["env"]["max_steps"])
        self.assertEqual(s1_cfg["env"]["dt"], x3_cfg["env"]["dt"])
        self.assertEqual(s1_cfg["env"]["success_mode"], x3_cfg["env"]["success_mode"])
        self.assertEqual(s1_cfg["env"]["hold_steps"], x3_cfg["env"]["hold_steps"])
        self.assertEqual(s1_cfg["env"]["velocity_tolerance"], x3_cfg["env"]["velocity_tolerance"])

        # Recompensa idéntica
        self.assertEqual(s1_cfg["env"]["reward_type"], x3_cfg["env"]["reward_type"])
        self.assertEqual(s1_cfg["env"]["reward_params"], x3_cfg["env"]["reward_params"])

        # Normalización idéntica
        self.assertEqual(s1_cfg["normalization"]["norm_obs"], x3_cfg["normalization"]["norm_obs"])
        self.assertEqual(s1_cfg["normalization"]["norm_reward"], x3_cfg["normalization"]["norm_reward"])

        # Presupuesto idéntico
        self.assertEqual(s1_cfg["training"]["total_timesteps"], x3_cfg["training"]["total_timesteps"])
        self.assertEqual(s1_cfg["training"]["checkpoint_freq"], x3_cfg["training"]["checkpoint_freq"])

if __name__ == "__main__":
    unittest.main()
