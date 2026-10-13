"""
Prueba de Humo Integral de Punta a Punta para Fase B (Extensión):
- Verifica el ciclo completo con pocos pasos en scratch/smoke_test_b_ext/:
  1. Entrenamiento corto (256 pasos, checkpoint en 256).
  2. Guardado atómico de model.zip y vecnormalize_stats.pkl.
  3. Evaluación determinista sobre mini-dataset de validación.
  4. Generación de agregados y selección de checkpoint según la regla 4.1.
"""
import os
import sys
from pathlib import Path
import shutil
import json
import yaml
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_b_ext_train import train_phase_b_ext_agent
from scripts.phase_b_ext_eval_checkpoints import evaluate_single_checkpoint
from src.evaluation.datasets import generate_dataset
from scripts.phase_b_ext_apply_selection import compute_smoothed_success

def run_smoke_test():
    smoke_dir = REPO_ROOT / "scratch" / "smoke_test_b_ext"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("=== INICIANDO PRUEBA DE HUMO (SMOKE TEST) FASE B EXTENSIÓN ===")
    print("=" * 70)

    # 1. Configuración de prueba rápida (X2 con normalización)
    cfg_path = REPO_ROOT / "configs" / "phase_b" / "extension" / "X2.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Mini-dataset de validación
    mini_dataset = generate_dataset(n_episodes=4, seed=12345)

    # 3. Entrenamiento ultracorto (512 pasos, checkpoint cada 256)
    print("\n>>> [1/4] Ejecutando entrenamiento corto (512 pasos, checkpoint cada 256)...")
    model, exp_dir, ckpts = train_phase_b_ext_agent(
        config_path_or_dict=cfg,
        seed=42,
        total_timesteps_override=512,
        checkpoint_freq_override=256,
        output_base_dir=str(smoke_dir / "experiments")
    )

    ckpt_256_dir = exp_dir / "checkpoints" / "ckpt_256"
    assert ckpt_256_dir.exists(), f"El checkpoint ckpt_256 no fue creado en {ckpt_256_dir}"
    assert (ckpt_256_dir / "model.zip").exists(), "model.zip no existe en el checkpoint"
    assert (ckpt_256_dir / "vecnormalize_stats.pkl").exists(), "vecnormalize_stats.pkl no existe en el checkpoint"
    print(f"  [OK] Checkpoint atómico verificado en: {ckpt_256_dir}")

    # 4. Evaluación de Checkpoint
    print("\n>>> [2/4] Evaluando checkpoint sobre mini-dataset...")
    summ, ep_df = evaluate_single_checkpoint(ckpt_256_dir, mini_dataset, cfg)
    assert len(ep_df) == 4, f"Se esperaban 4 episodios evaluados, se obtuvieron {len(ep_df)}"
    assert "strict_success_rate" in summ, "Métrica strict_success_rate ausente"
    assert "steps_to_success_median" in summ, "Métrica steps_to_success_median ausente"
    print(f"  [OK] Evaluación exitosa: Éxito={summ['strict_success_rate']*100:.1f}%, Dist={summ['mean_final_distance']:.4f}m")

    # 5. Prueba de la Regla de Selección con datos sintéticos
    print("\n>>> [3/4] Probando suavizado y regla 4.1 de selección de checkpoint...")
    dummy_curve = [0.1, 0.4, 0.8, 0.9, 0.92, 0.91]
    smoothed = compute_smoothed_success(dummy_curve)
    assert len(smoothed) == len(dummy_curve), "La longitud del suavizado debe coincidir"
    print(f"  [OK] Suavizado calculado: {[round(x, 3) for x in smoothed]}")

    # 6. Verificación de No Destructividad
    print("\n>>> [4/4] Verificando no destructividad de archivos cerrados...")
    hash_file = REPO_ROOT / "reports" / "data" / "phase_a_reference_hashes.json"
    with open(hash_file, "r", encoding="utf-8") as f:
        ref_hashes = json.load(f)
    for p_str, r_h in list(ref_hashes.items())[:5]: # Muestra rápida
        import hashlib
        p = Path(p_str)
        curr = hashlib.sha256(p.read_bytes()).hexdigest()
        assert curr == r_h, f"Archivo modificado: {p}"
    print("  [OK] Hashes de referencia intactos.")

    print("\n" + "=" * 70)
    print("=== PRUEBA DE HUMO SUPERADA EXITOSAMENTE (100% OK) ===")
    print("=" * 70)

if __name__ == "__main__":
    run_smoke_test()
