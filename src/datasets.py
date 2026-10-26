"""
Generación y Carga de Conjuntos de Datos Reproducibles para Validación y Evaluación Final (Sección 8).
"""
import os
import sys
import math
import json
import numpy as np
import pandas as pd

from pathlib import Path
from .config import load_env_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_dataset(n_episodes: int, seed: int, l1: float = None, l2: float = None,
                     r_min: float = None, r_max: float = None,
                     joint2_limit_deg: float = None) -> list:
    """
    Genera un conjunto de episodios reproducibles con distribución cinemática:
    - theta1 ~ U(-pi, pi)
    - theta2 ~ U(-joint2_max, joint2_max) con tope físico anti-colisión
    - target_r ~ U(r_min_geom, r_max)
    - target_phi ~ U(-pi, pi)
    """
    # Heredar de configs/environment.yaml (SSOT)
    env_cfg = load_env_config().get("env", {})
    l1 = float(l1 if l1 is not None else env_cfg.get("l1", 0.10))
    l2 = float(l2 if l2 is not None else env_cfg.get("l2", 0.11))
    joint2_limit_deg = float(joint2_limit_deg if joint2_limit_deg is not None else env_cfg.get("joint2_limit_deg", 171.887))

    rng = np.random.RandomState(seed)
    episodes = []

    th2_max = math.radians(joint2_limit_deg)
    r_min_geom = math.sqrt(l1**2 + l2**2 + 2 * l1 * l2 * math.cos(th2_max))
    effective_r_min = float(r_min if r_min is not None else max(0.04, r_min_geom))
    effective_r_max = float(r_max if r_max is not None else (l1 + l2) * 0.95)

    for ep_id in range(n_episodes):
        th1 = float(rng.uniform(-math.pi, math.pi))
        th2 = float(rng.uniform(-th2_max, th2_max))

        r = float(rng.uniform(effective_r_min, effective_r_max))
        phi = float(rng.uniform(-math.pi, math.pi))
        tgt_x = float(r * math.cos(phi))
        tgt_y = float(r * math.sin(phi))

        # Posición inicial del efector final
        x1 = l1 * math.cos(th1)
        y1 = l1 * math.sin(th1)
        ee_x = x1 + l2 * math.cos(th1 + th2)
        ee_y = y1 + l2 * math.sin(th1 + th2)
        init_dist = float(math.sqrt((tgt_x - ee_x)**2 + (tgt_y - ee_y)**2))

        episodes.append({
            "episode_id": ep_id,
            "seed": seed,
            "theta1_init": th1,
            "theta2_init": th2,
            "target_x": tgt_x,
            "target_y": tgt_y,
            "target_r": r,
            "target_phi": phi,
            "ee_init_x": ee_x,
            "ee_init_y": ee_y,
            "initial_distance": init_dist
        })

    return episodes

def save_and_get_datasets(data_dir: str = "datasets", force_regenerate: bool = False):
    p = Path(data_dir)
    resolved_dir = p if p.is_absolute() else (REPO_ROOT / p)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    val_json_path = str(resolved_dir / "validation_set_200.json")
    val_csv_path = str(resolved_dir / "validation_set_200.csv")
    final_json_path = str(resolved_dir / "final_eval_set_300.json")
    final_csv_path = str(resolved_dir / "final_eval_set_300.csv")

    # 1. Validation Set (200 episodios, seed=202601)
    if force_regenerate or not os.path.exists(val_json_path):
        val_data = generate_dataset(n_episodes=200, seed=202601)
        with open(val_json_path, "w", encoding="utf-8") as f:
            json.dump(val_data, f, indent=2)
        pd.DataFrame(val_data).to_csv(val_csv_path, index=False)
        print(f">>> [DATASET] Conjunto de Validación generado: {val_json_path} ({len(val_data)} episodios)")
    else:
        with open(val_json_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)

    # 2. Final Eval Set (300 episodios, seed=202602)
    if force_regenerate or not os.path.exists(final_json_path):
        final_data = generate_dataset(n_episodes=300, seed=202602)
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2)
        pd.DataFrame(final_data).to_csv(final_csv_path, index=False)
        print(f">>> [DATASET] Conjunto de Evaluación Final generado: {final_json_path} ({len(final_data)} episodios)")
    else:
        with open(final_json_path, "r", encoding="utf-8") as f:
            final_data = json.load(f)

    return val_data, final_data

if __name__ == "__main__":
    force = "--regenerate" in sys.argv or True
    val_set, final_set = save_and_get_datasets(force_regenerate=force)
    print(f"Validación: {len(val_set)} episodios. Final: {len(final_set)} episodios.")
