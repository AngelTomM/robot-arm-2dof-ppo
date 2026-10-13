"""
Generación y Carga de Conjuntos de Datos Reproducibles para Validación y Evaluación Final (Sección 8).
"""
import os
import sys
import math
import json
import numpy as np
import pandas as pd

def generate_dataset(n_episodes: int, seed: int, l1: float = 1.0, l2: float = 1.0,
                     r_min: float = 0.15, r_max: float = 1.90) -> list:
    """
    Genera un conjunto de episodios reproducibles con distribución documentada:
    - theta1, theta2 ~ U(-pi, pi)
    - target_r ~ U(r_min, r_max) (distribución uniforme en radio lineal alcanzable)
    - target_phi ~ U(-pi, pi)
    """
    rng = np.random.RandomState(seed)
    episodes = []

    for ep_id in range(n_episodes):
        th1 = float(rng.uniform(-math.pi, math.pi))
        th2 = float(rng.uniform(-math.pi, math.pi))

        r = float(rng.uniform(r_min, r_max))
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

def save_and_get_datasets(data_dir: str = "data/datasets"):
    os.makedirs(data_dir, exist_ok=True)

    val_json_path = os.path.join(data_dir, "validation_set_200.json")
    val_csv_path = os.path.join(data_dir, "validation_set_200.csv")
    final_json_path = os.path.join(data_dir, "final_eval_set_300.json")
    final_csv_path = os.path.join(data_dir, "final_eval_set_300.csv")

    # 1. Validation Set (200 episodios, seed=202601)
    if not os.path.exists(val_json_path):
        val_data = generate_dataset(n_episodes=200, seed=202601)
        with open(val_json_path, "w", encoding="utf-8") as f:
            json.dump(val_data, f, indent=2)
        pd.DataFrame(val_data).to_csv(val_csv_path, index=False)
        print(f">>> [DATASET] Conjunto de Validación generado: {val_json_path} ({len(val_data)} episodios)")
    else:
        with open(val_json_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)

    # 2. Final Eval Set (300 episodios, seed=202602)
    if not os.path.exists(final_json_path):
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
    val_set, final_set = save_and_get_datasets()
    print(f"Validación: {len(val_set)} episodios. Final: {len(final_set)} episodios.")
