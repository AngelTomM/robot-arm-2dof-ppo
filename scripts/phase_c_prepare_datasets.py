"""
Generación, Sellado Criptográfico y Selección Geométrica de Datasets de Fase C:
- Validación C: 500 episodios, seed=202631
- Final C: 500 episodios, seed=202641 (sellado criptográficamente, NO SE ABRE)
- Verificación exhaustiva de NO solapamiento entre los 6 conjuntos del repositorio.
- Identificación de los 4 episodios geométricos a priori (Q25, Q50, Q75, Edge).
"""
import os
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.datasets import generate_dataset, save_and_get_datasets

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    h.update(filepath.read_bytes())
    return h.hexdigest()

def prepare_phase_c_datasets():
    c_data_dir = REPO_ROOT / "data" / "datasets" / "phase_c"
    c_data_dir.mkdir(parents=True, exist_ok=True)
    
    reports_data_dir = REPO_ROOT / "reports" / "data" / "phase_c"
    reports_data_dir.mkdir(parents=True, exist_ok=True)
    
    val_c_path = c_data_dir / "validation_c_500.json"
    val_c_csv = c_data_dir / "validation_c_500.csv"
    final_c_path = c_data_dir / "final_c_500.json"
    final_c_csv = c_data_dir / "final_c_500.csv"
    
    # 1. Generar Validación C (500 episodios, seed=202631)
    if not val_c_path.exists():
        print(">>> [DATASET] Generando Validación C (500 episodios, seed=202631)...")
        val_c_data = generate_dataset(n_episodes=500, seed=202631)
        with open(val_c_path, "w", encoding="utf-8") as f:
            json.dump(val_c_data, f, indent=2)
        pd.DataFrame(val_c_data).to_csv(val_c_csv, index=False)
    else:
        with open(val_c_path, "r", encoding="utf-8") as f:
            val_c_data = json.load(f)
            
    # 2. Generar Final C (500 episodios, seed=202641)
    if not final_c_path.exists():
        print(">>> [DATASET] Generando Final C (500 episodios, seed=202641)...")
        final_c_data = generate_dataset(n_episodes=500, seed=202641)
        with open(final_c_path, "w", encoding="utf-8") as f:
            json.dump(final_c_data, f, indent=2)
        pd.DataFrame(final_c_data).to_csv(final_c_csv, index=False)
    else:
        with open(final_c_path, "r", encoding="utf-8") as f:
            final_c_data = json.load(f)
            
    # 3. Registrar Hashes SHA-256
    val_c_hash = compute_sha256(val_c_path)
    final_c_hash = compute_sha256(final_c_path)
    
    hashes_meta = {
        "validation_c_500_json": {
            "path": "data/datasets/phase_c/validation_c_500.json",
            "seed": 202631,
            "n_episodes": 500,
            "sha256": val_c_hash
        },
        "final_c_500_json": {
            "path": "data/datasets/phase_c/final_c_500.json",
            "seed": 202641,
            "n_episodes": 500,
            "sha256": final_c_hash,
            "sealed": True,
            "status": "SEALED_NOT_EVALUATED"
        }
    }
    
    hash_record_path = reports_data_dir / "datasets_hashes.json"
    with open(hash_record_path, "w", encoding="utf-8") as f:
        json.dump(hashes_meta, f, indent=2)
    print(f">>> [HASH] Hashes de Fase C registrados en: {hash_record_path}")
    print(f"    Validation C SHA-256: {val_c_hash}")
    print(f"    Final C SHA-256:      {final_c_hash} (SELLADO)")
    
    # 4. Verificación exhaustiva de NO solapamiento entre los 6 conjuntos
    val_b2_path = REPO_ROOT / "data" / "datasets" / "phase_b_ext" / "validation_b2_500.json"
    final_b2_path = REPO_ROOT / "data" / "datasets" / "phase_b_ext" / "final_b2_500.json"
    val_old_data, final_old_data = save_and_get_datasets()
    
    with open(val_b2_path, "r", encoding="utf-8") as f: val_b2_data = json.load(f)
    with open(final_b2_path, "r", encoding="utf-8") as f: final_b2_data = json.load(f)
    
    all_datasets = {
        "val_old (200)": val_old_data,
        "final_old (300)": final_old_data,
        "val_b2 (500)": val_b2_data,
        "final_b2 (500)": final_b2_data,
        "val_c (500)": val_c_data,
        "final_c (500)": final_c_data
    }
    
    coords = {}
    for name, data in all_datasets.items():
        coords[name] = set(
            (round(ep["target_x"], 5), round(ep["target_y"], 5),
             round(ep["theta1_init"], 5), round(ep["theta2_init"], 5))
            for ep in data
        )
        
    names = list(coords.keys())
    print("\n>>> [VERIFICACIÓN] Matriz de solapamiento entre los 6 datasets:")
    overlap_detected = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = coords[names[i]] & coords[names[j]]
            if len(inter) > 0:
                print(f"    [ERROR] Solapamiento entre {names[i]} y {names[j]}: {len(inter)} episodios.")
                overlap_detected = True
            else:
                print(f"    [OK] {names[i]} vs {names[j]}: 0 solapamientos.")
                
    if overlap_detected:
        raise ValueError("Se detectó solapamiento entre conjuntos de datos!")
    print(">>> [OK] Los 6 conjuntos son estrictamente disjuntos.")
    
    # 5. Selección de 4 Episodios Geométricos a Priori (sobre Validación C)
    radii = np.array([ep["target_r"] for ep in val_c_data])
    q25_val = float(np.percentile(radii, 25))
    q50_val = float(np.percentile(radii, 50))
    q75_val = float(np.percentile(radii, 75))
    max_val = float(np.max(radii))
    
    # Encontrar episodio más cercano a cada valor
    idx_q25 = int(np.argmin(np.abs(radii - q25_val)))
    idx_q50 = int(np.argmin(np.abs(radii - q50_val)))
    idx_q75 = int(np.argmin(np.abs(radii - q75_val)))
    idx_edge = int(np.argmax(radii))
    
    # Asegurar unicidad
    chosen_indices = [idx_q25, idx_q50, idx_q75, idx_edge]
    assert len(set(chosen_indices)) == 4, "Los índices geométricos deben ser únicos"
    
    geometric_selection = {
        "dataset": "validation_c_500.json",
        "episodes": [
            {
                "label": "Geom-Q1",
                "role": "Cuantil 25% radio (campo cercano)",
                "episode_id": val_c_data[idx_q25]["episode_id"],
                "target_r": val_c_data[idx_q25]["target_r"],
                "target_x": val_c_data[idx_q25]["target_x"],
                "target_y": val_c_data[idx_q25]["target_y"],
                "initial_distance": val_c_data[idx_q25]["initial_distance"]
            },
            {
                "label": "Geom-Q2",
                "role": "Cuantil 50% radio (mediana)",
                "episode_id": val_c_data[idx_q50]["episode_id"],
                "target_r": val_c_data[idx_q50]["target_r"],
                "target_x": val_c_data[idx_q50]["target_x"],
                "target_y": val_c_data[idx_q50]["target_y"],
                "initial_distance": val_c_data[idx_q50]["initial_distance"]
            },
            {
                "label": "Geom-Q3",
                "role": "Cuantil 75% radio (campo lejano)",
                "episode_id": val_c_data[idx_q75]["episode_id"],
                "target_r": val_c_data[idx_q75]["target_r"],
                "target_x": val_c_data[idx_q75]["target_x"],
                "target_y": val_c_data[idx_q75]["target_y"],
                "initial_distance": val_c_data[idx_q75]["initial_distance"]
            },
            {
                "label": "Geom-Edge",
                "role": "Caso de borde (alcance máximo)",
                "episode_id": val_c_data[idx_edge]["episode_id"],
                "target_r": val_c_data[idx_edge]["target_r"],
                "target_x": val_c_data[idx_edge]["target_x"],
                "target_y": val_c_data[idx_edge]["target_y"],
                "initial_distance": val_c_data[idx_edge]["initial_distance"]
            }
        ]
    }
    
    geom_file = reports_data_dir / "geometric_episodes_selection.json"
    with open(geom_file, "w", encoding="utf-8") as f:
        json.dump(geometric_selection, f, indent=2)
    print(f">>> [GEOMETRÍA] Episodios canónicos seleccionados y guardados en: {geom_file}")
    for item in geometric_selection["episodes"]:
        print(f"    {item['label']} (ep {item['episode_id']}): r = {item['target_r']:.4f} m, d_init = {item['initial_distance']:.4f} m")

if __name__ == "__main__":
    prepare_phase_c_datasets()
