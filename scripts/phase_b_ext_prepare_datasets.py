"""
Generación y Verificación de Conjuntos de Datos para Fase B (Extensión):
- Validación B2: 500 episodios, seed=202611
- Final B2: 500 episodios, seed=202622 (sellado criptográficamente)
- Verificación exhaustiva de NO solapamiento con conjuntos previos (200 y 300) y entre sí.
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

def prepare_phase_b_ext_datasets():
    ext_data_dir = REPO_ROOT / "data" / "datasets" / "phase_b_ext"
    ext_data_dir.mkdir(parents=True, exist_ok=True)
    
    reports_data_dir = REPO_ROOT / "reports" / "data" / "phase_b" / "extension"
    reports_data_dir.mkdir(parents=True, exist_ok=True)
    
    val_b2_path = ext_data_dir / "validation_b2_500.json"
    val_b2_csv = ext_data_dir / "validation_b2_500.csv"
    final_b2_path = ext_data_dir / "final_b2_500.json"
    final_b2_csv = ext_data_dir / "final_b2_500.csv"
    
    # 1. Generar Validación B2 (500 episodios, seed=202611)
    if not val_b2_path.exists():
        print(">>> [DATASET] Generando Validación B2 (500 episodios, seed=202611)...")
        val_b2_data = generate_dataset(n_episodes=500, seed=202611)
        with open(val_b2_path, "w", encoding="utf-8") as f:
            json.dump(val_b2_data, f, indent=2)
        pd.DataFrame(val_b2_data).to_csv(val_b2_csv, index=False)
    else:
        with open(val_b2_path, "r", encoding="utf-8") as f:
            val_b2_data = json.load(f)
            
    # 2. Generar Final B2 (500 episodios, seed=202622)
    if not final_b2_path.exists():
        print(">>> [DATASET] Generando Final B2 (500 episodios, seed=202622)...")
        final_b2_data = generate_dataset(n_episodes=500, seed=202622)
        with open(final_b2_path, "w", encoding="utf-8") as f:
            json.dump(final_b2_data, f, indent=2)
        pd.DataFrame(final_b2_data).to_csv(final_b2_csv, index=False)
    else:
        with open(final_b2_path, "r", encoding="utf-8") as f:
            final_b2_data = json.load(f)
            
    # 3. Registrar Hash SHA-256 de Final B2 y sellarlo
    final_b2_hash = compute_sha256(final_b2_path)
    val_b2_hash = compute_sha256(val_b2_path)
    
    hashes_meta = {
        "final_b2_500_json": {
            "path": str(final_b2_path.relative_to(REPO_ROOT)),
            "sha256": final_b2_hash,
            "episodes": len(final_b2_data),
            "seed": 202622,
            "status": "SEALED (No abrir hasta B-12)"
        },
        "validation_b2_500_json": {
            "path": str(val_b2_path.relative_to(REPO_ROOT)),
            "sha256": val_b2_hash,
            "episodes": len(val_b2_data),
            "seed": 202611,
            "status": "ACTIVE (Para toma de decisiones B-11)"
        }
    }
    with open(reports_data_dir / "datasets_hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes_meta, f, indent=2)
    with open(reports_data_dir / "final_b2_500_hash.json", "w", encoding="utf-8") as f:
        json.dump(hashes_meta["final_b2_500_json"], f, indent=2)
        
    print(f">>> [DATASET OK] Final B2 sellado con hash SHA-256: {final_b2_hash}")
    
    # 4. Verificación de NO solapamiento con conjuntos anteriores
    val_old, final_old = save_and_get_datasets()
    all_sets = {
        "val_old_200": val_old,
        "final_old_300": final_old,
        "val_b2_500": val_b2_data,
        "final_b2_500": final_b2_data
    }
    
    coords = {}
    for name, s in all_sets.items():
        coords[name] = set(
            (round(ep["target_x"], 5), round(ep["target_y"], 5),
             round(ep["theta1_init"], 5), round(ep["theta2_init"], 5))
            for ep in s
        )
        
    names = list(coords.keys())
    has_overlap = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = coords[names[i]] & coords[names[j]]
            if len(inter) > 0:
                print(f"ALERTA: Solapamiento detectado entre {names[i]} y {names[j]}: {len(inter)} episodios.")
                has_overlap = True
                
    if has_overlap:
        raise ValueError("Error de aislamiento: se detectó solapamiento entre conjuntos de datos!")
        
    print(">>> [DATASET OK] Verificación de NO solapamiento: 100% de los 1.500 episodios son estrictamente disjuntos.")
    return hashes_meta

if __name__ == "__main__":
    prepare_phase_b_ext_datasets()
