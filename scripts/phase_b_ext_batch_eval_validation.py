"""
Orquestador de Evaluación en Lote sobre Validación B2 para Fase B (Extensión):
- Itera sobre todos los checkpoints (paso <= 655.360) de todas las configuraciones.
- Soporta ejecución multi-proceso (paralela) para optimizar el tiempo de cómputo en CPUs multi-core.
- Guarda CSV detallado (por semilla y checkpoint) y CSV agregado (media +- std inter-semilla).
"""
import os
import sys
from pathlib import Path
import json
import re
import argparse
import yaml
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_b_ext_eval_checkpoints import (
    evaluate_single_checkpoint,
    evaluate_c0_controller,
    extract_step_from_dirname
)

import torch
import multiprocessing

def _worker_init():
    torch.set_num_threads(1)

def _eval_ckpt_worker(task):
    torch.set_num_threads(1)
    ckpt_dir_str, val_dataset_path_str, env_cfg, agent_id, seed, step = task
    with open(val_dataset_path_str, "r", encoding="utf-8") as f:
        val_dataset = json.load(f)
    summ, _ = evaluate_single_checkpoint(Path(ckpt_dir_str), val_dataset, env_cfg)
    summ["experiment"] = agent_id
    summ["step"] = step
    summ["seed"] = seed
    return summ

def run_batch_eval_validation(max_step_filter: int = 655360,
                              output_dir: str = "reports/data/phase_b/extension",
                              dataset_path: str = "data/datasets/phase_b_ext/validation_b2_500.json",
                              agents_to_eval: list = None,
                              num_workers: int = 8):
    val_dataset_file = REPO_ROOT / dataset_path
    if not val_dataset_file.exists():
        raise FileNotFoundError(f"Conjunto de validación no encontrado: {val_dataset_file}")

    with open(val_dataset_file, "r", encoding="utf-8") as f:
        val_dataset = json.load(f)

    out_path = REPO_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Definir rutas de agentes (reutilizados vs extensión)
    all_agent_configs = [
        ("X1a", REPO_ROOT / "configs/phase_b/extension/X1a.yaml", REPO_ROOT / "experiments/phase_b/extension/X1a", [42, 100, 2026]),
        ("X1b", REPO_ROOT / "configs/phase_b/extension/X1b.yaml", REPO_ROOT / "experiments/phase_b/extension/X1b", [42]),
        ("X1c", REPO_ROOT / "configs/phase_b/extension/X1c.yaml", REPO_ROOT / "experiments/phase_b/extension/X1c", [42]),
        ("X1s", REPO_ROOT / "configs/phase_b/extension/X1s.yaml", REPO_ROOT / "experiments/phase_b/extension/X1s", [42]),
        ("X2", REPO_ROOT / "configs/phase_b/extension/X2.yaml", REPO_ROOT / "experiments/phase_b/extension/X2", [42, 100, 2026]),
        ("X2b", REPO_ROOT / "configs/phase_b/extension/X2b.yaml", REPO_ROOT / "experiments/phase_b/extension/X2b", [42]),
        ("X3", REPO_ROOT / "configs/phase_b/extension/X3.yaml", REPO_ROOT / "experiments/phase_b/X3", [42, 100, 2026]),
        ("X4", REPO_ROOT / "configs/phase_b/extension/X4.yaml", REPO_ROOT / "experiments/phase_b/extension/X4", [42, 100, 2026]),
        ("X5_near", REPO_ROOT / "configs/phase_b/extension/X5_near.yaml", REPO_ROOT / "experiments/phase_b/X5_near", [42, 100, 2026]),
        ("X5", REPO_ROOT / "configs/phase_b/extension/X5.yaml", REPO_ROOT / "experiments/phase_b/extension/X5", [42, 100, 2026]),
        ("X6", REPO_ROOT / "configs/phase_b/extension/X6.yaml", REPO_ROOT / "experiments/phase_b/extension/X6", [42, 100, 2026]),
    ]

    if agents_to_eval:
        all_agent_configs = [item for item in all_agent_configs if item[0] in agents_to_eval]

    all_records = []

    # 2. Evaluar Controladores C0 (ambas versiones)
    ref_cfg_path = REPO_ROOT / "configs/phase_b/extension/X3.yaml"
    with open(ref_cfg_path, "r", encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f)

    print("\n>>> [EVAL VALIDATION B2] Evaluando C0 Consistente (max_speed=1.5)...")
    c0_cons_summary, _ = evaluate_c0_controller(val_dataset, env_cfg, max_speed=1.5, label="consistent_1_5")
    all_records.append(c0_cons_summary)

    print(">>> [EVAL VALIDATION B2] Evaluando C0 Histórico (max_speed=2.0)...")
    c0_hist_summary, _ = evaluate_c0_controller(val_dataset, env_cfg, max_speed=2.0, label="historical_2_0")
    all_records.append(c0_hist_summary)

    # 3. Preparar lista de tareas de checkpoints
    tasks = []
    for agent_id, cfg_path, exp_base_dir, seeds in all_agent_configs:
        if not exp_base_dir.exists():
            print(f"AVISO: Directorio del agente {agent_id} no existe ({exp_base_dir}), omitiendo...")
            continue

        with open(cfg_path, "r", encoding="utf-8") as f:
            agent_env_cfg = yaml.safe_load(f)

        for s in seeds:
            seed_dir = exp_base_dir / f"seed_{s}"
            ckpts_dir = seed_dir / "checkpoints"
            if not ckpts_dir.exists():
                print(f"AVISO: Checkpoints para {agent_id} seed {s} no existen en {ckpts_dir}, omitiendo...")
                continue

            ckpt_dirs = [p for p in ckpts_dir.iterdir() if p.is_dir()]
            ckpt_dirs = sorted(ckpt_dirs, key=lambda p: extract_step_from_dirname(p.name))

            for c_dir in ckpt_dirs:
                step = extract_step_from_dirname(c_dir.name)
                # Filtrar estrictamente presupuesto <= max_step_filter (655 360)
                if step > max_step_filter or step <= 0:
                    continue

                task = (str(c_dir), str(val_dataset_file), agent_env_cfg, agent_id, s, step)
                tasks.append(task)

    print(f"\n>>> [EVAL VALIDATION B2] Total checkpoints a evaluar: {len(tasks)} (Workers={num_workers})")

    # 4. Ejecución en paralelo o secuencial
    completed_count = 0
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init) as executor:
            future_to_task = {executor.submit(_eval_ckpt_worker, t): t for t in tasks}
            for fut in as_completed(future_to_task):
                summ = fut.result()
                all_records.append(summ)
                completed_count += 1
                if completed_count % 10 == 0 or completed_count == len(tasks):
                    print(f"  [{completed_count}/{len(tasks)}] Evaluado {summ['experiment']} s{summ['seed']} p{summ['step']:,} -> Éxito: {summ['strict_success_rate']*100:.1f}%", flush=True)
    else:
        for t in tasks:
            summ = _eval_ckpt_worker(t)
            all_records.append(summ)
            completed_count += 1
            print(f"  [{completed_count}/{len(tasks)}] Evaluado {summ['experiment']} s{summ['seed']} p{summ['step']:,} -> Éxito: {summ['strict_success_rate']*100:.1f}%", flush=True)

    # 5. Guardar CSV detallado ordenado determinísticamente
    df = pd.DataFrame(all_records)
    # Ordenar: C0 primero, luego por experiment, seed, step
    df["sort_step"] = df["step"].apply(lambda x: 99999999 if x == -1 else x)
    df = df.sort_values(by=["experiment", "seed", "sort_step"]).drop(columns=["sort_step"])

    csv_detail_path = out_path / "checkpoints_eval_validation.csv"
    df.to_csv(csv_detail_path, index=False)
    print(f"\n>>> [OK] CSV Detallado guardado en: {csv_detail_path} ({len(df)} registros)")

    # 6. Generar CSV Agregado (media +- std inter-semilla por checkpoint)
    ppo_df = df[df["step"] > 0].copy()
    
    def calc_std(x):
        return float(x.std(ddof=1)) if len(x) > 1 else np.nan

    agg_df = ppo_df.groupby(["experiment", "step"]).agg(
        n_seeds=("seed", "count"),
        mean_strict_success=("strict_success_rate", "mean"),
        std_strict_success=("strict_success_rate", calc_std),
        mean_steps_to_success_median=("steps_to_success_median", "mean"),
        std_steps_to_success_median=("steps_to_success_median", calc_std),
        mean_final_distance=("mean_final_distance", "mean"),
        mean_action_delta=("mean_action_delta", "mean"),
        mean_truncation_rate=("truncation_rate", "mean"),
        mean_ballistic_rate=("ballistic_rate", "mean"),
        mean_oscillatory_rate=("oscillatory_rate", "mean")
    ).reset_index()

    csv_agg_path = out_path / "checkpoints_eval_validation_aggregated.csv"
    agg_df.to_csv(csv_agg_path, index=False)
    print(f">>> [OK] CSV Agregado guardado en: {csv_agg_path} ({len(agg_df)} puntos agregados)")
    return df, agg_df

if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Evaluación masiva en Validación B2")
    parser.add_argument("--max_step", type=int, default=655360)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="reports/data/phase_b/extension")
    parser.add_argument("--agents", nargs="*", default=None, help="Lista opcional de agentes a evaluar")
    args = parser.parse_args()
    run_batch_eval_validation(max_step_filter=args.max_step, output_dir=args.output_dir, num_workers=args.workers, agents_to_eval=args.agents)
