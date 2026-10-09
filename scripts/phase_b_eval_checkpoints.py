"""
Script de Evaluación de Checkpoints para Fase B:
- Evalúa sistemáticamente checkpoints guardados sobre el conjunto de VALIDACIÓN (200 episodios).
- Modo determinista riguroso importando evaluator.py de Fase A.
- Exporta métricas agregadas por checkpoint e intervalos de Wilson.
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
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import evaluate_on_dataset
from src.evaluation.datasets import save_and_get_datasets
from src.evaluation.metrics import wilson_score_interval

def extract_step_from_dirname(dir_name: str) -> int:
    """Extrae el número de paso de nombres tipo ckpt_40960 o ppo_X5_near_40960_steps."""
    m = re.search(r"(\d+)", dir_name)
    return int(m.group(1)) if m else -1

def evaluate_single_checkpoint(ckpt_dir: Path, val_dataset: list, env_cfg: dict,
                               thresholds_path: str = "configs/phase_a/thresholds.yaml") -> dict:
    """
    Carga un checkpoint (model.zip + vecnormalize_stats.pkl) y lo evalúa sobre el dataset.
    """
    model_path = ckpt_dir / "model.zip"
    if not model_path.exists():
        # Intentar buscar cualquier .zip dentro del checkpoint
        zips = list(ckpt_dir.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"No se encontró archivo de modelo .zip en {ckpt_dir}")
        model_path = zips[0]

    norm_path = ckpt_dir / "vecnormalize_stats.pkl"
    norm_path_str = str(norm_path) if norm_path.exists() else None

    # Cargar agente PPO
    model = PPO.load(str(model_path))

    # Evaluar determinísticamente con el evaluador de Fase A
    episodes_df, agg_metrics = evaluate_on_dataset(
        agent=model,
        dataset=val_dataset,
        dataset_name="validation",
        eval_mode="deterministic",
        thresholds_path=thresholds_path,
        env_config=env_cfg,
        vec_normalize_path=norm_path_str
    )

    n_total = len(episodes_df)
    n_succ = int(episodes_df["success"].sum())
    p_hat, ci_low, ci_high = wilson_score_interval(n_succ, n_total)

    # Mediana de pasos al éxito (solo episodios exitosos)
    succ_df = episodes_df[episodes_df["success"]]
    med_steps = float(succ_df["steps"].median()) if len(succ_df) > 0 else np.nan

    # Medias operativas
    mean_dist = float(episodes_df["final_distance"].mean())
    mean_delta_a = float(episodes_df["mean_action_delta"].dropna().mean()) if "mean_action_delta" in episodes_df else np.nan

    trunc_count = int(episodes_df["truncated"].sum())
    ball_count = int(episodes_df["ballistic_flag"].sum())
    osc_count = int(episodes_df["oscillatory_flag"].sum())

    summary = {
        "checkpoint_dir": str(ckpt_dir),
        "total_episodes": n_total,
        "successes": n_succ,
        "strict_success_rate": p_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "truncation_rate": trunc_count / n_total if n_total > 0 else np.nan,
        "ballistic_rate": ball_count / n_total if n_total > 0 else np.nan,
        "oscillatory_rate": osc_count / n_total if n_total > 0 else np.nan,
        "steps_to_success_median": med_steps,
        "mean_final_distance": mean_dist,
        "mean_action_delta": mean_delta_a,
        "has_vecnormalize": norm_path.exists()
    }
    return summary, episodes_df

def evaluate_experiment_checkpoints(exp_dir: Path, cfg_path: Path, output_csv: Path = None) -> pd.DataFrame:
    """
    Evalúa todos los checkpoints ordenados por timesteps para una corrida.
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f)

    val_dataset, _ = save_and_get_datasets()

    ckpts_base = exp_dir / "checkpoints"
    if not ckpts_base.exists():
        raise FileNotFoundError(f"Directorio de checkpoints no existe: {ckpts_base}")

    # Listar carpetas ckpt_*
    ckpt_dirs = [p for p in ckpts_base.iterdir() if p.is_dir()]
    ckpt_dirs = sorted(ckpt_dirs, key=lambda p: extract_step_from_dirname(p.name))

    results = []
    print(f"\n>>> [EVAL CHECKPOINTS] Evaluando {len(ckpt_dirs)} checkpoints en {exp_dir}...")
    for c_dir in ckpt_dirs:
        step = extract_step_from_dirname(c_dir.name)
        summ, _ = evaluate_single_checkpoint(c_dir, val_dataset, env_cfg)
        summ["step"] = step
        summ["experiment"] = env_cfg.get("experiment", {}).get("name", exp_dir.parent.name)
        # extraer semilla de nombre de carpeta (ej. seed_42)
        m_seed = re.search(r"seed_(\d+)", str(exp_dir))
        summ["seed"] = int(m_seed.group(1)) if m_seed else 42
        results.append(summ)
        print(f"  Step {step:>7,}: Exito Estricto = {summ['strict_success_rate']*100:5.1f}% [{summ['ci_low']*100:5.1f}%, {summ['ci_high']*100:5.1f}%] | Pasos Mediana = {summ['steps_to_success_median']}")

    df = pd.DataFrame(results)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f">>> [OK] Resultados guardados en: {output_csv}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación de Checkpoints Fase B")
    parser.add_argument("--exp_dir", type=str, required=True, help="Ruta al directorio de la corrida (ej. experiments/phase_b/X5_near/seed_42)")
    parser.add_argument("--config", type=str, required=True, help="Ruta al YAML de configuración")
    parser.add_argument("--output_csv", type=str, default=None, help="Ruta al CSV de salida")
    args = parser.parse_args()

    evaluate_experiment_checkpoints(
        exp_dir=Path(args.exp_dir),
        cfg_path=Path(args.config),
        output_csv=Path(args.output_csv) if args.output_csv else None
    )
