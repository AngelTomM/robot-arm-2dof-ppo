"""
Script de Evaluación de Checkpoints para Fase C:
- Evalúa los 16 checkpoints (paso <= 655.360) de S1 (SAC) y X3 (PPO) sobre Validación C (500 episodios).
- Modo determinista riguroso por defecto; soporte para modo estocástico diagnóstico.
- Genera resúmenes individuales (por semilla) y agregados (media +- std inter-semilla).
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
from stable_baselines3 import SAC, PPO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import evaluate_on_dataset
from src.evaluation.metrics import wilson_score_interval
from src.controllers.jacobian_controller import JacobianDampedController

def extract_step_from_dirname(dir_name: str) -> int:
    m = re.search(r"(\d+)", dir_name)
    return int(m.group(1)) if m else -1

def load_agent_model(model_path: Path, algo_hint: str = None):
    """Carga modelo SB3 detectando automáticamente si es SAC o PPO."""
    if algo_hint == "SAC" or "S1" in str(model_path):
        try:
            return SAC.load(str(model_path))
        except Exception:
            return PPO.load(str(model_path))
    else:
        try:
            return PPO.load(str(model_path))
        except Exception:
            return SAC.load(str(model_path))

def evaluate_single_checkpoint(ckpt_dir: Path, val_dataset: list, env_cfg: dict,
                               eval_mode: str = "deterministic",
                               thresholds_path: str = "configs/phase_a/thresholds.yaml",
                               algo_hint: str = None) -> tuple:
    """
    Carga un checkpoint (model.zip + vecnormalize_stats.pkl) y lo evalúa sobre el dataset.
    """
    model_path = ckpt_dir / "model.zip"
    if not model_path.exists():
        zips = list(ckpt_dir.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"No se encontró archivo .zip en {ckpt_dir}")
        model_path = zips[0]

    norm_path = ckpt_dir / "vecnormalize_stats.pkl"
    norm_path_str = str(norm_path) if norm_path.exists() else None

    model = load_agent_model(model_path, algo_hint)

    episodes_df, agg_metrics = evaluate_on_dataset(
        agent=model,
        dataset=val_dataset,
        dataset_name="validation_c",
        eval_mode=eval_mode,
        thresholds_path=thresholds_path,
        env_config=env_cfg,
        vec_normalize_path=norm_path_str
    )

    n_total = len(episodes_df)
    n_succ = int(episodes_df["success"].sum())
    p_hat, ci_low, ci_high = wilson_score_interval(n_succ, n_total)

    succ_df = episodes_df[episodes_df["success"]]
    med_steps = float(succ_df["steps"].median()) if len(succ_df) > 0 else np.nan
    mean_dist = float(episodes_df["final_distance"].mean())
    mean_delta_a = float(episodes_df["mean_action_delta"].dropna().mean()) if "mean_action_delta" in episodes_df else np.nan

    trunc_count = int(episodes_df["truncated"].sum())
    ball_count = int(episodes_df["ballistic_flag"].sum())
    osc_count = int(episodes_df["oscillatory_flag"].sum())

    # Calcular fracción que entró al radio en algún momento
    entered_radius = int((episodes_df["steps_to_first_target_entry"].notna()).sum())
    frac_entered = entered_radius / n_total if n_total > 0 else 0.0

    summary_row = {
        "n_episodes": n_total,
        "success_rate": p_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "median_steps_to_success": med_steps,
        "mean_final_distance": mean_dist,
        "mean_action_delta": mean_delta_a,
        "truncation_rate": trunc_count / n_total,
        "ballistic_rate": ball_count / n_total,
        "oscillatory_rate": osc_count / n_total,
        "frac_entered_radius": frac_entered
    }

    return summary_row, episodes_df

def evaluate_c0_controller(val_dataset: list, max_speed: float = 1.5,
                           thresholds_path: str = "configs/phase_a/thresholds.yaml") -> dict:
    """Evalúa el controlador analítico C0 sobre Validación C."""
    c0 = JacobianDampedController(kp=3.0, lambda_damping=0.01, max_speed=max_speed)
    env_cfg = {"env": {"max_speed": max_speed}}

    episodes_df, agg_metrics = evaluate_on_dataset(
        agent=c0,
        dataset=val_dataset,
        dataset_name="validation_c",
        eval_mode="deterministic",
        thresholds_path=thresholds_path,
        env_config=env_cfg,
        vec_normalize_path=None
    )

    n_total = len(episodes_df)
    n_succ = int(episodes_df["success"].sum())
    p_hat, ci_low, ci_high = wilson_score_interval(n_succ, n_total)
    succ_df = episodes_df[episodes_df["success"]]
    med_steps = float(succ_df["steps"].median()) if len(succ_df) > 0 else np.nan
    mean_dist = float(episodes_df["final_distance"].mean())
    mean_delta_a = float(episodes_df["mean_action_delta"].dropna().mean()) if "mean_action_delta" in episodes_df else np.nan

    return {
        "experiment": f"C0_v{max_speed}",
        "step": 655360,
        "seed": "analytic",
        "n_episodes": n_total,
        "success_rate": p_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "median_steps_to_success": med_steps,
        "mean_final_distance": mean_dist,
        "mean_action_delta": mean_delta_a,
        "truncation_rate": float(episodes_df["truncated"].sum()) / n_total,
        "ballistic_rate": float(episodes_df["ballistic_flag"].sum()) / n_total,
        "oscillatory_rate": float(episodes_df["oscillatory_flag"].sum()) / n_total,
        "frac_entered_radius": float((episodes_df["steps_to_first_target_entry"].notna()).sum()) / n_total
    }

def run_phase_c_evaluation(dataset_path: str = "data/datasets/phase_c/validation_c_500.json",
                           output_dir: str = "reports/data/phase_c",
                           max_step: int = 655360,
                           eval_mode: str = "deterministic"):
    val_file = REPO_ROOT / dataset_path
    if not val_file.exists():
        raise FileNotFoundError(f"Dataset de validación no encontrado: {val_file}")

    with open(val_file, "r", encoding="utf-8") as f:
        val_dataset = json.load(f)

    out_path = REPO_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    env_cfg = {
        "env": {
            "max_speed": 1.5,
            "success_mode": "hold",
            "hold_steps": 20,
            "velocity_tolerance": 0.10,
            "target_radius": 0.06
        }
    }

    # Definir agentes a evaluar: S1 (SAC) y X3 (PPO)
    targets = [
        {"agent": "S1", "algo": "SAC", "base_dir": REPO_ROOT / "experiments" / "phase_c" / "S1"},
        {"agent": "X3", "algo": "PPO", "base_dir": REPO_ROOT / "experiments" / "phase_b" / "X3"}
    ]

    all_rows = []
    print(f"\n{'='*75}")
    print(f" EVALUACIÓN DE CHECKPOINTS EN VALIDACIÓN C ({eval_mode.upper()})")
    print(f" Dataset: {dataset_path} ({len(val_dataset)} episodios)")
    print(f"{'='*75}\n")

    for target in targets:
        agent_id = target["agent"]
        algo_hint = target["algo"]
        base_dir = target["base_dir"]

        if not base_dir.exists():
            print(f">>> [AVISO] Directorio no encontrado para {agent_id}: {base_dir}. Omitiendo.")
            continue

        seed_dirs = sorted([d for d in base_dir.glob("seed_*") if d.is_dir()])
        for sdir in seed_dirs:
            seed_str = sdir.name.replace("seed_", "")
            ckpt_dirs = sorted([d for d in (sdir / "checkpoints").glob("ckpt_*") if d.is_dir()],
                               key=lambda p: extract_step_from_dirname(p.name))

            for cdir in ckpt_dirs:
                step = extract_step_from_dirname(cdir.name)
                if step > max_step:
                    continue

                print(f" Evaluando: {agent_id} | Semilla: {seed_str} | Paso: {step:,}...", end="", flush=True)
                row, _ = evaluate_single_checkpoint(cdir, val_dataset, env_cfg,
                                                   eval_mode=eval_mode,
                                                   algo_hint=algo_hint)
                row["experiment"] = agent_id
                row["seed"] = int(seed_str) if seed_str.isdigit() else seed_str
                row["step"] = step
                row["eval_mode"] = eval_mode
                all_rows.append(row)
                print(f" -> Éxito: {row['success_rate']*100:.2f}% | Pasos: {row['median_steps_to_success']}")

    # Evaluar también C0 como referencia
    print("\n Evaluando referencia analítica C0 (v_max = 1.5)...", end="", flush=True)
    c0_row = evaluate_c0_controller(val_dataset, max_speed=1.5)
    c0_row["eval_mode"] = eval_mode
    all_rows.append(c0_row)
    print(f" -> Éxito: {c0_row['success_rate']*100:.2f}%")

    # Guardar CSV detallado
    details_df = pd.DataFrame(all_rows)
    suffix = f"_{eval_mode}" if eval_mode != "deterministic" else ""
    details_csv_path = out_path / f"checkpoints_eval_validation_details{suffix}.csv"
    details_df.to_csv(details_csv_path, index=False)
    print(f"\n>>> [CSV] Resultados detallados guardados en: {details_csv_path}")

    # Guardar CSV agregado (media +- std muestral ddof=1)
    rl_df = details_df[details_df["seed"] != "analytic"]
    if len(rl_df) > 0:
        agg_records = []
        for (exp_name, step_val), group in rl_df.groupby(["experiment", "step"]):
            n_seeds = len(group)
            m_succ = float(group["success_rate"].mean())
            s_succ = float(group["success_rate"].std(ddof=1)) if n_seeds > 1 else np.nan
            m_steps = float(group["median_steps_to_success"].mean())
            s_steps = float(group["median_steps_to_success"].std(ddof=1)) if n_seeds > 1 else np.nan
            m_dist = float(group["mean_final_distance"].mean())
            m_delta_a = float(group["mean_action_delta"].mean())
            s_delta_a = float(group["mean_action_delta"].std(ddof=1)) if n_seeds > 1 else np.nan
            m_trunc = float(group["truncation_rate"].mean())
            m_ball = float(group["ballistic_rate"].mean())
            m_entered = float(group["frac_entered_radius"].mean())

            agg_records.append({
                "experiment": exp_name,
                "step": step_val,
                "n_seeds": n_seeds,
                "mean_success_rate": m_succ,
                "std_success_rate": s_succ,
                "mean_steps_median": m_steps,
                "std_steps_median": s_steps,
                "mean_final_distance": m_dist,
                "mean_action_delta": m_delta_a,
                "std_action_delta": s_delta_a,
                "mean_truncation_rate": m_trunc,
                "mean_ballistic_rate": m_ball,
                "mean_frac_entered_radius": m_entered,
                "eval_mode": eval_mode
            })

        agg_df = pd.DataFrame(agg_records)
        agg_csv_path = out_path / f"checkpoints_eval_validation_aggregated{suffix}.csv"
        agg_df.to_csv(agg_csv_path, index=False)
        print(f">>> [CSV] Resultados agregados guardados en: {agg_csv_path}")

    return details_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación de checkpoints Fase C")
    parser.add_argument("--dataset", type=str, default="data/datasets/phase_c/validation_c_500.json")
    parser.add_argument("--output-dir", type=str, default="reports/data/phase_c")
    parser.add_argument("--mode", type=str, default="deterministic", choices=["deterministic", "stochastic"])
    args = parser.parse_args()

    run_phase_c_evaluation(dataset_path=args.dataset, output_dir=args.output_dir, eval_mode=args.mode)
