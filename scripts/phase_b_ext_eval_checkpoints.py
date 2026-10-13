"""
Script de Evaluación de Checkpoints para Fase B (Extensión):
- Evalúa sistemáticamente los 16 checkpoints (paso <= 655.360) sobre Validación B2 (500 episodios).
- Para X3 y X5_near, lee de experiments/phase_b/ filtrando paso <= 655.360.
- Para las nuevas candidatas, lee de experiments/phase_b/extension/.
- Evalúa también ambas versiones de C0 (max_speed=1.5 y 2.0).
- Modo determinista riguroso importando evaluator.py de Fase A.
- Exporta métricas individuales y agregadas a reports/data/phase_b/extension/.
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
from src.evaluation.metrics import wilson_score_interval
from src.controllers.jacobian_controller import JacobianDampedController

def extract_step_from_dirname(dir_name: str) -> int:
    m = re.search(r"(\d+)", dir_name)
    return int(m.group(1)) if m else -1

def evaluate_single_checkpoint(ckpt_dir: Path, val_dataset: list, env_cfg: dict,
                               thresholds_path: str = "configs/phase_a/thresholds.yaml") -> tuple:
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

    model = PPO.load(str(model_path))

    episodes_df, agg_metrics = evaluate_on_dataset(
        agent=model,
        dataset=val_dataset,
        dataset_name="validation_b2",
        eval_mode="deterministic",
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

def evaluate_c0_controller(val_dataset: list, env_cfg: dict, max_speed: float, label: str) -> dict:
    """
    Evalúa el controlador C0 analítico sobre el dataset.
    """
    c0 = JacobianDampedController(kp=3.0, lambda_damping=0.01, max_speed=max_speed)
    episodes_df, _ = evaluate_on_dataset(
        agent=c0,
        dataset=val_dataset,
        dataset_name="validation_b2",
        eval_mode="deterministic",
        env_config=env_cfg
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

    summary = {
        "experiment": f"C0_{label}",
        "step": -1,
        "seed": 0,
        "checkpoint_dir": f"C0_{label}_analytic",
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
        "has_vecnormalize": False
    }
    return summary, episodes_df
