"""
Evaluación de Checkpoints para Fase D (MDP Continuo con Regulación LQR).

Evalúa sistemáticamente checkpoints guardados (carpetas ``ckpt_<paso>/`` con
``model.zip`` + ``vecnormalize_stats.pkl``) o controladores analíticos (C0)
sobre los conjuntos de validación congelados en modo determinista riguroso
utilizando la evaluación dual desacoplada:
  - Opción 2 (Principal): Estabilización asintótica en los últimos k=20 pasos
    (mean_dist_20 <= 0.06 m y mean_vel_20 <= 0.10 rad/s).
  - Opción 3 (Comparativa Canónica): Costo integral LQR canónico
    J_LQR = -sum(1.0*d^2 + 0.05*v^2 + 0.01*||a||^2) con desglose (pos, vel, ctrl).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO, SAC

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import evaluate_on_dataset
from src.evaluation.metrics import wilson_score_interval
from src.controllers.jacobian_controller import JacobianDampedController
from src.utils.config import load_config, load_env_config


def extract_step_from_dirname(dir_name: str) -> int:
    """Extrae el número de paso de nombres tipo ckpt_40960 o ppo_X5_near_40960_steps."""
    m = re.search(r"(\d+)", str(dir_name))
    return int(m.group(1)) if m else -1


def load_agent_model(model_path: Path, algo_hint: str = None):
    """Carga modelo SB3 detectando automáticamente si es SAC o PPO."""
    if algo_hint == "SAC" or "SAC" in str(model_path) or "S1" in str(model_path):
        try:
            return SAC.load(str(model_path))
        except Exception:
            return PPO.load(str(model_path))
    else:
        try:
            return PPO.load(str(model_path))
        except Exception:
            return SAC.load(str(model_path))


def _resolve_model_path(ckpt_dir: Path) -> Path:
    """model.zip con respaldo a cualquier *.zip del directorio del checkpoint."""
    model_path = ckpt_dir / "model.zip"
    if not model_path.exists():
        zips = list(ckpt_dir.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"No se encontró archivo de modelo .zip en {ckpt_dir}")
        model_path = zips[0]
    return model_path


def _operational_metrics(episodes_df: pd.DataFrame) -> dict:
    """Métricas operativas duales (Opción 2 y Opción 3) de un bloque de episodios."""
    n_total = len(episodes_df)
    n_succ = int(episodes_df["success"].sum()) if "success" in episodes_df else 0
    p_hat, ci_low, ci_high = wilson_score_interval(n_succ, n_total)

    mean_dist = float(episodes_df["final_distance"].mean()) if "final_distance" in episodes_df else np.nan
    mean_dist_20 = float(episodes_df["mean_dist_last_20"].mean()) if "mean_dist_last_20" in episodes_df else mean_dist
    mean_vel_20 = float(episodes_df["mean_vel_last_20"].mean()) if "mean_vel_last_20" in episodes_df else np.nan
    j_lqr = float(episodes_df["j_lqr"].mean()) if "j_lqr" in episodes_df else np.nan
    j_pos = float(episodes_df["j_pos"].mean()) if "j_pos" in episodes_df else np.nan
    j_vel = float(episodes_df["j_vel"].mean()) if "j_vel" in episodes_df else np.nan
    j_ctrl = float(episodes_df["j_ctrl"].mean()) if "j_ctrl" in episodes_df else np.nan
    mean_delta_a = (float(episodes_df["mean_action_delta"].dropna().mean())
                    if "mean_action_delta" in episodes_df and len(episodes_df["mean_action_delta"].dropna()) > 0 else np.nan)

    return {
        "n_total": n_total,
        "n_succ": n_succ,
        "p_hat": p_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mean_dist": mean_dist,
        "mean_dist_20": mean_dist_20,
        "mean_vel_20": mean_vel_20,
        "j_lqr": j_lqr,
        "j_pos": j_pos,
        "j_vel": j_vel,
        "j_ctrl": j_ctrl,
        "mean_delta_a": mean_delta_a,
    }


def _format_row(m: dict, schema: str = "d", **identity) -> dict:
    """Formatea métricas duales para tablas de resultados y CSVs."""
    row = {
        "checkpoint_dir": identity.get("checkpoint_dir"),
        "total_episodes": m["n_total"],
        "successes": m["n_succ"],
        "success_rate": m["p_hat"],
        "ci_low": m["ci_low"],
        "ci_high": m["ci_high"],
        "mean_dist_last_20": m["mean_dist_20"],
        "mean_vel_last_20": m["mean_vel_20"],
        "mean_final_distance": m["mean_dist"],
        "mean_j_lqr": m["j_lqr"],
        "mean_j_pos": m["j_pos"],
        "mean_j_vel": m["j_vel"],
        "mean_j_ctrl": m["j_ctrl"],
        "mean_action_delta": m["mean_delta_a"],
        "has_vecnormalize": identity.get("has_vecnormalize"),
    }
    if schema in ("b", "c"):
        row["strict_success_rate"] = m["p_hat"]
        row["n_episodes"] = m["n_total"]
    return row


def evaluate_single_checkpoint(ckpt_dir: Path, val_dataset: list, env_cfg: dict = None,
                               dataset_name: str = "validation",
                               eval_mode: str = "deterministic",
                               algo_hint: str = None,
                               schema: str = "d") -> tuple:
    """
    Carga un checkpoint (model.zip + vecnormalize_stats.pkl) y lo evalúa sobre
    el dataset indicado con métricas duales desacopladas.
    """
    ckpt_dir = Path(ckpt_dir)
    model_path = _resolve_model_path(ckpt_dir)

    norm_path = ckpt_dir / "vecnormalize_stats.pkl"
    norm_path_str = str(norm_path) if norm_path.exists() else None

    model = load_agent_model(model_path, algo_hint)

    episodes_df, agg_metrics = evaluate_on_dataset(
        agent=model,
        dataset=val_dataset,
        dataset_name=dataset_name,
        eval_mode=eval_mode,
        env_config=env_cfg,
        vec_normalize_path=norm_path_str
    )

    summary = _format_row(_operational_metrics(episodes_df), schema=schema,
                          checkpoint_dir=str(ckpt_dir),
                          has_vecnormalize=norm_path.exists())
    return summary, episodes_df


def evaluate_c0_controller(val_dataset: list, env_cfg: dict = None, max_speed: float = None,
                           label: str = "v1.5", dataset_name: str = "validation") -> tuple:
    """
    Evalúa el controlador cinemático analítico C0 bajo las métricas duales canónicas.
    """
    if env_cfg is None:
        try:
            env_cfg = load_env_config()
        except Exception:
            env_cfg = {"env": {}}

    speed = max_speed if max_speed is not None else float(env_cfg.get("env", {}).get("max_speed", 1.5))
    c0 = JacobianDampedController(kp=3.0, lambda_damping=0.01, max_speed=speed, env_config=env_cfg)
    episodes_df, _ = evaluate_on_dataset(
        agent=c0,
        dataset=val_dataset,
        dataset_name=dataset_name,
        eval_mode="deterministic",
        env_config=env_cfg
    )

    m = _operational_metrics(episodes_df)
    summary = _format_row(m, schema="d", checkpoint_dir=f"C0_{label}_analytic",
                          has_vecnormalize=False)
    summary["experiment"] = f"C0_{label}"
    summary["step"] = -1
    summary["seed"] = 0
    return summary, episodes_df


def evaluate_experiment_checkpoints(exp_dir: Path, cfg_path: Path = None, output_csv: Path = None,
                                    dataset_path: str = None) -> pd.DataFrame:
    """
    Evalúa todos los checkpoints ordenados por timesteps para una corrida.
    """
    if cfg_path and Path(cfg_path).exists():
        env_cfg = load_config(cfg_path)
    else:
        try:
            env_cfg = load_env_config()
        except Exception:
            env_cfg = {"env": {}}

    if dataset_path and Path(dataset_path).exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            val_dataset = json.load(f)
    else:
        from src.evaluation.datasets import save_and_get_datasets
        val_dataset, _ = save_and_get_datasets()

    ckpts_base = exp_dir / "checkpoints"
    if not ckpts_base.exists():
        raise FileNotFoundError(f"Directorio de checkpoints no existe: {ckpts_base}")

    ckpt_dirs = [p for p in ckpts_base.iterdir() if p.is_dir()]
    ckpt_dirs = sorted(ckpt_dirs, key=lambda p: extract_step_from_dirname(p.name))

    results = []
    print(f"\n>>> [EVAL CHECKPOINTS] Evaluando {len(ckpt_dirs)} checkpoints en {exp_dir}...")
    for c_dir in ckpt_dirs:
        step = extract_step_from_dirname(c_dir.name)
        summ, _ = evaluate_single_checkpoint(c_dir, val_dataset, env_cfg,
                                             dataset_name="validation", schema="d")
        summ["step"] = step
        exp_name = (env_cfg.get("experiment", {}).get("name") if env_cfg else None) or exp_dir.parent.name
        summ["experiment"] = exp_name
        m_seed = re.search(r"seed_(\d+)", str(exp_dir))
        summ["seed"] = int(m_seed.group(1)) if m_seed else 42
        results.append(summ)
        print(f"  Step {step:>7,}: Éxito Op2 = {summ['success_rate']*100:5.1f}% [{summ['ci_low']*100:5.1f}%, {summ['ci_high']*100:5.1f}%] | J_LQR = {summ['mean_j_lqr']:7.2f} | d_20 = {summ['mean_dist_last_20']*100:4.1f}cm")

    df = pd.DataFrame(results)
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f">>> [OK] Resultados guardados en: {output_csv}")
    return df


def run_phase_d_evaluation(dataset_path: str = "data/datasets/validation_set_200.json",
                           output_dir: str = "reports/data/phase_X",
                           eval_mode: str = "deterministic"):
    """
    Runner de evaluación para agentes Fase D (PPO_1, SAC_1) y referencia analítica C0.
    """
    val_file = REPO_ROOT / dataset_path
    if not val_file.exists():
        from src.evaluation.datasets import save_and_get_datasets
        val_dataset, _ = save_and_get_datasets()
    else:
        with open(val_file, "r", encoding="utf-8") as f:
            val_dataset = json.load(f)

    out_path = REPO_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    targets = [
        {"agent": "PPO_1", "algo": "PPO", "base_dir": REPO_ROOT / "experiments" / "phase_X" / "prueba" / "PPO_1"},
        {"agent": "SAC_1", "algo": "SAC", "base_dir": REPO_ROOT / "experiments" / "phase_X" / "prueba" / "SAC_1"},
    ]

    all_rows = []
    print(f"\n{'='*75}")
    print(f" EVALUACIÓN DE CHECKPOINTS FASE D ({eval_mode.upper()})")
    print(f" Dataset: {dataset_path} ({len(val_dataset)} episodios)")
    print(f"{'='*75}\n")

    for target in targets:
        agent_id = target["agent"]
        algo_hint = target["algo"]
        base_dir = target["base_dir"]

        if not base_dir.exists():
            continue

        seed_dirs = sorted([d for d in base_dir.glob("seed_*") if d.is_dir()])
        for sdir in seed_dirs:
            seed_str = sdir.name.replace("seed_", "")
            ckpt_dirs = sorted([d for d in (sdir / "checkpoints").glob("ckpt_*") if d.is_dir()],
                               key=lambda p: extract_step_from_dirname(p.name))

            for cdir in ckpt_dirs:
                step = extract_step_from_dirname(cdir.name)
                print(f" Evaluando: {agent_id} | Semilla: {seed_str} | Paso: {step:,}...", end="", flush=True)
                row, _ = evaluate_single_checkpoint(cdir, val_dataset, env_cfg=None,
                                                    dataset_name="validation",
                                                    eval_mode=eval_mode,
                                                    algo_hint=algo_hint,
                                                    schema="d")
                row["experiment"] = agent_id
                row["seed"] = int(seed_str) if seed_str.isdigit() else seed_str
                row["step"] = step
                row["eval_mode"] = eval_mode
                all_rows.append(row)
                print(f" -> Éxito Op2: {row['success_rate']*100:.2f}% | J_LQR: {row['mean_j_lqr']:.2f}")

    # Evaluar referencia C0
    print("\n Evaluando referencia analítica C0 (v_max = 1.5)...", end="", flush=True)
    c0_row, _ = evaluate_c0_controller(val_dataset, env_cfg=None, max_speed=1.5, label="v1.5")
    c0_row["eval_mode"] = eval_mode
    all_rows.append(c0_row)
    print(f" -> Éxito Op2: {c0_row['success_rate']*100:.2f}% | J_LQR: {c0_row['mean_j_lqr']:.2f}")

    details_df = pd.DataFrame(all_rows)
    suffix = f"_{eval_mode}" if eval_mode != "deterministic" else ""
    details_csv_path = out_path / f"checkpoint_eval{suffix}.csv"
    details_df.to_csv(details_csv_path, index=False)
    print(f"\n>>> [CSV] Resultados guardados en: {details_csv_path}")
    return details_df


# Alias de compatibilidad
run_phase_c_evaluation = run_phase_d_evaluation

