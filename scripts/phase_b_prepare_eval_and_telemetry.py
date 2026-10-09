"""
Script Preparatorio de Datos para Figuras de Fase B:
1. Genera evaluaciones por episodio sobre VALIDACIÓN (200 eps) para los checkpoints seleccionados:
   - X5_near_phase_b_selected (ckpt_901120, semillas 42, 100, 2026, det y stoch)
   - X3_phase_b_selected (ckpt_409600, semillas 42, 100, 2026, det y stoch)
   - X5_near_phase_a (seed 42, det y stoch)
   - C0 (analítico, det)
2. Extrae telemetría paso a paso para los 4 episodios geométricos (34, 195, 40, 90) sobre VALIDACIÓN.
3. Guarda todo en reports/data/phase_b/.
"""
import sys
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import evaluate_on_dataset
from src.evaluation.datasets import save_and_get_datasets
from src.controllers.jacobian_controller import JacobianDampedController
from scripts.phase_b_step_data import collect_step_telemetry_for_episodes

def prepare_data():
    val_set, _ = save_and_get_datasets()
    data_dir = REPO_ROOT / "reports" / "data" / "phase_b"
    data_dir.mkdir(parents=True, exist_ok=True)

    geom_ids = [34, 195, 40, 90]
    geom_eps = [ep for ep in val_set if ep["episode_id"] in geom_ids]

    # Modelos a evaluar
    c0 = JacobianDampedController(kp=3.0, lambda_damping=0.01)
    
    # Cargar configuraciones
    with open(REPO_ROOT / "configs" / "phase_b" / "X5_near.yaml") as f:
        cfg_x5 = yaml.safe_load(f)
    with open(REPO_ROOT / "configs" / "phase_b" / "X3.yaml") as f:
        cfg_x3 = yaml.safe_load(f)

    # 1. Evaluación C0
    print(">>> Evaluando C0...")
    c0_df, _ = evaluate_on_dataset(c0, val_set, "validation", "deterministic", env_config=cfg_x5)
    c0_df["model_id"] = "C0"
    c0_df["seed"] = 0
    c0_df.to_csv(data_dir / "eval_val_c0_episodes.csv", index=False)

    # 2. Evaluación X5_near Fase A (seed 42)
    print(">>> Evaluando X5_near Fase A (seed 42)...")
    m_x5_a = PPO.load(str(REPO_ROOT / "experiments" / "phase_a" / "X5_near" / "seed_42" / "models" / "ppo_arm2d_final.zip"))
    norm_x5_a = str(REPO_ROOT / "experiments" / "phase_a" / "X5_near" / "seed_42" / "models" / "vecnormalize_stats.pkl")
    x5_a_det, _ = evaluate_on_dataset(m_x5_a, val_set, "validation", "deterministic", env_config=cfg_x5, vec_normalize_path=norm_x5_a)
    x5_a_det["model_id"] = "X5_near_FaseA"
    x5_a_det["seed"] = 42
    x5_a_det.to_csv(data_dir / "eval_val_x5_near_phase_a_det.csv", index=False)

    # 3. Evaluación Checkpoints Seleccionados Fase B
    # X5_near ckpt_901120
    x5_b_dfs = []
    for s in [42, 100, 2026]:
        ckpt_dir = REPO_ROOT / "experiments" / "phase_b" / "X5_near" / f"seed_{s}" / "checkpoints" / "ckpt_901120"
        m = PPO.load(str(ckpt_dir / "model.zip"))
        norm = str(ckpt_dir / "vecnormalize_stats.pkl")
        for mode in ["deterministic", "stochastic"]:
            print(f">>> Evaluando X5_near Fase B ckpt_901120 seed {s} ({mode})...")
            df, _ = evaluate_on_dataset(m, val_set, "validation", mode, env_config=cfg_x5, vec_normalize_path=norm)
            df["model_id"] = "X5_near_FaseB_ckpt901120"
            df["seed"] = s
            x5_b_dfs.append(df)
    pd.concat(x5_b_dfs, ignore_index=True).to_csv(data_dir / "eval_val_x5_near_phase_b_selected.csv", index=False)

    # X3 ckpt_409600
    x3_b_dfs = []
    for s in [42, 100, 2026]:
        ckpt_dir = REPO_ROOT / "experiments" / "phase_b" / "X3" / f"seed_{s}" / "checkpoints" / "ckpt_409600"
        m = PPO.load(str(ckpt_dir / "model.zip"))
        norm = str(ckpt_dir / "vecnormalize_stats.pkl")
        for mode in ["deterministic", "stochastic"]:
            print(f">>> Evaluando X3 Fase B ckpt_409600 seed {s} ({mode})...")
            df, _ = evaluate_on_dataset(m, val_set, "validation", mode, env_config=cfg_x3, vec_normalize_path=norm)
            df["model_id"] = "X3_FaseB_ckpt409600"
            df["seed"] = s
            x3_b_dfs.append(df)
    pd.concat(x3_b_dfs, ignore_index=True).to_csv(data_dir / "eval_val_x3_phase_b_selected.csv", index=False)

    # 4. Telemetría Paso a Paso de los 4 Episodios Geométricos
    print(">>> Extrayendo telemetría paso a paso para episodios geométricos...")
    telemetry_dfs = []

    # C0
    df_c0_step = collect_step_telemetry_for_episodes(c0, geom_eps, cfg_x5, "deterministic")
    df_c0_step["model_id"] = "C0"
    telemetry_dfs.append(df_c0_step)

    # X5_near Fase A
    df_x5a_step = collect_step_telemetry_for_episodes(m_x5_a, geom_eps, cfg_x5, "deterministic", norm_x5_a)
    df_x5a_step["model_id"] = "X5_near_FaseA"
    telemetry_dfs.append(df_x5a_step)

    # X5_near Fase B ckpt_901120 seed 42 (det y stoch)
    ckpt_x5_b = REPO_ROOT / "experiments" / "phase_b" / "X5_near" / "seed_42" / "checkpoints" / "ckpt_901120"
    m_x5_b = PPO.load(str(ckpt_x5_b / "model.zip"))
    norm_x5_b = str(ckpt_x5_b / "vecnormalize_stats.pkl")
    df_x5b_det = collect_step_telemetry_for_episodes(m_x5_b, geom_eps, cfg_x5, "deterministic", norm_x5_b)
    df_x5b_det["model_id"] = "X5_near_FaseB"
    telemetry_dfs.append(df_x5b_det)
    df_x5b_stoch = collect_step_telemetry_for_episodes(m_x5_b, geom_eps, cfg_x5, "stochastic", norm_x5_b)
    df_x5b_stoch["model_id"] = "X5_near_FaseB"
    telemetry_dfs.append(df_x5b_stoch)

    # X3 Fase B ckpt_409600 seed 42 (det y stoch)
    ckpt_x3_b = REPO_ROOT / "experiments" / "phase_b" / "X3" / "seed_42" / "checkpoints" / "ckpt_409600"
    m_x3_b = PPO.load(str(ckpt_x3_b / "model.zip"))
    norm_x3_b = str(ckpt_x3_b / "vecnormalize_stats.pkl")
    df_x3b_det = collect_step_telemetry_for_episodes(m_x3_b, geom_eps, cfg_x3, "deterministic", norm_x3_b)
    df_x3b_det["model_id"] = "X3_FaseB"
    telemetry_dfs.append(df_x3b_det)
    df_x3b_stoch = collect_step_telemetry_for_episodes(m_x3_b, geom_eps, cfg_x3, "stochastic", norm_x3_b)
    df_x3b_stoch["model_id"] = "X3_FaseB"
    telemetry_dfs.append(df_x3b_stoch)

    all_telemetry = pd.concat(telemetry_dfs, ignore_index=True)
    csv_path = data_dir / "step_data_val_geometry.csv"
    all_telemetry.to_csv(csv_path, index=False)
    print(f">>> [OK] Telemetría guardada en: {csv_path} ({len(all_telemetry)} pasos)")

if __name__ == "__main__":
    prepare_data()
