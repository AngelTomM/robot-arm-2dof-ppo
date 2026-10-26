#!/usr/bin/env python3
"""
Generador de la Matriz Unificada de Benchmarking (SSOT).
Recorre todas las configuraciones YAML en configs/ y asocia las métricas de evaluación
existentes en reports/data/evaluation/ generando un archivo CSV estructurado.
"""

import os
import glob
import json
import yaml
import pandas as pd
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = WORKSPACE_ROOT / "configs"
ENV_CONFIG_PATH = CONFIGS_DIR / "environment.yaml"
EVAL_DATA_DIR = WORKSPACE_ROOT / "reports" / "data" / "evaluation"
OUTPUT_CSV_PATH = WORKSPACE_ROOT / "reports" / "data" / "benchmark_matrix.csv"

def load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error cargando {path}: {e}")
            return {}

def load_summary(model_name: str) -> dict:
    # Prioridad: carpeta del modelo reports/data/evaluation/<model>/summary.json
    path_folder = EVAL_DATA_DIR / model_name / "summary.json"
    if path_folder.is_file():
        with open(path_folder, "r", encoding="utf-8") as f:
            try:
                return json.load(f) or {}
            except Exception:
                pass
    
    # Alternativa: archivo plano reports/data/evaluation/<model>_summary.json
    path_flat = EVAL_DATA_DIR / f"{model_name}_summary.json"
    if path_flat.is_file():
        with open(path_flat, "r", encoding="utf-8") as f:
            try:
                return json.load(f) or {}
            except Exception:
                pass

    return {}

def main():
    env_base = load_yaml(ENV_CONFIG_PATH).get("env", {})
    base_reward = env_base.get("reward_params", {})

    yaml_files = sorted(glob.glob(str(CONFIGS_DIR / "**" / "*.yaml"), recursive=True))
    rows = []

    for yf_str in yaml_files:
        yf = Path(yf_str)
        if yf.resolve() == ENV_CONFIG_PATH.resolve():
            continue  # Omitir el archivo de entorno base aislado

        cfg = load_yaml(yf)
        rel_path = yf.relative_to(WORKSPACE_ROOT).as_posix()
        model_name = yf.stem
        subfolder = yf.parent.name

        # Identificar algoritmo
        if "ppo" in cfg:
            algo = "PPO"
            algo_cfg = cfg.get("ppo", {})
        elif "sac" in cfg:
            algo = "SAC"
            algo_cfg = cfg.get("sac", {})
        elif "controller" in cfg:
            algo = cfg.get("controller", {}).get("name", "C0")
            algo_cfg = cfg.get("controller", {})
        else:
            algo = "DESCONOCIDO"
            algo_cfg = {}

        # Preprocesamiento y normalización
        norm = cfg.get("normalization", {})
        norm_obs = norm.get("norm_obs", None)
        norm_reward = norm.get("norm_reward", None)
        clip_obs = norm.get("clip_obs", None)

        # Entrenamiento
        training = cfg.get("training", {})
        total_timesteps = training.get("total_timesteps", None)

        # Arquitectura de red
        net_arch_info = algo_cfg.get("net_arch", {})
        if isinstance(net_arch_info, dict):
            layers = str(net_arch_info.get("layers", ""))
            activation = net_arch_info.get("hidden_activation", "")
        elif isinstance(net_arch_info, list):
            layers = str(net_arch_info)
            activation = ""
        else:
            layers = ""
            activation = ""

        # Parámetros del entorno para este agente
        # Si tiene sobreescritura de entorno o usa el base
        w_pos = base_reward.get("w_pos", None)
        w_vel = base_reward.get("w_vel", None)
        w_ctrl = base_reward.get("w_ctrl", None)
        target_radius = env_base.get("target_radius", None)
        max_steps = env_base.get("max_steps", None)
        frame_skip = env_base.get("frame_skip", None)
        dt = env_base.get("dt", None)
        dt_eff = (dt * frame_skip) if (dt is not None and frame_skip is not None) else None

        # Métricas de evaluación
        summary = load_summary(model_name)
        metrics = summary.get("metrics", {})
        eval_set = summary.get("eval_set", None)
        total_eps = metrics.get("total_episodes", None)
        success_rate = metrics.get("success_rate", None)
        success_rate_pct = round(success_rate * 100.0, 2) if success_rate is not None else None
        ci_low = metrics.get("ci_low", None)
        ci_low_pct = round(ci_low * 100.0, 2) if ci_low is not None else None
        ci_high = metrics.get("ci_high", None)
        ci_high_pct = round(ci_high * 100.0, 2) if ci_high is not None else None

        e_ss_mean = metrics.get("e_ss_mean", None)
        e_ss_median = metrics.get("e_ss_median", None)
        e_ss_p90 = metrics.get("e_ss_p90", None)
        v_ss_mean = metrics.get("v_ss_mean", None)
        t_s_mean = metrics.get("t_s_mean", None)
        j_u_mean = metrics.get("j_u_mean", None)
        j_lqr_mean = metrics.get("j_lqr_mean", None)
        mean_action_delta = metrics.get("mean_action_delta", None)

        row = {
            "model_id": model_name,
            "group": subfolder,
            "algorithm": algo,
            "config_path": rel_path,
            "description": cfg.get("experiment", {}).get("description", ""),
            # Parámetros de Entorno (SSOT)
            "w_pos": w_pos,
            "w_vel": w_vel,
            "w_ctrl": w_ctrl,
            "target_radius_m": target_radius,
            "dt_effective_s": dt_eff,
            "max_steps": max_steps,
            # Normalización
            "norm_obs": norm_obs,
            "norm_reward": norm_reward,
            "clip_obs": clip_obs,
            # Hiperparámetros Generales RL
            "net_arch": layers,
            "activation": activation,
            "learning_rate": algo_cfg.get("learning_rate", None),
            "gamma": algo_cfg.get("gamma", None),
            "batch_size": algo_cfg.get("batch_size", None),
            "total_timesteps": total_timesteps,
            # Específicos PPO
            "n_steps": algo_cfg.get("n_steps", None),
            "n_epochs": algo_cfg.get("n_epochs", None),
            "gae_lambda": algo_cfg.get("gae_lambda", None),
            "clip_range": algo_cfg.get("clip_range", None),
            "ent_coef": algo_cfg.get("ent_coef", None),
            "max_grad_norm": algo_cfg.get("max_grad_norm", None),
            "num_envs": algo_cfg.get("num_envs", None),
            # Específicos SAC
            "buffer_size": algo_cfg.get("buffer_size", None),
            "learning_starts": algo_cfg.get("learning_starts", None),
            "tau": algo_cfg.get("tau", None),
            "train_freq": algo_cfg.get("train_freq", None),
            "gradient_steps": algo_cfg.get("gradient_steps", None),
            "target_entropy": algo_cfg.get("target_entropy", None),
            # Específicos C0
            "c0_kp": algo_cfg.get("kp", None),
            "c0_kd": algo_cfg.get("kd", None),
            # Métricas de Evaluación
            "eval_set": eval_set,
            "eval_episodes": total_eps,
            "success_rate_pct": success_rate_pct,
            "ci_low_pct": ci_low_pct,
            "ci_high_pct": ci_high_pct,
            "e_ss_mean_m": round(e_ss_mean, 5) if e_ss_mean is not None else None,
            "e_ss_median_m": round(e_ss_median, 5) if e_ss_median is not None else None,
            "e_ss_p90_m": round(e_ss_p90, 5) if e_ss_p90 is not None else None,
            "v_ss_mean_rad_s": round(v_ss_mean, 4) if v_ss_mean is not None else None,
            "t_s_mean_s": round(t_s_mean, 3) if t_s_mean is not None else None,
            "j_u_mean": round(j_u_mean, 4) if j_u_mean is not None else None,
            "j_lqr_mean": round(j_lqr_mean, 3) if j_lqr_mean is not None else None,
            "mean_action_delta": round(mean_action_delta, 5) if mean_action_delta is not None else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
    print(f"Matriz generada exitosamente en: {OUTPUT_CSV_PATH}")
    print(f"Total modelos indexados: {len(df)}")
    print(f"Modelos con evaluacion vinculada: {df['success_rate_pct'].notnull().sum()}")

if __name__ == "__main__":
    main()
