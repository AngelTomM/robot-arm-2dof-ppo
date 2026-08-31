"""
Script de Entrenamiento Multialgoritmo (PPO / SAC) para Brazo Robótico 2D.
Soporta archivos de configuración YAML externos (configs/*.yaml) y registro automático de metadatos.
"""

import os
import sys
import json
import argparse
import datetime
import yaml
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from arm2d_env import Arm2DEnv


def load_config(config_path):
    """Carga configuración desde YAML o retorna configuración por defecto."""
    if config_path and os.path.exists(config_path):
        print(f">>> [CONFIG] Cargando configuración desde: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        print(">>> [CONFIG] No se especificó YAML válido. Usando configuración base predeterminada (PPO).")
        return {
            "algorithm": "PPO",
            "experiment": {"name": "exp1_baseline", "description": "Base", "author": "Default"},
            "env": {"l1": 1.0, "l2": 1.0, "max_speed": 2.0, "target_radius": 0.08, "reward_type": "distance_only"},
            "ppo": {"learning_rate": 3e-4, "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.001, "num_envs": 4},
            "training": {"total_timesteps": 150000, "eval_freq": 5000, "n_eval_episodes": 10, "save_freq": 25000}
        }


def save_experiment_metadata(models_dir, cfg, config_path=None):
    """Guarda un registro JSON con todos los parámetros del experimento tanto en models/ como en reports/runs/."""
    exp_name = cfg.get("experiment", {}).get("name", "exp_default")
    author = cfg.get("experiment", {}).get("author", "anonimo").replace("/", "_").replace(" ", "_")

    meta = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_file": config_path or "default",
        "parameters": cfg
    }
    
    # 1. Guardar localmente junto al modelo
    meta_path = os.path.join(models_dir, "experiment_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f">>> [METADATA] Parámetros registrados en: {meta_path}")

    # 2. Guardar en reports/runs/ para que se comparta por Git
    runs_dir = "reports/runs"
    os.makedirs(runs_dir, exist_ok=True)
    shared_run_path = os.path.join(runs_dir, f"{exp_name}_{author}.json")
    with open(shared_run_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f">>> [METADATA] Resumen de sesión guardado para el equipo en: {shared_run_path}")


def train(config_path=None):
    cfg = load_config(config_path)

    algo_name = cfg.get("algorithm", "PPO").upper()
    exp_name = cfg.get("experiment", {}).get("name", "exp_default")
    models_dir = f"models/{exp_name}"
    log_dir = f"logs/{exp_name}"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Guardar metadatos del experimento
    save_experiment_metadata(models_dir, cfg, config_path)

    train_cfg = cfg.get("training", {})

    # 1. Configuración de Entornos y Algoritmo según PPO o SAC
    if algo_name == "SAC":
        sac_cfg = cfg.get("sac", {})
        num_envs = int(sac_cfg.get("num_envs", 1))
        print(f">>> [ALGO: SAC] Creando {num_envs} entorno(s) para '{exp_name}'...")

        vec_env = make_vec_env(lambda: Arm2DEnv(config=cfg), n_envs=num_envs)
        eval_env = make_vec_env(lambda: Arm2DEnv(config=cfg), n_envs=1)

        ent_coef_val = sac_cfg.get("ent_coef", "auto")
        if isinstance(ent_coef_val, str) and ent_coef_val.lower() == "auto":
            ent_coef = "auto"
        else:
            ent_coef = float(ent_coef_val)

        print(f">>> Inicializando SAC con LR={sac_cfg.get('learning_rate', 3e-4)}, Buffer={sac_cfg.get('buffer_size', 100_000)}...")
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=float(sac_cfg.get("learning_rate", 3e-4)),
            buffer_size=int(sac_cfg.get("buffer_size", 100_000)),
            learning_starts=int(sac_cfg.get("learning_starts", 1000)),
            batch_size=int(sac_cfg.get("batch_size", 256)),
            tau=float(sac_cfg.get("tau", 0.005)),
            gamma=float(sac_cfg.get("gamma", 0.99)),
            train_freq=int(sac_cfg.get("train_freq", 1)),
            gradient_steps=int(sac_cfg.get("gradient_steps", 1)),
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log=log_dir
        )
    else:  # PPO por defecto
        ppo_cfg = cfg.get("ppo", {})
        num_envs = int(ppo_cfg.get("num_envs", 4))
        print(f">>> [ALGO: PPO] Creando {num_envs} entornos vectorizados paralelos para '{exp_name}'...")

        vec_env = make_vec_env(lambda: Arm2DEnv(config=cfg), n_envs=num_envs)
        eval_env = make_vec_env(lambda: Arm2DEnv(config=cfg), n_envs=1)

        print(f">>> Inicializando PPO con LR={ppo_cfg.get('learning_rate', 3e-4)}, n_steps={ppo_cfg.get('n_steps', 1024)}...")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
            n_steps=int(ppo_cfg.get("n_steps", 1024)),
            batch_size=int(ppo_cfg.get("batch_size", 64)),
            n_epochs=int(ppo_cfg.get("n_epochs", 10)),
            gamma=float(ppo_cfg.get("gamma", 0.99)),
            gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
            clip_range=float(ppo_cfg.get("clip_range", 0.2)),
            ent_coef=float(ppo_cfg.get("ent_coef", 0.001)),
            verbose=1,
            tensorboard_log=log_dir
        )

    # 2. Callbacks de Evaluación y Checkpoints
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{models_dir}/best_model",
        log_path=log_dir,
        eval_freq=int(train_cfg.get("eval_freq", 5000)),
        n_eval_episodes=int(train_cfg.get("n_eval_episodes", 10)),
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=int(train_cfg.get("save_freq", 25000)),
        save_path=f"{models_dir}/checkpoints",
        name_prefix=f"{algo_name.lower()}_arm2d"
    )

    # 3. Iniciar Entrenamiento
    total_timesteps = int(train_cfg.get("total_timesteps", 150000))
    print(f">>> Iniciando entrenamiento con {algo_name} por {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
        tb_log_name=exp_name
    )

    # 4. Guardar Modelo Final
    final_model_path = f"{models_dir}/{algo_name.lower()}_arm2d_final"
    model.save(final_model_path)
    print(f"\n>>> [OK] ¡Entrenamiento completado! Modelo final guardado en: {final_model_path}.zip")
    print(f">>> [OK] Mejor modelo guardado en: {models_dir}/best_model/best_model.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar Brazo Robótico 2 DOF con PPO o SAC")
    parser.add_argument("--config", "-c", type=str, default="configs/exp1_baseline.yaml", help="Ruta al archivo YAML de configuración")
    args = parser.parse_args()

    train(args.config)
