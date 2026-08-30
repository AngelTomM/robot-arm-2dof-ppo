"""
Script de Entrenamiento con PPO para Brazo Robótico 2D.
Soporta archivos de configuración YAML externos (configs/*.yaml) y registro automático de metadatos.
"""

import os
import sys
import json
import argparse
import datetime
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
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
        print(">>> [CONFIG] No se especificó YAML válido. Usando configuración base predeterminada.")
        return {
            "experiment": {"name": "exp1_baseline", "description": "Base", "author": "Default"},
            "env": {"l1": 1.0, "l2": 1.0, "max_speed": 2.0, "target_radius": 0.08, "reward_type": "distance_only"},
            "ppo": {"learning_rate": 3e-4, "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.001, "num_envs": 4},
            "training": {"total_timesteps": 150000, "eval_freq": 5000, "n_eval_episodes": 10, "save_freq": 25000}
        }


def save_experiment_metadata(models_dir, cfg, config_path=None):
    """Guarda un registro JSON con todos los parámetros del experimento."""
    meta = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_file": config_path or "default",
        "parameters": cfg
    }
    meta_path = os.path.join(models_dir, "experiment_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f">>> [METADATA] Parámetros registrados en: {meta_path}")


def train(config_path=None):
    cfg = load_config(config_path)

    exp_name = cfg.get("experiment", {}).get("name", "exp_default")
    models_dir = f"models/{exp_name}"
    log_dir = f"logs/{exp_name}"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Guardar metadatos del experimento
    save_experiment_metadata(models_dir, cfg, config_path)

    # 1. Crear Entornos Vectorizados
    ppo_cfg = cfg.get("ppo", {})
    num_envs = int(ppo_cfg.get("num_envs", 4))
    print(f">>> Creando {num_envs} entornos vectorizados paralelos para '{exp_name}'...")

    vec_env = make_vec_env(lambda: Arm2DEnv(config=cfg), n_envs=num_envs)
    eval_env = make_vec_env(lambda: Arm2DEnv(config=cfg), n_envs=1)

    # 2. Callbacks
    train_cfg = cfg.get("training", {})
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
        name_prefix="ppo_arm2d"
    )

    # 3. Agente PPO
    print(f">>> Inicializando PPO con LR={ppo_cfg.get('learning_rate')}, n_steps={ppo_cfg.get('n_steps')}...")
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

    # 4. Entrenar
    total_timesteps = int(train_cfg.get("total_timesteps", 150000))
    print(f">>> Iniciando entrenamiento por {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
        tb_log_name=exp_name
    )

    # 5. Guardar Modelo Final
    final_model_path = f"{models_dir}/ppo_arm2d_final"
    model.save(final_model_path)
    print(f"\n>>> [OK] ¡Entrenamiento completado! Modelo final guardado en: {final_model_path}.zip")
    print(f">>> [OK] Mejor modelo guardado en: {models_dir}/best_model/best_model.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar PPO para Brazo Robótico 2 DOF con YAML")
    parser.add_argument("--config", "-c", type=str, default="configs/exp1_baseline.yaml", help="Ruta al archivo YAML de configuración")
    args = parser.parse_args()

    train(args.config)
