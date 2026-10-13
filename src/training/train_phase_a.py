"""
Pipeline Centralizado de Entrenamiento de Fase A con PPO (Secciones 6, 7, 11, 12).
Garantiza el orden riguroso de wrappers: Monitor -> DummyVecEnv -> VecNormalize.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import yaml
import random
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from arm2d_env import Arm2DEnv
from src.training.callbacks import PhaseATrainingCallback, PhaseAEvalCallback

def set_all_seeds(seed: int):
    """Fija todas las fuentes de aleatoriedad para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env_fn(cfg: dict, rank: int, base_seed: int):
    """Crea una función fábrica para instanciar Arm2DEnv envuelto individualmente en Monitor."""
    def _init():
        env = Arm2DEnv(config=cfg)
        env_seed = base_seed + rank * 100
        env.reset(seed=env_seed)
        # Monitor envuelve el entorno individual ANTES de la vectorización y normalización
        return Monitor(env)
    return _init

def train_agent(config_path_or_dict, seed: int = 42, total_timesteps_override: int = None,
                output_base_dir: str = "experiments/phase_a"):
    """
    Ejecuta el entrenamiento PPO completo para una configuración dada y semilla.
    """
    # 1. Cargar Configuración
    if isinstance(config_path_or_dict, str):
        with open(config_path_or_dict, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg_source = config_path_or_dict
    else:
        cfg = config_path_or_dict
        cfg_source = "dict"

    exp_id = cfg.get("experiment", {}).get("name", "exp_phase_a")
    exp_dir = os.path.join(output_base_dir, exp_id, f"seed_{seed}")
    models_dir = os.path.join(exp_dir, "models")
    logs_dir = os.path.join(exp_dir, "logs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    set_all_seeds(seed)

    # 2. Configurar Entornos Vectorizados con Monitor y VecNormalize
    ppo_cfg = cfg.get("ppo", {})
    norm_cfg = cfg.get("normalization", {})
    norm_obs = bool(norm_cfg.get("norm_obs", False))
    norm_reward = bool(norm_cfg.get("norm_reward", False))
    num_envs = int(ppo_cfg.get("num_envs", 4))

    env_fns = [make_env_fn(cfg, rank=i, base_seed=seed) for i in range(num_envs)]
    vec_env = DummyVecEnv(env_fns)

    if norm_obs or norm_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=float(norm_cfg.get("clip_obs", 10.0)),
            clip_reward=float(norm_cfg.get("clip_reward", 10.0)),
            gamma=float(ppo_cfg.get("gamma", 0.99))
        )
        vec_env.training = True

    # Entorno de Evaluación Periódica
    eval_env_fn = make_env_fn(cfg, rank=999, base_seed=seed + 9999)
    eval_env = DummyVecEnv([eval_env_fn])
    if norm_obs or norm_reward:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=norm_obs,
            norm_reward=False, # Recompensa física sin normalizar en evaluación
            clip_obs=float(norm_cfg.get("clip_obs", 10.0)),
            training=False
        )

    # 3. Configuración de Arquitectura de Red (Sección 12)
    arch_type = str(ppo_cfg.get("net_arch", "64x64_tanh"))
    if arch_type == "256x256_relu":
        policy_kwargs = {
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "activation_fn": nn.ReLU
        }
    else: # 64x64_tanh (por defecto)
        policy_kwargs = {
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": nn.Tanh
        }

    # Exploración log_std_init (X1s)
    if "log_std_init" in ppo_cfg:
        policy_kwargs["log_std_init"] = float(ppo_cfg["log_std_init"])

    # 4. Callbacks
    train_cfg = cfg.get("training", {})
    eval_freq = int(train_cfg.get("eval_freq", 5000))
    save_freq = int(train_cfg.get("save_freq", 50000))

    eval_callback = PhaseAEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, "best_model"),
        log_path=logs_dir,
        eval_freq=eval_freq,
        n_eval_episodes=int(train_cfg.get("n_eval_episodes", 15)),
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(models_dir, "checkpoints"),
        name_prefix=f"ppo_{exp_id}"
    )

    training_callback = PhaseATrainingCallback(window_size=100)

    # 5. Instanciar Agente PPO
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(ppo_cfg.get("learning_rate", 1.5e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 2048)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.002)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
        tensorboard_log=logs_dir
    )

    # 6. Guardar Metadatos del Experimento
    meta_info = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_source": cfg_source,
        "seed": seed,
        "parameters": cfg,
        "effective_policy_kwargs": str(policy_kwargs)
    }
    with open(os.path.join(exp_dir, "experiment_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    # 7. Ejecutar Entrenamiento
    total_timesteps = total_timesteps_override or int(train_cfg.get("total_timesteps", 300000))
    print(f"\n>>> [TRAIN] Iniciando entrenamiento '{exp_id}' | Semilla={seed} | Pasos={total_timesteps:,}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, training_callback],
        progress_bar=False,
        tb_log_name=f"{exp_id}_seed_{seed}"
    )

    # 8. Guardar Modelo Final y Estadísticas de VecNormalize
    final_model_path = os.path.join(models_dir, "ppo_arm2d_final.zip")
    model.save(final_model_path)
    print(f">>> [OK] Modelo final guardado en: {final_model_path}")

    if isinstance(vec_env, VecNormalize):
        norm_stats_path = os.path.join(models_dir, "vecnormalize_stats.pkl")
        vec_env.save(norm_stats_path)
        # Guardar también en best_model si existe
        best_dir = os.path.join(models_dir, "best_model")
        if os.path.exists(best_dir):
            vec_env.save(os.path.join(best_dir, "vecnormalize_stats.pkl"))
        print(f">>> [OK] Estadísticas de normalización guardadas en: {norm_stats_path}")

    vec_env.close()
    eval_env.close()

    # Actualizar automáticamente el archivo EXPERIMENTS_SUMMARY.md
    try:
        from compare_experiments import generate_experiment_summary
        generate_experiment_summary()
    except Exception as e:
        print(f">>> [AVISO] No se pudo actualizar EXPERIMENTS_SUMMARY.md automáticamente: {e}")

    return model, exp_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento PPO Fase A")
    parser.add_argument("--config", "-c", type=str, required=True, help="Ruta al archivo YAML de configuración")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Semilla de aleatoriedad")
    parser.add_argument("--timesteps", "-t", type=int, default=None, help="Sobrescribir timesteps totales")
    args = parser.parse_args()

    train_agent(args.config, seed=args.seed, total_timesteps_override=args.timesteps)
