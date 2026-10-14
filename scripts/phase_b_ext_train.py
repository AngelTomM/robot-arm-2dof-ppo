"""
Pipeline de Entrenamiento Centralizado para Fase B (Extensión):
- Presupuesto parametrizable: 655.360 timesteps reales (80 rollouts de 8.192 timesteps).
- Checkpoints atómicos periódicos acoplados con VecNormalize (k=5 rollouts = 40.960 timesteps, 16 checkpoints).
- Aislamiento en experiments/phase_b/extension/<experiment>/seed_<seed>/
- Compatible con cualquier configuración YAML de la extensión (X1a, X1b, X1c, X1s, X2, X2b, X3, X4, X5, X5_near, X6).
"""
import os
import sys
from pathlib import Path
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arm2d_env import Arm2DEnv
from src.training.callbacks import PhaseATrainingCallback, PhaseAEvalCallback
from src.training.phase_b_callbacks import PhaseBCheckpointCallback

def set_all_seeds(seed: int):
    """Fija todas las semillas aleatorias para reproducibilidad determinista."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env_fn(cfg: dict, rank: int, base_seed: int):
    """Fábrica de entornos individuales envueltos en Monitor."""
    def _init():
        env = Arm2DEnv(config=cfg)
        env_seed = base_seed + rank * 100
        env.reset(seed=env_seed)
        return Monitor(env)
    return _init

def train_phase_b_ext_agent(config_path_or_dict, seed: int = 42,
                            total_timesteps_override: int = None,
                            checkpoint_freq_override: int = None,
                            output_base_dir: str = "experiments/phase_b/extension"):
    """
    Ejecuta el entrenamiento PPO de Fase B Extensión con guardado atómico de checkpoints.
    """
    # 1. Cargar Configuración
    if isinstance(config_path_or_dict, (str, Path)):
        cfg_path = Path(config_path_or_dict)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg_source = str(cfg_path)
    else:
        cfg = config_path_or_dict
        cfg_source = "dict"

    exp_id = cfg.get("experiment", {}).get("name", "exp_phase_b_ext")
    exp_dir = Path(output_base_dir) / exp_id / f"seed_{seed}"
    models_dir = exp_dir / "models"
    checkpoints_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"

    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

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
            norm_reward=False,
            clip_obs=float(norm_cfg.get("clip_obs", 10.0)),
            training=False
        )

    # 3. Arquitectura de Red y Política
    arch_type = str(ppo_cfg.get("net_arch", "64x64_tanh"))
    if arch_type == "256x256_relu":
        policy_kwargs = {
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "activation_fn": nn.ReLU
        }
    else:
        policy_kwargs = {
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": nn.Tanh
        }

    # Soporte para log_std_init (X1s)
    if "log_std_init" in ppo_cfg and ppo_cfg["log_std_init"] is not None:
        policy_kwargs["log_std_init"] = float(ppo_cfg["log_std_init"])

    # 4. Callbacks
    train_cfg = cfg.get("training", {})
    eval_freq = int(train_cfg.get("eval_freq", 5000))
    n_eval_episodes = int(train_cfg.get("n_eval_episodes", 15))
    ckpt_freq = checkpoint_freq_override or int(train_cfg.get("checkpoint_freq", 40960))

    eval_callback = PhaseAEvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "best_model"),
        log_path=str(logs_dir),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )

    training_callback = PhaseATrainingCallback(window_size=100)
    checkpoint_callback = PhaseBCheckpointCallback(
        save_freq=ckpt_freq,
        save_path=str(checkpoints_dir),
        verbose=1
    )

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
        tensorboard_log=str(logs_dir)
    )

    # 6. Metadatos del Experimento
    meta_info = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_source": cfg_source,
        "seed": seed,
        "parameters": cfg,
        "effective_policy_kwargs": str(policy_kwargs),
        "phase": "B_extension"
    }
    with open(exp_dir / "experiment_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    # 7. Ejecutar Entrenamiento
    total_timesteps = total_timesteps_override or int(train_cfg.get("total_timesteps", 655360))
    print(f"\n>>> [PHASE B EXT TRAIN] Iniciando '{exp_id}' | Semilla={seed} | Presupuesto={total_timesteps:,} pasos")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, training_callback, checkpoint_callback],
        progress_bar=True,
        tb_log_name=f"{exp_id}_seed_{seed}"
    )

    # 8. Guardar Modelo Final y Estadísticas de VecNormalize
    final_model_path = models_dir / "ppo_arm2d_final.zip"
    model.save(str(final_model_path))
    if isinstance(vec_env, VecNormalize):
        norm_stats_path = models_dir / "vecnormalize_stats.pkl"
        vec_env.save(str(norm_stats_path))
        best_dir = models_dir / "best_model"
        if best_dir.exists():
            vec_env.save(str(best_dir / "vecnormalize_stats.pkl"))

    vec_env.close()
    eval_env.close()

    print(f">>> [PHASE B EXT OK] Entrenamiento concluido: {exp_dir}")
    print(f">>> [PHASE B EXT OK] Checkpoints guardados: {len(checkpoint_callback.saved_checkpoints)}")
    return model, exp_dir, checkpoint_callback.saved_checkpoints

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento Fase B Extensión")
    parser.add_argument("--config", type=str, required=True, help="Ruta al YAML en configs/phase_b/extension/")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Override de timesteps")
    parser.add_argument("--checkpoint_freq", type=int, default=None, help="Override de frecuencia de checkpoints")
    parser.add_argument("--output_base_dir", type=str, default="experiments/phase_b/extension", help="Directorio base")
    args = parser.parse_args()

    train_phase_b_ext_agent(
        config_path_or_dict=args.config,
        seed=args.seed,
        total_timesteps_override=args.total_timesteps,
        checkpoint_freq_override=args.checkpoint_freq,
        output_base_dir=args.output_base_dir
    )
