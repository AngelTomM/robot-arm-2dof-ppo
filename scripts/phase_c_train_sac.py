"""
Pipeline de Entrenamiento Centralizado para Fase C (SAC S1):
- Algoritmo: Soft Actor-Critic (SAC, Stable-Baselines3)
- Presupuesto: 655.360 timesteps reales.
- Checkpoints atómicos periódicos acoplados con VecNormalize (k=40.960 timesteps, 16 checkpoints).
- Aislamiento en experiments/phase_c/S1/seed_<seed>/
- Telemetría: logs de TensorBoard, Monitor, experiment_meta.json y train_metrics.json.
"""
import os
import sys
from pathlib import Path
import json
import yaml
import random
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arm2d_env import Arm2DEnv
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

def train_phase_c_sac_agent(config_path_or_dict, seed: int = 42,
                            total_timesteps_override: int = None,
                            checkpoint_freq_override: int = None,
                            output_base_dir: str = "experiments/phase_c"):
    """
    Ejecuta el entrenamiento SAC de Fase C con guardado atómico de checkpoints.
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

    exp_id = cfg.get("experiment", {}).get("name", "S1")
    exp_dir = Path(output_base_dir) / exp_id / f"seed_{seed}"
    models_dir = exp_dir / "models"
    checkpoints_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"

    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(seed)

    # 2. Configurar Entorno con Monitor y VecNormalize
    sac_cfg = cfg.get("sac", {})
    norm_cfg = cfg.get("normalization", {})
    norm_obs = bool(norm_cfg.get("norm_obs", True))
    norm_reward = bool(norm_cfg.get("norm_reward", False))
    num_envs = int(sac_cfg.get("num_envs", 1))

    env_fns = [make_env_fn(cfg, rank=i, base_seed=seed) for i in range(num_envs)]
    vec_env = DummyVecEnv(env_fns)

    if norm_obs or norm_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=float(norm_cfg.get("clip_obs", 10.0)),
            gamma=float(sac_cfg.get("gamma", 0.99))
        )
        vec_env.training = True

    # 3. Arquitectura de Red y Política
    arch_type = str(sac_cfg.get("net_arch", "256x256_relu"))
    if arch_type == "256x256_relu":
        policy_kwargs = {
            "net_arch": {"pi": [256, 256], "qf": [256, 256]},
            "activation_fn": nn.ReLU
        }
    else:
        policy_kwargs = {
            "net_arch": {"pi": [64, 64], "qf": [64, 64]},
            "activation_fn": nn.ReLU
        }

    # 4. Callbacks de Checkpoint Atómico
    train_cfg = cfg.get("training", {})
    ckpt_freq = checkpoint_freq_override or int(train_cfg.get("checkpoint_freq", 40960))
    checkpoint_callback = PhaseBCheckpointCallback(
        save_freq=ckpt_freq,
        save_path=str(checkpoints_dir),
        verbose=1
    )

    # 5. Instanciar SAC
    total_timesteps = total_timesteps_override or int(train_cfg.get("total_timesteps", 655360))

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(sac_cfg.get("learning_rate", 3e-4)),
        buffer_size=int(sac_cfg.get("buffer_size", 300000)),
        learning_starts=int(sac_cfg.get("learning_starts", 5000)),
        batch_size=int(sac_cfg.get("batch_size", 256)),
        tau=float(sac_cfg.get("tau", 0.005)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        train_freq=int(sac_cfg.get("train_freq", 1)),
        gradient_steps=int(sac_cfg.get("gradient_steps", 1)),
        ent_coef=sac_cfg.get("ent_coef", "auto"),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(logs_dir),
        seed=seed,
        verbose=1,
        device="cpu"
    )

    # 6. Guardar Metadatos del Experimento
    meta_info = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_name": exp_id,
        "seed": seed,
        "config_source": cfg_source,
        "algorithm": "SAC",
        "total_timesteps_planned": total_timesteps,
        "checkpoint_freq": ckpt_freq,
        "parameters": cfg
    }
    with open(exp_dir / "experiment_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4)

    # 7. Ejecutar Entrenamiento
    print(f"\n{'='*75}")
    print(f" INICIANDO ENTRENAMIENTO SAC: {exp_id} | SEMILLA: {seed}")
    print(f" Presupuesto: {total_timesteps:,} pasos | Checkpoint cada: {ckpt_freq:,} pasos")
    print(f" Directorio de salida: {exp_dir}")
    print(f"{'='*75}\n")

    t_start = time.perf_counter()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback],
        progress_bar=True,
        reset_num_timesteps=True
    )
    t_end = time.perf_counter()
    elapsed_sec = t_end - t_start
    fps = total_timesteps / elapsed_sec if elapsed_sec > 0 else 0.0

    # 8. Guardar Modelo Final y Estadísticas
    final_model_path = models_dir / "sac_arm2d_final.zip"
    model.save(str(final_model_path))
    if norm_obs or norm_reward:
        final_norm_path = models_dir / "vecnormalize_stats.pkl"
        vec_env.save(str(final_norm_path))

    # Guardar métricas de tiempo y rendimiento
    runtime_metrics = {
        "experiment_name": exp_id,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "elapsed_seconds": elapsed_sec,
        "elapsed_minutes": elapsed_sec / 60.0,
        "elapsed_hours": elapsed_sec / 3600.0,
        "mean_fps": fps,
        "saved_checkpoints_count": len(checkpoint_callback.saved_checkpoints),
        "saved_checkpoints": checkpoint_callback.saved_checkpoints
    }
    with open(exp_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(runtime_metrics, f, indent=4)

    print(f"\n{'='*75}")
    print(f" ENTRENAMIENTO COMPLETADO: {exp_id} | SEMILLA: {seed}")
    print(f" Tiempo: {elapsed_sec:.2f} s ({elapsed_sec/60:.2f} min) | FPS: {fps:.2f}")
    print(f" Checkpoints guardados: {len(checkpoint_callback.saved_checkpoints)}")
    print(f"{'='*75}\n")

    return runtime_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento SAC Fase C")
    parser.add_argument("--config", type=str, default="configs/phase_c/S1.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--ckpt-freq", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/phase_c")
    args = parser.parse_args()

    train_phase_c_sac_agent(
        config_path_or_dict=args.config,
        seed=args.seed,
        total_timesteps_override=args.timesteps,
        checkpoint_freq_override=args.ckpt_freq,
        output_base_dir=args.output_dir
    )
