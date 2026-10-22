"""
Entrenamiento (Brazo Robótico 2 DOF).

Fuente Única de Verdad (SSOT) de la fase y el experimento: la RUTA del YAML.
  configs/phase_b/X3.yaml           ->  experiments/phase_b/X3/seed_<seed>/
  configs/phase_b/extension/X3.yaml ->  experiments/phase_b_ext/X3/seed_<seed>/
  configs/phase_c/S1.yaml           ->  experiments/phase_c/S1/seed_<seed>/

Soporte algorítmico dual vía Stable-Baselines3:
  - Sección 'sac:' en el YAML      -> SAC
  - Sección 'ppo:' en el YAML      -> PPO

Uso:
  python train.py --config configs/phase_b/X3.yaml --seed 42 --device auto
"""

import argparse
import datetime
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arm2d_env import Arm2DEnv  # noqa: E402
from src.training.callbacks import LQRMetricsCallback, VecNormalizeSyncEvalCallback  # noqa: E402
from src.training.phase_b_callbacks import PhaseBCheckpointCallback  # noqa: E402


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
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
        env.reset(seed=base_seed + rank * 100)
        return Monitor(env)
    return _init


# Funcion arquitectura de red
def parse_net_arch(spec, default_activation=nn.ReLU) -> dict:

    ACTIVATION_MAP = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "selu": nn.SELU,
        "sigmoid": nn.Sigmoid,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "identity": nn.Identity,
        "linear": nn.Identity,
        "none": nn.Identity,
    }

    if isinstance(spec, dict):
        layers = spec.get("layers", [256, 256])
        if not isinstance(layers, list) or not all(isinstance(x, int) and x > 0 for x in layers):
            layers = [256, 256]

        act_name = str(spec.get("hidden_activation", "relu")).strip().lower()
        activation_fn = ACTIVATION_MAP.get(act_name, default_activation)

        policy_kwargs = {
            "net_arch": layers,
            "activation_fn": activation_fn,
        }
        return policy_kwargs

    # Compatibilidad hacia atrás con especificaciones tipo cadena (ej. '256x256_relu')
    spec_str = str(spec or "256x256_relu").strip().lower()
    match = re.match(r"^(\d+)x(\d+)", spec_str)
    sizes = [int(match.group(1)), int(match.group(2))] if match else [256, 256]
    activation = nn.Tanh if "tanh" in spec_str else default_activation
    return {"net_arch": sizes, "activation_fn": activation}


def resolve_phase(config_path: Path) -> str:
    """
    SSOT: la fase es la ruta del config relativa al directorio 'configs/',
    excluyendo el nombre del archivo y aplanando subdirectorios.
      configs/phase_b/X3.yaml           -> phase_b
      configs/phase_b/extension/X3.yaml -> phase_b_ext
      configs/phase_c/S1.yaml           -> phase_c
    """
    configs_root = (REPO_ROOT / "configs").resolve()
    try:
        rel = config_path.resolve().relative_to(configs_root)
    except ValueError:
        # YAML fuera de configs/: usar el nombre del directorio contenedor.
        return config_path.resolve().parent.name or "misc"
    parts = rel.parts[:-1]
    if not parts:
        return "misc"
    return "_".join("ext" if p == "extension" else p for p in parts)


def select_algorithm(cfg: dict) -> str:
    """Selecciona el algoritmo según las secciones del YAML (dual PPO/SAC)."""
    has_ppo = "ppo" in cfg
    has_sac = "sac" in cfg
    if has_ppo and has_sac:
        raise ValueError("El YAML define secciones 'ppo' y 'sac' a la vez; deja solo una.")
    if has_sac:
        return "SAC"
    if has_ppo:
        return "PPO"
    raise ValueError("El YAML no define la sección del algoritmo ('ppo:' o 'sac:').")


# ----------------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------------

def run_training(cfg: dict, config_path: Path, seed: int, device: str,
                 total_timesteps_override=None,
                 checkpoint_freq_override=None) -> Path:
    """Ejecuta el entrenamiento PPO/SAC y devuelve el directorio de salida."""
    algo = select_algorithm(cfg)
    algo_cfg = cfg.get("sac" if algo == "SAC" else "ppo", {})

    exp_id = str(cfg.get("experiment", {}).get("name") or config_path.stem)
    phase = resolve_phase(config_path)

    # Directorio de salida estricto: experiments/<phase>/<exp_id>/seed_<seed>/
    exp_dir = REPO_ROOT / "experiments" / Path(phase) / exp_id / f"seed_{seed}"
    models_dir = exp_dir / "models"
    checkpoints_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"
    for d in (models_dir, checkpoints_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    set_all_seeds(seed)

    # 1. Entornos vectorizados con Monitor y VecNormalize opcional
    norm_cfg = cfg.get("normalization", {})
    norm_obs = bool(norm_cfg.get("norm_obs", False))
    norm_reward = bool(norm_cfg.get("norm_reward", False))
    clip_obs = float(norm_cfg.get("clip_obs", 10.0))
    gamma = float(algo_cfg.get("gamma", 0.99))
    num_envs = int(algo_cfg.get("num_envs", 4 if algo == "PPO" else 1))

    vec_env = DummyVecEnv([make_env_fn(cfg, rank=i, base_seed=seed) for i in range(num_envs)])
    if norm_obs or norm_reward:
        vec_env = VecNormalize(vec_env, norm_obs=norm_obs, norm_reward=norm_reward,
                               clip_obs=clip_obs, gamma=gamma)
        vec_env.training = True

    # 2. Callbacks
    train_cfg = cfg.get("training", {})
    ckpt_freq = int(checkpoint_freq_override or train_cfg.get("checkpoint_freq", 40960))
    checkpoint_cb = PhaseBCheckpointCallback(save_freq=ckpt_freq,
                                             save_path=str(checkpoints_dir), verbose=1)

    callbacks = [LQRMetricsCallback(window_size=100)]
    eval_freq = int(train_cfg.get("eval_freq", 0))
    if eval_freq > 0:
        # Entorno de evaluación: VecNormalize en modo inference
        # (training=False, norm_reward=False; sincronizado por VecNormalizeSyncEvalCallback).
        eval_env = DummyVecEnv([make_env_fn(cfg, rank=999, base_seed=seed + 9999)])
        if norm_obs or norm_reward:
            eval_env = VecNormalize(eval_env, norm_obs=norm_obs, norm_reward=False,
                                    clip_obs=clip_obs, training=False)
        callbacks.append(VecNormalizeSyncEvalCallback(
            eval_env,
            best_model_save_path=str(models_dir / "best_model"),
            log_path=str(logs_dir),
            eval_freq=eval_freq,
            n_eval_episodes=int(train_cfg.get("n_eval_episodes", 15)),
            deterministic=True, render=False,
        ))
    callbacks.append(checkpoint_cb)

    # 3. Instanciar el agente
    common_kwargs = dict(
        policy="MlpPolicy", env=vec_env, seed=seed, verbose=1,
        tensorboard_log=str(logs_dir), device=device,
        gamma=gamma, policy_kwargs=parse_net_arch(algo_cfg.get("net_arch", "256x256_relu")),
    )

    budget = int(total_timesteps_override or train_cfg.get("total_timesteps", 1_000_000))

    if algo == "PPO":
        model = PPO(
            learning_rate=float(algo_cfg.get("learning_rate", 1.5e-4)),
            n_steps=int(algo_cfg.get("n_steps", 2048)),
            batch_size=int(algo_cfg.get("batch_size", 64)),
            n_epochs=int(algo_cfg.get("n_epochs", 10)),
            gae_lambda=float(algo_cfg.get("gae_lambda", 0.95)),
            clip_range=float(algo_cfg.get("clip_range", 0.2)),
            ent_coef=float(algo_cfg.get("ent_coef", 0.002)),
            max_grad_norm=float(algo_cfg.get("max_grad_norm", 0.5)),
            **common_kwargs,
        )
        final_name = "ppo_arm2d_final.zip"
    else:
        target_entropy_raw = algo_cfg.get("target_entropy", "auto")
        target_entropy = ("auto" if str(target_entropy_raw).lower() == "auto"
                          else float(target_entropy_raw))
        model = SAC(
            learning_rate=float(algo_cfg.get("learning_rate", 3e-4)),
            buffer_size=int(algo_cfg.get("buffer_size", 300_000)),
            learning_starts=int(algo_cfg.get("learning_starts", 5_000)),
            batch_size=int(algo_cfg.get("batch_size", 256)),
            tau=float(algo_cfg.get("tau", 0.005)),
            train_freq=int(algo_cfg.get("train_freq", 1)),
            gradient_steps=int(algo_cfg.get("gradient_steps", 1)),
            ent_coef=str(algo_cfg.get("ent_coef", "auto")),
            target_entropy=target_entropy,
            **common_kwargs,
        )
        final_name = "sac_arm2d_final.zip"

    # 4. Metadatos del experimento
    meta = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_source": str(config_path),
        "phase": phase,
        "experiment_name": exp_id,
        "algorithm": algo,
        "seed": seed,
        "device": device,
        "total_timesteps_planned": budget,
        "checkpoint_freq": ckpt_freq,
        "parameters": cfg,
    }
    with open(exp_dir / "experiment_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 5. Entrenamiento
    print(f"\n>>> [TRAIN] {algo} | fase={phase} | exp={exp_id} | seed={seed} "
          f"| device={device} | budget={budget:,} pasos")
    print(f">>> [TRAIN] Salida: {exp_dir}\n")

    t0 = time.perf_counter()
    model.learn(total_timesteps=budget, callback=callbacks, progress_bar=True,
                tb_log_name=f"{exp_id}_seed_{seed}")
    elapsed = time.perf_counter() - t0

    # 6. Guardar modelo final y estadísticas de normalización acopladas
    model.save(str(models_dir / final_name))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(models_dir / "vecnormalize_stats.pkl"))
        best_dir = models_dir / "best_model"
        if best_dir.exists():
            vec_env.save(str(best_dir / "vecnormalize_stats.pkl"))
    vec_env.close()
    if eval_freq > 0:
        eval_env.close()

    with open(exp_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment_name": exp_id,
            "algorithm": algo,
            "seed": seed,
            "total_timesteps": budget,
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60.0,
            "mean_fps": budget / elapsed if elapsed > 0 else 0.0,
            "saved_checkpoints": checkpoint_cb.saved_checkpoints,
        }, f, indent=2)

    print(f"\n>>> [TRAIN OK] {algo} '{exp_id}' completado en {elapsed:.1f} s "
          f"({elapsed / 60.0:.2f} min) | Checkpoints: {len(checkpoint_cb.saved_checkpoints)}")
    print(f">>> [TRAIN OK] Salida: {exp_dir}")
    return exp_dir


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento unificado PPO/SAC (Brazo Robótico 2 DOF)")
    parser.add_argument("--config", type=str, required=True,
                        help="Ruta al YAML del experimento (p. ej. configs/phase_b/X3.yaml)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--device", type=str, default="auto",
                        help="Dispositivo de cómputo: auto, cpu, cuda, ...")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Override del presupuesto de timesteps")
    parser.add_argument("--checkpoint_freq", type=int, default=None,
                        help="Override de la frecuencia de checkpoints")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        parser.error(f"No existe el archivo de configuración: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_training(cfg, config_path, seed=args.seed, device=args.device,
                 total_timesteps_override=args.total_timesteps,
                 checkpoint_freq_override=args.checkpoint_freq)


if __name__ == "__main__":
    main()
