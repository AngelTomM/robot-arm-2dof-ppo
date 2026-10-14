import os
import sys
import time
import tempfile
import json
from pathlib import Path
import numpy as np
import torch
import yaml

REPO_ROOT = Path("c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo").resolve()
sys.path.insert(0, str(REPO_ROOT))

from arm2d_env import Arm2DEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

def run_benchmark():
    # Cargar config de entorno idéntica a X3/X4 de Fase B (hold relajado 5 pasos, v=0.30)
    env_cfg = {
        "l1": 1.0,
        "l2": 1.0,
        "max_speed": 1.5,
        "target_radius": 0.06,
        "max_steps": 200,
        "dt": 0.05,
        "success_mode": "hold",
        "hold_steps": 5,
        "velocity_tolerance": 0.30,
        "reward_type": "smooth_precision",
        "reward_params": {
            "distance_penalty_coef": 1.0,
            "action_penalty_coef": 0.02,
            "success_bonus": 15.0,
            "near_zone_threshold": 0.30,
            "velocity_penalty_near": 0.15
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        def make_env():
            env = Arm2DEnv(config=env_cfg)
            env.reset(seed=42)
            return Monitor(env)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
            activation_fn=torch.nn.ReLU
        )

        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=300_000,
            learning_starts=2000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            seed=42,
            verbose=0,
            device="cpu"
        )

        steps_to_run = 20_000
        print(f"Iniciando benchmark SAC por {steps_to_run} pasos...")
        t0 = time.perf_counter()
        model.learn(total_timesteps=steps_to_run)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        fps = steps_to_run / elapsed

        print(f"Tiempo: {elapsed:.2f} s")
        print(f"FPS: {fps:.2f}")

        # Guardar un checkpoint para medir tamaño de disco
        ckpt_dir = tmp_path / "ckpt_sample"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(ckpt_dir / "model.zip"))
        vec_env.save(str(ckpt_dir / "vecnormalize_stats.pkl"))

        model_size = (ckpt_dir / "model.zip").stat().st_size
        norm_size = (ckpt_dir / "vecnormalize_stats.pkl").stat().st_size

        # Medir tamaño del replay buffer si se serializara
        buf_file = ckpt_dir / "buffer.pkl"
        model.save_replay_buffer(str(buf_file))
        buf_size_20k = buf_file.stat().st_size

        res = {
            "steps": steps_to_run,
            "elapsed_seconds": elapsed,
            "fps": fps,
            "model_size_bytes": model_size,
            "norm_size_bytes": norm_size,
            "buf_size_20k_bytes": buf_size_20k
        }
        with open("scratch/bench_sac_20k_results.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print("Resultados guardados en scratch/bench_sac_20k_results.json")

if __name__ == "__main__":
    run_benchmark()
