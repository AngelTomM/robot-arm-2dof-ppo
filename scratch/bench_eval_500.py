import time
import json
from pathlib import Path
import sys

REPO_ROOT = Path("c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo").resolve()
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import evaluate_on_dataset
from stable_baselines3 import PPO

val_dataset_path = REPO_ROOT / "data" / "datasets" / "phase_b_ext" / "validation_b2_500.json"
with open(val_dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Probar con checkpoint existente de X3
ckpt_path = REPO_ROOT / "experiments" / "phase_b" / "X3" / "seed_42" / "checkpoints" / "ckpt_409600"
model = PPO.load(str(ckpt_path / "model.zip"))
norm_path = str(ckpt_path / "vecnormalize_stats.pkl")

env_cfg = {
    "env": {
        "max_speed": 1.5,
        "success_mode": "hold",
        "hold_steps": 20,
        "velocity_tolerance": 0.10,
        "target_radius": 0.06
    }
}

print(f"Evaluando 500 episodios estrictos...")
t0 = time.perf_counter()
df, metrics = evaluate_on_dataset(
    agent=model,
    dataset=dataset,
    dataset_name="val_bench",
    eval_mode="deterministic",
    env_config=env_cfg,
    vec_normalize_path=norm_path
)
t1 = time.perf_counter()
elapsed = t1 - t0
print(f"Tiempo evaluando 500 episodios: {elapsed:.2f} s")
print(f"Tiempo por episodio: {elapsed / len(dataset) * 1000:.2f} ms")
print(f"Éxito estricto medido: {metrics['success_rate']:.4f}")

with open("scratch/bench_eval_500_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "n_episodes": len(dataset),
        "elapsed_seconds": elapsed,
        "ms_per_episode": elapsed / len(dataset) * 1000,
        "success_rate": metrics["success_rate"]
    }, f, indent=2)
