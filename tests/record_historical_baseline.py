"""
Script para registrar la referencia de regresión histórica desde el entorno sin modificar.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import yaml
import numpy as np
from arm2d_env import Arm2DEnv

def record_baseline():
    os.makedirs("tests/data", exist_ok=True)
    
    test_configs = [
        ("default", None),
        ("exp1_baseline", "configs/exp1_baseline.yaml"),
        ("exp2_smooth_precision", "configs/exp2_smooth_precision.yaml")
    ]
    
    seeds = [42, 123, 999]
    records = {}
    
    for cfg_name, cfg_path in test_configs:
        cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8")) if cfg_path else None
        records[cfg_name] = {}
        
        for seed in seeds:
            env = Arm2DEnv(config=cfg)
            obs, info = env.reset(seed=seed)
            
            # Generar acciones deterministas reproducibles
            rng = np.random.RandomState(seed + 1000)
            episode_data = {
                "initial_obs": obs.tolist(),
                "initial_info": {k: float(v) if isinstance(v, (np.floating, float)) else bool(v) if isinstance(v, (np.bool_, bool)) else v for k, v in info.items()},
                "steps": []
            }
            
            for step_idx in range(50):
                action = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                
                step_record = {
                    "step_idx": step_idx,
                    "action": action.tolist(),
                    "obs": next_obs.tolist(),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": {k: float(v) if isinstance(v, (np.floating, float)) else bool(v) if isinstance(v, (np.bool_, bool)) else v for k, v in step_info.items()},
                    "theta1": float(env.theta1),
                    "theta2": float(env.theta2),
                    "dtheta1": float(env.dtheta1),
                    "dtheta2": float(env.dtheta2),
                    "target": env.target.tolist()
                }
                episode_data["steps"].append(step_record)
                if terminated or truncated:
                    break
            
            records[cfg_name][str(seed)] = episode_data
            env.close()

    output_file = "tests/data/historical_regression_reference.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f">>> [OK] Referencia histórica grabada exitosamente en {output_file}")

if __name__ == "__main__":
    record_baseline()
