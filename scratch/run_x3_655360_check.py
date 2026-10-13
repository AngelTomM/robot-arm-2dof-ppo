import sys
from pathlib import Path
import shutil
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_b_train import train_phase_b_agent
from arm2d_env import Arm2DEnv

def main():
    scratch_dir = REPO_ROOT / "scratch" / "check_prefix_655360"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    cfg_path = REPO_ROOT / "configs" / "phase_b" / "X3.yaml"
    
    print(">>> [CHECK 655360] Iniciando entrenamiento de comprobación X3 seed 42 a 655,360 pasos...")
    train_phase_b_agent(
        config_path_or_dict=cfg_path,
        seed=42,
        total_timesteps_override=655360,
        checkpoint_freq_override=40960,
        output_base_dir=str(scratch_dir)
    )
    
    new_ckpt = scratch_dir / "X3" / "seed_42" / "checkpoints" / "ckpt_655360"
    ref_ckpt = REPO_ROOT / "experiments" / "phase_b" / "X3" / "seed_42" / "checkpoints" / "ckpt_655360"
    
    print("\n>>> [CHECK 655360] Comparando nuevo ckpt_655360 contra referencia histórica...")
    new_model = PPO.load(str(new_ckpt / "model.zip"))
    ref_model = PPO.load(str(ref_ckpt / "model.zip"))
    
    max_diff = 0.0
    for (n1, p1), (n2, p2) in zip(new_model.policy.named_parameters(), ref_model.policy.named_parameters()):
        diff = torch.max(torch.abs(p1 - p2)).item()
        if diff > max_diff:
            max_diff = diff
            
    dummy1 = DummyVecEnv([lambda: Arm2DEnv()])
    dummy2 = DummyVecEnv([lambda: Arm2DEnv()])
    new_norm = VecNormalize.load(str(new_ckpt / "vecnormalize_stats.pkl"), dummy1)
    ref_norm = VecNormalize.load(str(ref_ckpt / "vecnormalize_stats.pkl"), dummy2)
    
    obs_mean_diff = float(np.max(np.abs(new_norm.obs_rms.mean - ref_norm.obs_rms.mean)))
    obs_var_diff = float(np.max(np.abs(new_norm.obs_rms.var - ref_norm.obs_rms.var)))
    
    is_exact = (max_diff == 0.0) and (obs_mean_diff == 0.0) and (obs_var_diff == 0.0)
    
    print("=" * 60)
    print(f"Max parameter difference: {max_diff}")
    print(f"Obs mean diff: {obs_mean_diff}")
    print(f"Obs var diff: {obs_var_diff}")
    print(f"EXACT PREFIX MATCH AT 655,360 STEPS: {is_exact}")
    print("=" * 60)
    
    # Escribir resultado a JSON para trazabilidad
    import json
    res = {
        "verified_step": 655360,
        "max_param_diff": max_diff,
        "obs_mean_diff": obs_mean_diff,
        "obs_var_diff": obs_var_diff,
        "is_exact": is_exact,
        "label": "verificado hasta 655360" if is_exact else "no coincide en 655360"
    }
    with open(scratch_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
