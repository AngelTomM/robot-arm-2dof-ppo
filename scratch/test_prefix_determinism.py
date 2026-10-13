import sys
from pathlib import Path
import tempfile
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_b_train import train_phase_b_agent

def verify_step_determinism():
    print("Testing prefix determinism on X3 seed 42...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg_path = REPO_ROOT / "configs/phase_b/X3.yaml"
        
        # Train for 40,960 steps (1 checkpoint) with total_timesteps override = 40960
        train_phase_b_agent(
            config_path_or_dict=cfg_path,
            seed=42,
            total_timesteps_override=40960,
            checkpoint_freq_override=40960,
            output_base_dir=str(tmp_path)
        )
        
        new_ckpt = tmp_path / "X3" / "seed_42" / "checkpoints" / "ckpt_40960"
        ref_ckpt = REPO_ROOT / "experiments" / "phase_b" / "X3" / "seed_42" / "checkpoints" / "ckpt_40960"
        
        new_model = PPO.load(str(new_ckpt / "model.zip"))
        ref_model = PPO.load(str(ref_ckpt / "model.zip"))
        
        # Compare model weights
        max_diff = 0.0
        for p1, p2 in zip(new_model.policy.parameters(), ref_model.policy.parameters()):
            diff = torch.max(torch.abs(p1 - p2)).item()
            if diff > max_diff:
                max_diff = diff
        
        print(f"Max parameter difference at step 40,960: {max_diff}")
        
        # Compare VecNormalize stats
        from arm2d_env import Arm2DEnv
        dummy1 = DummyVecEnv([lambda: Arm2DEnv()])
        dummy2 = DummyVecEnv([lambda: Arm2DEnv()])
        new_norm = VecNormalize.load(str(new_ckpt / "vecnormalize_stats.pkl"), dummy1)
        ref_norm = VecNormalize.load(str(ref_ckpt / "vecnormalize_stats.pkl"), dummy2)
        
        obs_mean_diff = np.max(np.abs(new_norm.obs_rms.mean - ref_norm.obs_rms.mean))
        obs_var_diff = np.max(np.abs(new_norm.obs_rms.var - ref_norm.obs_rms.var))
        print(f"Obs mean diff: {obs_mean_diff}, Obs var diff: {obs_var_diff}")
        
        is_exact = (max_diff == 0.0) and (obs_mean_diff == 0.0) and (obs_var_diff == 0.0)
        print(f"EXACT PREFIX MATCH: {is_exact}")
        return is_exact

if __name__ == "__main__":
    verify_step_determinism()
