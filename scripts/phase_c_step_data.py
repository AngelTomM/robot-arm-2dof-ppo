"""
Extractor de Telemetría Paso a Paso para Episodios Geométricos en Validación C:
- Lee los 4 episodios geométricos a priori de geometric_episodes_selection.json.
- Extrae telemetría cartesiana y articular (paso a paso) para S1, X3 y C0.
- Exporta reports/data/phase_c/step_data_val_geometry.csv.
"""
import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from stable_baselines3 import SAC, PPO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arm2d_env import Arm2DEnv
from src.controllers.jacobian_controller import JacobianDampedController

def collect_step_telemetry(agent, target_episodes: list, env_cfg: dict,
                           eval_mode: str = "deterministic",
                           vec_normalize_path: str = None) -> pd.DataFrame:
    """
    Ejecuta el agente registrando cada paso individual sobre los episodios seleccionados.
    """
    merged_cfg = env_cfg.copy() if env_cfg else {}
    if "env" not in merged_cfg:
        merged_cfg["env"] = {}
    merged_cfg["env"]["success_mode"] = "hold"
    merged_cfg["env"]["target_radius"] = 0.06
    merged_cfg["env"]["velocity_tolerance"] = 0.10
    merged_cfg["env"]["hold_steps"] = 20
    merged_cfg["env"]["max_steps"] = 200
    merged_cfg["env"]["dt"] = 0.05

    env = Arm2DEnv(config=merged_cfg)

    norm_wrapper = None
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        dummy = DummyVecEnv([lambda: Arm2DEnv(config=merged_cfg)])
        norm_wrapper = VecNormalize.load(vec_normalize_path, dummy)
        norm_wrapper.training = False
        norm_wrapper.norm_reward = False

    step_records = []

    for ep_meta in target_episodes:
        ep_id = ep_meta["episode_id"]
        th1_init = ep_meta["theta1_init"]
        th2_init = ep_meta["theta2_init"]
        tgt_x = ep_meta["target_x"]
        tgt_y = ep_meta["target_y"]
        tgt_r = ep_meta.get("target_r", np.nan)

        env.reset()
        env.theta1 = float(th1_init)
        env.theta2 = float(th2_init)
        env.dtheta1 = 0.0
        env.dtheta2 = 0.0
        env.target = np.array([tgt_x, tgt_y], dtype=np.float32)
        env.step_count = 0
        env.hold_counter = 0
        env.hold_counter_max = 0
        env.prev_action = None
        env.has_prev_action = False
        env.first_entry_step = None
        env.joint_speed_at_first_entry = None
        env.ee_speed_at_first_entry = None
        env.n_exits = 0
        env.n_cross = 0
        env.time_inside_near = 0
        env.was_inside_target = False
        env.prev_error_vec = None

        obs = env._get_obs()
        steps = 0

        # Registro del estado inicial (paso 0)
        p1, ee = env._forward_kinematics(env.theta1, env.theta2)
        dist_0 = float(np.linalg.norm(ee - env.target))
        step_records.append({
            "episode_id": ep_id,
            "target_r": tgt_r,
            "target_x": tgt_x,
            "target_y": tgt_y,
            "step": 0,
            "theta1": float(env.theta1),
            "theta2": float(env.theta2),
            "dtheta1": 0.0,
            "dtheta2": 0.0,
            "joint_speed": 0.0,
            "ee_x": float(ee[0]),
            "ee_y": float(ee[1]),
            "ee_speed": 0.0,
            "distance": dist_0,
            "distance_rate": 0.0,
            "hold_counter": 0,
            "is_success": False,
            "action_0": 0.0,
            "action_1": 0.0,
            "delta_action_norm": 0.0,
            "reward": 0.0,
            "eval_mode": eval_mode
        })

        prev_dist = dist_0

        while True:
            steps += 1
            if norm_wrapper is not None:
                policy_obs = norm_wrapper.normalize_obs(obs)
            else:
                policy_obs = obs

            if hasattr(agent, "predict"):
                if hasattr(agent, "policy"):
                    det = (eval_mode == "deterministic")
                    action, _ = agent.predict(policy_obs, deterministic=det)
                else:
                    action = agent.predict(env)
            else:
                raise ValueError("Agente incompatible")

            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs

            p1, ee = env._forward_kinematics(env.theta1, env.theta2)
            curr_dist = info["distance"]
            dist_rate = (curr_dist - prev_dist) / env.dt
            prev_dist = curr_dist

            step_records.append({
                "episode_id": ep_id,
                "target_r": tgt_r,
                "target_x": tgt_x,
                "target_y": tgt_y,
                "step": steps,
                "theta1": float(env.theta1),
                "theta2": float(env.theta2),
                "dtheta1": float(env.dtheta1),
                "dtheta2": float(env.dtheta2),
                "joint_speed": float(info["joint_speed"]),
                "ee_x": float(ee[0]),
                "ee_y": float(ee[1]),
                "ee_speed": float(info["ee_speed"]),
                "distance": curr_dist,
                "distance_rate": dist_rate,
                "hold_counter": int(info["hold_counter"]),
                "is_success": bool(info["is_success"]),
                "action_0": float(action[0]),
                "action_1": float(action[1]),
                "delta_action_norm": float(info["delta_action_norm"]) if info["delta_action_norm"] is not None else 0.0,
                "reward": float(reward),
                "eval_mode": eval_mode
            })

            if terminated or truncated:
                break

    return pd.DataFrame(step_records)

def extract_all_geometric_telemetry():
    c_data_dir = REPO_ROOT / "data" / "datasets" / "phase_c"
    reports_data_dir = REPO_ROOT / "reports" / "data" / "phase_c"
    reports_data_dir.mkdir(parents=True, exist_ok=True)

    val_path = c_data_dir / "validation_c_500.json"
    geom_meta_path = reports_data_dir / "geometric_episodes_selection.json"

    with open(val_path, "r", encoding="utf-8") as f:
        val_dataset = json.load(f)
    with open(geom_meta_path, "r", encoding="utf-8") as f:
        geom_meta = json.load(f)

    target_ids = [item["episode_id"] for item in geom_meta["episodes"]]
    canonical_eps = [ep for ep in val_dataset if ep["episode_id"] in target_ids]

    env_cfg = {
        "env": {
            "max_speed": 1.5,
            "success_mode": "hold",
            "hold_steps": 20,
            "velocity_tolerance": 0.10,
            "target_radius": 0.06
        }
    }

    all_dfs = []

    # 1. C0 Analítico
    print(">>> Extrayendo telemetría para C0...")
    c0 = JacobianDampedController(kp=3.0, lambda_damping=0.01, max_speed=1.5)
    df_c0 = collect_step_telemetry(c0, canonical_eps, env_cfg, eval_mode="deterministic")
    df_c0["agent"] = "C0"
    df_c0["seed"] = "analytic"
    all_dfs.append(df_c0)

    # 2. X3 (PPO) - Checkpoint final 655360 (seed 42)
    x3_ckpt = REPO_ROOT / "experiments" / "phase_b" / "X3" / "seed_42" / "checkpoints" / "ckpt_655360"
    if x3_ckpt.exists():
        print(">>> Extrayendo telemetría para X3 (seed 42, deterministic & stochastic)...")
        m_x3 = PPO.load(str(x3_ckpt / "model.zip"))
        n_x3 = str(x3_ckpt / "vecnormalize_stats.pkl")

        df_x3_det = collect_step_telemetry(m_x3, canonical_eps, env_cfg, eval_mode="deterministic", vec_normalize_path=n_x3)
        df_x3_det["agent"] = "X3"
        df_x3_det["seed"] = 42
        all_dfs.append(df_x3_det)

        df_x3_sto = collect_step_telemetry(m_x3, canonical_eps, env_cfg, eval_mode="stochastic", vec_normalize_path=n_x3)
        df_x3_sto["agent"] = "X3"
        df_x3_sto["seed"] = 42
        all_dfs.append(df_x3_sto)

    # 3. S1 (SAC) - Checkpoint final 655360 (seed 42) si existe
    s1_ckpt = REPO_ROOT / "experiments" / "phase_c" / "S1" / "seed_42" / "checkpoints" / "ckpt_655360"
    if s1_ckpt.exists():
        print(">>> Extrayendo telemetría para S1 (seed 42, deterministic & stochastic)...")
        m_s1 = SAC.load(str(s1_ckpt / "model.zip"))
        n_s1 = str(s1_ckpt / "vecnormalize_stats.pkl")

        df_s1_det = collect_step_telemetry(m_s1, canonical_eps, env_cfg, eval_mode="deterministic", vec_normalize_path=n_s1)
        df_s1_det["agent"] = "S1"
        df_s1_det["seed"] = 42
        all_dfs.append(df_s1_det)

        df_s1_sto = collect_step_telemetry(m_s1, canonical_eps, env_cfg, eval_mode="stochastic", vec_normalize_path=n_s1)
        df_s1_sto["agent"] = "S1"
        df_s1_sto["seed"] = 42
        all_dfs.append(df_s1_sto)

    final_df = pd.concat(all_dfs, ignore_index=True)
    out_csv = reports_data_dir / "step_data_val_geometry.csv"
    final_df.to_csv(out_csv, index=False)
    print(f">>> [TELEMETRÍA] Telemetría exportada a: {out_csv} ({len(final_df)} registros)")

if __name__ == "__main__":
    extract_all_geometric_telemetry()
