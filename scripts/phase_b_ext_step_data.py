"""
Extractor de Telemetría Paso a Paso para Episodios Geométricos Fijos en Validación B2:
- Episodios fijos fijados a priori: Q10 (311), Q50 (85), Q90 (167), EDGE (480).
- Se extrae exclusivamente sobre validation_b2_500.json.
- Final B2 permanece estrictamente sellado.
- Exporta reports/data/phase_b/extension/step_data_val_geometry.csv.
"""
import os
import sys
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arm2d_env import Arm2DEnv
from src.controllers.jacobian_controller import JacobianDampedController

CANONICAL_EPISODES = [311, 85, 167, 480]

def collect_step_telemetry(agent, target_episodes: list, env_cfg: dict,
                           eval_mode: str = "deterministic",
                           vec_normalize_path: str = None) -> pd.DataFrame:
    """
    Ejecuta el agente registrando cada paso individual sobre los episodios geométricos.
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

        obs = env.set_state_and_target(th1_init, th2_init, tgt_x, tgt_y)
        steps = 0

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
                raise ValueError("Agente sin método predict()")

            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs

            _, ee_pos = env._forward_kinematics(env.theta1, env.theta2)

            step_records.append({
                "episode_id": ep_id,
                "target_r": tgt_r,
                "step": steps,
                "distance": float(info["distance"]),
                "joint_speed": float(info["joint_speed"]),
                "ee_speed": float(info["ee_speed"]),
                "action_1": float(action[0]),
                "action_2": float(action[1]),
                "delta_action_norm": float(info["delta_action_norm"]) if info.get("delta_action_norm") is not None else 0.0,
                "hold_counter": int(info["hold_counter"]),
                "is_success": bool(info["is_success"]),
                "ee_x": float(ee_pos[0]),
                "ee_y": float(ee_pos[1]),
                "target_x": float(tgt_x),
                "target_y": float(tgt_y)
            })

            if terminated or truncated:
                break

    return pd.DataFrame(step_records)

def run_collect_all_step_data(dataset_path: str = "data/datasets/phase_b_ext/validation_b2_500.json",
                              selected_ckpts_json: str = "reports/data/phase_b/extension/selected_checkpoints.json",
                              output_csv: str = "reports/data/phase_b/extension/step_data_val_geometry.csv"):
    with open(REPO_ROOT / dataset_path, "r", encoding="utf-8") as f:
        val_dataset = json.load(f)

    target_episodes = [ep for ep in val_dataset if ep["episode_id"] in CANONICAL_EPISODES]
    if len(target_episodes) != 4:
        raise ValueError(f"No se encontraron los 4 episodios canónicos en {dataset_path}")

    with open(REPO_ROOT / selected_ckpts_json, "r", encoding="utf-8") as f:
        selected_ckpts = json.load(f)

    all_dfs = []

    # 1. Recolectar C0 Consistente y C0 Histórico
    ref_cfg_path = REPO_ROOT / "configs/phase_b/extension/X3.yaml"
    with open(ref_cfg_path, "r", encoding="utf-8") as f:
        ref_env_cfg = yaml.safe_load(f)

    print(">>> Recolectando telemetría C0 Consistente (max_speed=1.5)...")
    c0_cons = JacobianDampedController(kp=3.0, lambda_damping=0.01, max_speed=1.5)
    df_c0_cons = collect_step_telemetry(c0_cons, target_episodes, ref_env_cfg)
    df_c0_cons["agent"] = "C0_consistent_1_5"
    df_c0_cons["seed"] = 0
    df_c0_cons["ckpt_step"] = -1
    all_dfs.append(df_c0_cons)

    print(">>> Recolectando telemetría C0 Histórico (max_speed=2.0)...")
    c0_hist = JacobianDampedController(kp=3.0, lambda_damping=0.01, max_speed=2.0)
    df_c0_hist = collect_step_telemetry(c0_hist, target_episodes, ref_env_cfg)
    df_c0_hist["agent"] = "C0_historical_2_0"
    df_c0_hist["seed"] = 0
    df_c0_hist["ckpt_step"] = -1
    all_dfs.append(df_c0_hist)

    # 2. Recolectar para cada agente seleccionado (semilla canónica 42)
    for sel in selected_ckpts:
        agent_id = sel["experiment"]
        chosen_step = sel["chosen_step"]
        cfg_path = REPO_ROOT / f"configs/phase_b/extension/{agent_id}.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            env_cfg = yaml.safe_load(f)

        if agent_id in ["X3", "X5_near"]:
            exp_base = REPO_ROOT / "experiments/phase_b" / agent_id / "seed_42"
        else:
            exp_base = REPO_ROOT / "experiments/phase_b/extension" / agent_id / "seed_42"

        ckpt_dir = exp_base / "checkpoints" / f"ckpt_{chosen_step}"
        model_path = ckpt_dir / "model.zip"
        norm_path = ckpt_dir / "vecnormalize_stats.pkl"
        norm_path_str = str(norm_path) if norm_path.exists() else None

        if not model_path.exists():
            print(f"AVISO: {model_path} no existe, omitiendo {agent_id}")
            continue

        print(f">>> Recolectando telemetría {agent_id} (ckpt {chosen_step}, semilla 42)...")
        model = PPO.load(str(model_path))
        df_agent = collect_step_telemetry(model, target_episodes, env_cfg, vec_normalize_path=norm_path_str)
        df_agent["agent"] = agent_id
        df_agent["seed"] = 42
        df_agent["ckpt_step"] = chosen_step
        all_dfs.append(df_agent)

    final_df = pd.concat(all_dfs, ignore_index=True)
    out_file = REPO_ROOT / output_csv
    out_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_file, index=False)
    print(f"\n>>> [OK] Telemetría guardada en: {out_file} ({len(final_df)} filas)")
    return final_df

if __name__ == "__main__":
    run_collect_all_step_data()
