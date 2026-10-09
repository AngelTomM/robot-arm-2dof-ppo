"""
Extractor de Telemetría Paso a Paso para Fase B:
- Captura por paso: posición del efector (x, y), distancia al target, velocidad articular,
  velocidad cartesiana, acciones, suavidad ||Delta a|| y contador de hold.
- Funciona con modelos PPO (Fase A y Fase B) y con el controlador analítico C0.
- Solo inferencia, sin modificar el entorno ni alterar los datasets.
"""
import os
import sys
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arm2d_env import Arm2DEnv
from src.controllers.jacobian_controller import JacobianDampedController
from src.evaluation.datasets import save_and_get_datasets
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def collect_step_telemetry_for_episodes(agent, episodes: list, env_cfg: dict,
                                       eval_mode: str = "deterministic",
                                       vec_normalize_path: str = None) -> pd.DataFrame:
    """
    Ejecuta un agente sobre una lista de episodios y registra el vector completo de variables paso a paso.
    """
    merged_cfg = env_cfg.copy() if env_cfg else {}
    if "env" not in merged_cfg:
        merged_cfg["env"] = {}
    
    # Criterio estricto de evaluación
    merged_cfg["env"]["success_mode"] = "hold"
    merged_cfg["env"]["target_radius"] = 0.06
    merged_cfg["env"]["velocity_tolerance"] = 0.10
    merged_cfg["env"]["hold_steps"] = 20
    merged_cfg["env"]["max_steps"] = 200
    merged_cfg["env"]["dt"] = 0.05

    env = Arm2DEnv(config=merged_cfg)

    norm_wrapper = None
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        dummy = DummyVecEnv([lambda: Arm2DEnv(config=merged_cfg)])
        norm_wrapper = VecNormalize.load(vec_normalize_path, dummy)
        norm_wrapper.training = False
        norm_wrapper.norm_reward = False

    all_steps = []

    for ep_meta in episodes:
        ep_id = ep_meta["episode_id"]
        th1_init = ep_meta["theta1_init"]
        th2_init = ep_meta["theta2_init"]
        tgt_x = ep_meta["target_x"]
        tgt_y = ep_meta["target_y"]
        tgt_r = ep_meta.get("target_r", np.sqrt(tgt_x**2 + tgt_y**2))

        obs = env.set_state_and_target(th1_init, th2_init, tgt_x, tgt_y)
        
        step_idx = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            step_idx += 1
            if norm_wrapper is not None:
                policy_obs = norm_wrapper.normalize_obs(obs)
            else:
                policy_obs = obs

            if hasattr(agent, "predict"):
                if hasattr(agent, "policy"):  # PPO SB3
                    det = (eval_mode == "deterministic")
                    action, _ = agent.predict(policy_obs, deterministic=det)
                else:  # C0 Jacobian
                    action = agent.predict(env)
            else:
                raise ValueError("El agente debe tener método predict()")

            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs

            _, ee_pos = env._forward_kinematics(env.theta1, env.theta2)
            joint_speed = info.get("joint_speed", float(np.sqrt(env.dtheta1**2 + env.dtheta2**2)))
            ee_spd = info.get("ee_speed", 0.0)

            step_data = {
                "episode_id": ep_id,
                "step": step_idx,
                "eval_mode": eval_mode,
                "target_x": tgt_x,
                "target_y": tgt_y,
                "target_r": tgt_r,
                "ee_x": float(ee_pos[0]),
                "ee_y": float(ee_pos[1]),
                "theta1": float(env.theta1),
                "theta2": float(env.theta2),
                "dtheta1": float(env.dtheta1),
                "dtheta2": float(env.dtheta2),
                "joint_speed": float(joint_speed),
                "ee_speed": float(ee_spd),
                "distance": float(info["distance"]),
                "action_0": float(action[0]),
                "action_1": float(action[1]),
                "delta_action_norm": float(info["delta_action_norm"]) if info["delta_action_norm"] is not None else np.nan,
                "hold_counter": int(info["hold_counter"]),
                "is_inside_target": bool(float(info["distance"]) < float(env.target_radius)),
                "time_inside_near": int(info["time_inside_near"]),
                "is_success": bool(info["is_success"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated)
            }
            all_steps.append(step_data)

    env.close()
    return pd.DataFrame(all_steps)

if __name__ == "__main__":
    val_set, _ = save_and_get_datasets()
    # Prueba rápida sobre 1 episodio
    c0_ctrl = JacobianDampedController(kp=3.0, lambda_damping=0.01)
    with open("configs/phase_a/X5_near.yaml") as f:
        cfg = yaml.safe_load(f)
    df = collect_step_telemetry_for_episodes(c0_ctrl, val_set[:1], cfg)
    print(f"Recolectados {len(df)} pasos para episodio {df['episode_id'].iloc[0]}. Exito: {df['is_success'].iloc[-1]}")
