"""
Evaluador de Políticas y Controladores para Fase D (MDP Continuo LQR).
Ejecuta la evaluación dual sobre datasets canónicos reproducibles:
- Métrica Principal (Opción 2): Estabilización en los últimos 20 pasos (d_20 <= 0.06 m, v_20 <= 0.10 rad/s).
- Métrica Comparativa (Opción 3): Costo LQR acumulado canónico independiente del reward de entrenamiento.
"""

import os
import sys
import numpy as np
import pandas as pd
from .arm2d_env import Arm2DEnv
from .metrics import (
    K_STAB, TARGET_RADIUS_EVAL, VEL_TOL_EVAL,
    compute_stabilization_metrics, compute_lqr_canonical_cost,
    compute_settling_time, compute_control_effort,
    compute_aggregate_metrics
)


def evaluate_on_dataset(agent, dataset: list, dataset_name: str,
                        eval_mode: str = "deterministic",
                        env_config: dict = None,
                        vec_normalize_path: str = None) -> tuple:
    """
    Ejecuta un agente (PPO, SAC o Controlador C0) en un conjunto de episodios reproducibles.
    Calcula sobre la trayectoria física cruda:
      1. Métrica Principal: Estabilización en ventana final de 20 pasos (Opción 2).
      2. Métrica Comparativa: Costo acumulado canónico LQR (Opción 3).
    Retorna (episodes_df, aggregate_metrics).
    """
    if env_config is None:
        try:
            from .config import load_env_config
            merged_cfg = load_env_config()
        except Exception:
            merged_cfg = {"env": {}}
    else:
        merged_cfg = env_config.copy()

    if "env" not in merged_cfg:
        merged_cfg["env"] = {}

    # Forzar parámetros del entorno para Fase D (MDP continuo de horizonte fijo)
    merged_cfg["env"]["success_mode"] = "none"
    merged_cfg["env"]["reward_type"] = merged_cfg["env"].get("reward_type", "lyapunov_lqr")
    merged_cfg["env"]["target_radius"] = TARGET_RADIUS_EVAL
    merged_cfg["env"]["max_steps"] = int(merged_cfg["env"].get("max_steps", 200))
    merged_cfg["env"]["dt"] = float(merged_cfg["env"].get("dt", 0.05))

    env = Arm2DEnv(config=merged_cfg)

    # Cargar VecNormalize si se especifica
    norm_wrapper = None
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        dummy = DummyVecEnv([lambda: Arm2DEnv(config=merged_cfg)])
        norm_wrapper = VecNormalize.load(vec_normalize_path, dummy)
        norm_wrapper.training = False
        norm_wrapper.norm_reward = False

    records = []

    for ep_meta in dataset:
        ep_id = ep_meta["episode_id"]
        th1_init = ep_meta["theta1_init"]
        th2_init = ep_meta["theta2_init"]
        tgt_x = ep_meta["target_x"]
        tgt_y = ep_meta["target_y"]

        obs = env.set_state_and_target(th1_init, th2_init, tgt_x, tgt_y)

        steps = 0
        distances = []
        joint_speeds = []
        actions = []
        action_deltas = []
        rewards = []
        prev_action = None

        while True:
            steps += 1

            # Normalizar observación para la red si aplica
            if norm_wrapper is not None:
                policy_obs = norm_wrapper.normalize_obs(obs)
            else:
                policy_obs = obs

            # Obtener acción del agente (SB3 o C0)
            if hasattr(agent, "predict"):
                det = (eval_mode == "deterministic")
                pred_out = agent.predict(policy_obs, deterministic=det)
                action = pred_out[0] if isinstance(pred_out, tuple) else pred_out
            elif hasattr(agent, "get_action"):
                action = agent.get_action(policy_obs)
            else:
                raise ValueError("El agente debe tener método predict() o get_action()")

            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs

            # Registrar telemetría física cruda
            distances.append(float(info["distance"]))
            joint_speeds.append(float(info["joint_speed"]))
            actions.append(np.copy(action))
            rewards.append(float(reward))

            if prev_action is not None:
                action_deltas.append(float(np.linalg.norm(action - prev_action)))
            prev_action = np.copy(action)

            if terminated or truncated:
                break

        dt_base = float(merged_cfg["env"].get("dt", 0.01))
        frame_skip = int(merged_cfg["env"].get("frame_skip", 2))
        dt = dt_base * frame_skip

        # 1. Métrica Principal — Estabilización régimen permanente
        stab = compute_stabilization_metrics(distances, joint_speeds, k=K_STAB)

        # 2. Métrica Comparativa — Costo LQR Canónico
        lqr = compute_lqr_canonical_cost(distances, joint_speeds, actions)

        # 3. Métricas Temporales y Energéticas (t_s, J_u)
        t_s, k_settle, is_settled = compute_settling_time(distances, target_radius=TARGET_RADIUS_EVAL, dt=dt)
        j_u = compute_control_effort(actions, dt=dt)

        mean_act_delta = float(np.mean(action_deltas)) if action_deltas else np.nan

        records.append({
            "eval_set": dataset_name,
            "episode_id": ep_id,
            "evaluation_mode": eval_mode,
            "success": bool(stab["is_success"]),
            "e_ss": float(stab["e_ss"]),
            "v_ss": float(stab["v_ss"]),
            "t_s": float(t_s),
            "j_u": float(j_u),
            "is_settled": bool(is_settled),
            "settling_steps": int(k_settle),
            "mean_dist_last_20": float(stab["mean_dist_last_k"]),
            "mean_vel_last_20": float(stab["mean_vel_last_k"]),
            "final_distance": float(distances[-1]),
            "final_joint_speed": float(joint_speeds[-1]),
            "minimum_distance": float(np.min(distances)),
            "steps": steps,
            "j_lqr": float(lqr["j_lqr"]),
            "j_pos": float(lqr["j_pos"]),
            "j_vel": float(lqr["j_vel"]),
            "j_ctrl": float(lqr["j_ctrl"]),
            "mean_action_delta": mean_act_delta,
            "total_env_reward": float(np.sum(rewards))
        })

    env.close()
    episodes_df = pd.DataFrame(records)
    agg_metrics = compute_aggregate_metrics(episodes_df)
    return episodes_df, agg_metrics
