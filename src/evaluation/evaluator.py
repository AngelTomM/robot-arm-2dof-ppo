"""
Evaluador de Políticas y Controladores bajo el Protocolo Estricto de Fase A (Secciones 8, 9, 16).
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
from arm2d_env import Arm2DEnv
from src.evaluation.metrics import compute_aggregate_metrics

def evaluate_on_dataset(agent, dataset: list, dataset_name: str,
                        eval_mode: str = "deterministic",
                        thresholds_path: str = "configs/phase_a/thresholds.yaml",
                        env_config: dict = None,
                        vec_normalize_path: str = None) -> tuple:
    """
    Ejecuta un agente (PPO o Controlador C0) en un conjunto de episodios bajo el Criterio Estricto.
    Retorna (episodes_df, aggregate_metrics).
    """
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresh_cfg = yaml.safe_load(f)

    strict_cfg = thresh_cfg.get("strict_eval_criteria", {})
    target_radius = float(strict_cfg.get("target_radius", 0.06))
    velocity_tolerance = float(strict_cfg.get("velocity_tolerance", 0.10))
    hold_steps = int(strict_cfg.get("hold_steps", 20))
    max_steps = int(strict_cfg.get("max_steps", 200))
    dt = float(strict_cfg.get("dt", 0.05))

    # Configuración estricta del entorno
    merged_cfg = env_config.copy() if env_config else {}
    if "env" not in merged_cfg:
        merged_cfg["env"] = {}
    merged_cfg["env"]["success_mode"] = "hold"
    merged_cfg["env"]["target_radius"] = target_radius
    merged_cfg["env"]["velocity_tolerance"] = velocity_tolerance
    merged_cfg["env"]["hold_steps"] = hold_steps
    merged_cfg["env"]["max_steps"] = max_steps
    merged_cfg["env"]["dt"] = dt

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
        steps_to_first_entry = None
        steps_to_success = None
        action_deltas = []

        reward_dist_sum = 0.0
        reward_near_sum = 0.0
        reward_action_sum = 0.0
        reward_vel_sum = 0.0
        reward_smooth_sum = 0.0
        reward_success_sum = 0.0

        joint_speed_at_success = np.nan
        ee_speed_at_success = np.nan
        min_distance = float("inf")

        while True:
            steps += 1
            # Normalizar observación para la red si aplica
            if norm_wrapper is not None:
                policy_obs = norm_wrapper.normalize_obs(obs)
            else:
                policy_obs = obs

            # Obtener acción
            if hasattr(agent, "predict"):
                # Si es un modelo SB3 o controlador con interfaz predict
                if hasattr(agent, "policy"): # SB3 Model
                    det = (eval_mode == "deterministic")
                    action, _ = agent.predict(policy_obs, deterministic=det)
                else: # Controlador C0
                    action = agent.predict(env)
            else:
                raise ValueError("El agente debe tener método predict()")

            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs

            # Acumular métricas
            if info["distance"] < min_distance:
                min_distance = info["distance"]

            if info["delta_action_norm"] is not None:
                action_deltas.append(info["delta_action_norm"])

            reward_dist_sum += info.get("reward_dist", 0.0)
            reward_near_sum += info.get("reward_near", 0.0)
            reward_action_sum += info.get("reward_action", 0.0)
            reward_vel_sum += info.get("reward_vel", 0.0)
            reward_smooth_sum += info.get("reward_smooth", 0.0)
            reward_success_sum += info.get("reward_success", 0.0)

            if info["first_entry_step"] is not None and steps_to_first_entry is None:
                steps_to_first_entry = info["first_entry_step"]

            if terminated and info["is_success"]:
                steps_to_success = steps
                joint_speed_at_success = info["joint_speed"]
                ee_speed_at_success = info["ee_speed"]

            if terminated or truncated:
                break

        # Métricas de suavidad agregadas por episodio
        if len(action_deltas) > 0:
            mean_act_delta = float(np.mean(action_deltas))
            med_act_delta = float(np.median(action_deltas))
            p90_act_delta = float(np.quantile(action_deltas, 0.90))
        else:
            mean_act_delta, med_act_delta, p90_act_delta = np.nan, np.nan, np.nan

        u_b = float(thresh_cfg["operational_thresholds"]["u_b"])
        n_o = int(thresh_cfg["operational_thresholds"]["n_exits_threshold"])
        n_c = int(thresh_cfg["operational_thresholds"]["n_cross_threshold"])

        entry_spd = info["joint_speed_at_first_entry"]
        is_ballistic = bool(entry_spd is not None and entry_spd > u_b)
        is_oscillatory = bool(info["n_exits"] >= n_o or info["n_cross"] >= n_c)

        records.append({
            "eval_set": dataset_name,
            "episode_id": ep_id,
            "evaluation_mode": eval_mode,
            "success": bool(info["is_success"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "steps": steps,
            "steps_to_first_target_entry": steps_to_first_entry if steps_to_first_entry is not None else np.nan,
            "steps_to_success": steps_to_success if steps_to_success is not None else np.nan,
            "final_distance": float(info["distance"]),
            "minimum_distance": min_distance,
            "joint_speed_at_first_entry": entry_spd if entry_spd is not None else np.nan,
            "joint_speed_at_success": joint_speed_at_success,
            "ee_speed_at_first_entry": info["ee_speed_at_first_entry"] if info["ee_speed_at_first_entry"] is not None else np.nan,
            "ee_speed_at_success": ee_speed_at_success,
            "mean_action_delta": mean_act_delta,
            "median_action_delta": med_act_delta,
            "p90_action_delta": p90_act_delta,
            "time_inside_030": int(info["time_inside_near"]),
            "hold_counter_max": int(info["hold_counter_max"]),
            "n_exits_after_first_entry": int(info["n_exits"]),
            "n_error_sign_crossings": int(info["n_cross"]),
            "ballistic_flag": is_ballistic,
            "oscillatory_flag": is_oscillatory,
            "reward_distance_sum": reward_dist_sum,
            "reward_near_sum": reward_near_sum,
            "reward_action_sum": reward_action_sum,
            "reward_velocity_sum": reward_vel_sum,
            "reward_smoothness_sum": reward_smooth_sum,
            "reward_success_sum": reward_success_sum
        })

    env.close()
    episodes_df = pd.DataFrame(records)
    agg_metrics = compute_aggregate_metrics(episodes_df, thresh_cfg)
    return episodes_df, agg_metrics
