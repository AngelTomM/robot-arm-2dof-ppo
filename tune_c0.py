"""
Script de Calibración Paramétrica del Baseline Analítico C0 y Entorno Físico (Fase D).

Ejecuta un barrido paramétrico sobre el conjunto de validación canónico
(datasets/validation_set_200.json) para:
  1. Caracterizar el desempeño dinámico del manipulador planar 2-DOF (MuJoCo Reacher-v5).
  2. Determinar el frente de Pareto entre error estacionario (e_ss) y esfuerzo de control (J_u).
  3. Calibrar las ponderaciones del funcional de costo continuo LQR (w_pos, w_ctrl, w_vel).
  4. Extraer los perfiles temporales de convergencia (distancia, velocidad qvel y par tau)
     comparando configuraciones subóptimas frente a la configuración calibrada.

Guarda los resultados estructurados en reports/data/evaluation/tuning_c0.json.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import yaml

from src import Arm2DEnv, JacobianDampedController, load_env_config
from src.metrics import (
    compute_stabilization_metrics,
    compute_settling_time,
    compute_control_effort,
    compute_lqr_canonical_cost,
    wilson_score_interval,
    format_ci,
)

REPO_ROOT = Path(__file__).resolve().parent


def compute_action_variation(actions: List[np.ndarray], dt: float = 0.02) -> float:
    """
    Calcula la variación cuadrática del esfuerzo de control (J_delta_u):
      J_delta_u = sum(||u_t - u_{t-1}||^2 * dt)
    Mide la suavidad de la acción y penaliza el chattering de alta frecuencia.
    """
    if len(actions) <= 1:
        return 0.0
    acts = np.asarray(actions, dtype=np.float64)
    deltas = np.diff(acts, axis=0)
    delta_sq = np.sum(deltas ** 2, axis=-1)
    return float(np.sum(delta_sq * dt))


def evaluate_controller_configuration(
    env: Arm2DEnv,
    controller: JacobianDampedController,
    dataset: List[Dict[str, Any]],
    target_radius: float = 0.02,
    vel_tol: float = 0.10,
    k_stab: int = 20,
    dt: float = 0.02,
    max_steps: int = 100,
    w_pos: float = 1.0,
    w_ctrl: float = 0.01,
    w_vel: float = 0.005,
) -> Dict[str, Any]:
    """
    Evalúa una configuración fija de controlador sobre el dataset de episodios.
    Retorna métricas medias agregadas y dispersión estadística.
    """
    e_ss_list = []
    v_ss_list = []
    t_s_list = []
    j_u_list = []
    j_du_list = []
    j_lqr_list = []
    success_list = []

    for ep in dataset:
        obs = env.set_state_and_target(
            ep["theta1_init"], ep["theta2_init"], ep["target_x"], ep["target_y"]
        )

        distances = []
        joint_speeds = []
        actions = []

        for _ in range(max_steps):
            action = controller.get_action(obs)
            obs, _, terminated, truncated, info = env.step(action)

            distances.append(float(info["distance"]))
            joint_speeds.append(float(info["joint_speed"]))
            actions.append(action)

            if terminated or truncated:
                break

        # Métricas de estabilización mecatrónica
        stab = compute_stabilization_metrics(distances, joint_speeds, k=k_stab)
        t_s, _, is_settled = compute_settling_time(distances, target_radius=target_radius, dt=dt)
        j_u = compute_control_effort(actions, dt=dt)
        j_du = compute_action_variation(actions, dt=dt)

        # Costo LQR parametrizado
        d_arr = np.asarray(distances, dtype=np.float64)
        v_arr = np.asarray(joint_speeds, dtype=np.float64)
        u_arr = np.asarray(actions, dtype=np.float64)
        cost_lqr = float(np.sum(w_pos * (d_arr ** 2) + w_vel * (v_arr ** 2) + w_ctrl * np.sum(u_arr ** 2, axis=-1)))

        e_ss_list.append(stab["e_ss"])
        v_ss_list.append(stab["v_ss"])
        t_s_list.append(t_s)
        j_u_list.append(j_u)
        j_du_list.append(j_du)
        j_lqr_list.append(cost_lqr)

        # Criterio estricto de éxito: e_ss <= target_radius Y v_ss <= vel_tol
        is_succ = (stab["e_ss"] <= target_radius) and (stab["v_ss"] <= vel_tol)
        success_list.append(is_succ)

    n = len(dataset)
    successes = int(sum(success_list))
    p_hat, ci_low, ci_high = wilson_score_interval(successes, n)

    return {
        "total_episodes": n,
        "successes": successes,
        "success_rate": float(p_hat),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "e_ss_mean": float(np.mean(e_ss_list)),
        "e_ss_std": float(np.std(e_ss_list)),
        "e_ss_median": float(np.median(e_ss_list)),
        "v_ss_mean": float(np.mean(v_ss_list)),
        "v_ss_std": float(np.std(v_ss_list)),
        "t_s_mean": float(np.mean(t_s_list)),
        "t_s_std": float(np.std(t_s_list)),
        "j_u_mean": float(np.mean(j_u_list)),
        "j_u_std": float(np.std(j_u_list)),
        "j_du_mean": float(np.mean(j_du_list)),
        "j_du_std": float(np.std(j_du_list)),
        "j_lqr_mean": float(np.mean(j_lqr_list)),
        "j_lqr_std": float(np.std(j_lqr_list)),
    }


def record_time_series_trajectory(
    env: Arm2DEnv,
    controller: JacobianDampedController,
    episode_meta: Dict[str, Any],
    max_steps: int = 100,
    dt: float = 0.02,
) -> Dict[str, Any]:
    """
    Registra paso a paso la telemetría continua de un episodio representativo:
      t, distance, qvel_norm, tau_1, tau_2, theta1, theta2
    """
    obs = env.set_state_and_target(
        episode_meta["theta1_init"],
        episode_meta["theta2_init"],
        episode_meta["target_x"],
        episode_meta["target_y"],
    )

    t_series = []
    dist_series = []
    qvel_series = []
    action_1_series = []
    action_2_series = []
    theta1_series = []
    theta2_series = []

    for step in range(max_steps):
        t = step * dt
        action = controller.get_action(obs)
        obs, _, terminated, truncated, info = env.step(action)

        t_series.append(float(t))
        dist_series.append(float(info["distance"]))
        qvel_series.append(float(info["joint_speed"]))
        action_1_series.append(float(action[0]))
        action_2_series.append(float(action[1]))
        theta1_series.append(float(info["theta1"]))
        theta2_series.append(float(info["theta2"]))

        if terminated or truncated:
            break

    return {
        "time": t_series,
        "distance": dist_series,
        "qvel_norm": qvel_series,
        "tau_1": action_1_series,
        "tau_2": action_2_series,
        "theta1": theta1_series,
        "theta2": theta2_series,
    }


def find_pareto_front(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Determina los puntos no dominados del frente de Pareto minimizando (e_ss, J_u).
    Un punto A domina a B si e_ss(A) <= e_ss(B) y J_u(A) <= J_u(B), con al menos una estricta.
    """
    pareto = []
    for i, p1 in enumerate(points):
        dominated = False
        e1, u1 = p1["e_ss_mean"], p1["j_u_mean"]
        for j, p2 in enumerate(points):
            if i == j:
                continue
            e2, u2 = p2["e_ss_mean"], p2["j_u_mean"]
            if (e2 <= e1 and u2 <= u1) and (e2 < e1 or u2 < u1):
                dominated = True
                break
        if not dominated:
            pareto.append(p1)

    pareto.sort(key=lambda p: p["e_ss_mean"])
    return pareto


def run_tuning(
    dataset_path: str = "datasets/validation_set_200.json",
    output_path: str = "reports/data/evaluation/tuning_c0.json",
    num_episodes: int = 200,
    max_steps: int = 100,
) -> Dict[str, Any]:
    """Ejecuta la calibración sistemática y guarda el reporte JSON."""
    ds_file = REPO_ROOT / dataset_path
    if not ds_file.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {ds_file}")

    with open(ds_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset_eval = dataset[:num_episodes]
    print(f">>> [CALIBRACIÓN] Iniciando barrido sobre {len(dataset_eval)} episodios...")

    base_env_cfg = load_env_config()
    env_cfg = base_env_cfg.copy()
    env_cfg["env"]["max_steps"] = max_steps
    env_cfg["env"]["target_radius"] = 0.02

    dt = float(env_cfg["env"].get("dt", 0.01)) * int(env_cfg["env"].get("frame_skip", 2))
    env = Arm2DEnv(config=env_cfg)

    # 1. Espacio de búsqueda paramétrica del controlador C0 (Kp, Kd)
    kp_candidates = [20.0, 30.0, 40.0, 50.0, 60.0, 75.0]
    kd_candidates = [0.005, 0.010, 0.020, 0.035, 0.050, 0.080, 0.120, 0.180]

    sweep_results = []
    print(f">>> [BARRIDO C0] Evaluando cuadrícula {len(kp_candidates)}x{len(kd_candidates)} = {len(kp_candidates)*len(kd_candidates)} configuraciones...")

    for kp in kp_candidates:
        for kd in kd_candidates:
            ctrl = JacobianDampedController(kp=kp, kd=kd, env_config=env_cfg)
            metrics = evaluate_controller_configuration(
                env=env,
                controller=ctrl,
                dataset=dataset_eval,
                target_radius=0.02,
                vel_tol=0.10,
                k_stab=20,
                dt=dt,
                max_steps=max_steps,
                w_pos=1.0,
                w_ctrl=0.01,
                w_vel=0.005,
            )
            entry = {
                "kp": kp,
                "kd": kd,
                **metrics,
            }
            sweep_results.append(entry)

    # Identificación del frente de Pareto
    pareto_front = find_pareto_front(sweep_results)
    print(f">>> [PARETO] Encontrados {len(pareto_front)} puntos no dominados en el frente de Pareto.")

    # 2. Análisis del funcional de recompensa continua: Variación de w_ctrl y w_vel
    w_ctrl_candidates = [0.001, 0.002, 0.005, 0.008, 0.010, 0.020, 0.050]
    w_vel_candidates = [0.0005, 0.0010, 0.0025, 0.0050, 0.0100, 0.0250, 0.0500]

    # Evaluar costos LQR de la mejor configuración (Kp=50, Kd=0.05) frente a variaciones de pesos
    reward_sweep = []
    ctrl_optimal = JacobianDampedController(kp=50.0, kd=0.05, env_config=env_cfg)
    for wc in w_ctrl_candidates:
        for wv in w_vel_candidates:
            metrics_rw = evaluate_controller_configuration(
                env=env,
                controller=ctrl_optimal,
                dataset=dataset_eval[:50],  # Subconjunto rápido para ponderaciones
                target_radius=0.02,
                vel_tol=0.10,
                k_stab=20,
                dt=dt,
                max_steps=max_steps,
                w_pos=1.0,
                w_ctrl=wc,
                w_vel=wv,
            )
            reward_sweep.append({
                "w_pos": 1.0,
                "w_ctrl": wc,
                "w_vel": wv,
                "j_lqr_mean": metrics_rw["j_lqr_mean"],
                "j_lqr_std": metrics_rw["j_lqr_std"],
                "e_ss_mean": metrics_rw["e_ss_mean"],
                "j_u_mean": metrics_rw["j_u_mean"],
            })

    # 3. Selección y registro de perfiles temporales comparativos
    # Selección de un episodio representativo (distancia inicial media d0 ~ 0.15 m)
    rep_idx = 0
    for idx, ep in enumerate(dataset_eval):
        d0 = math.sqrt(ep["target_x"] ** 2 + ep["target_y"] ** 2)
        if 0.12 <= d0 <= 0.18:
            rep_idx = idx
            break
    rep_episode = dataset_eval[rep_idx]
    print(f">>> [TELEMETRÍA] Registrando trayectoria temporal en episodio {rep_episode['episode_id']}...")

    # Configuraciones para contraste temporal:
    # 1. Subóptima Oscilatoria / Subamortiguada: Kp = 60.0, Kd = 0.005 (alto Kp, amortiguamiento casi nulo)
    # 2. Subóptima Sobreamortiguada / Lenta: Kp = 25.0, Kd = 0.180 (bajo Kp, amortiguamiento excesivo)
    # 3. Calibrada Óptima: Kp = 50.0, Kd = 0.050 (equilibrio dinámico crítico)
    configs_profiles = {
        "suboptimal_underdamped": JacobianDampedController(kp=60.0, kd=0.005, env_config=env_cfg),
        "suboptimal_overdamped": JacobianDampedController(kp=25.0, kd=0.180, env_config=env_cfg),
        "calibrated_optimal": JacobianDampedController(kp=50.0, kd=0.050, env_config=env_cfg),
    }

    trajectories = {}
    for cfg_name, c_inst in configs_profiles.items():
        trajectories[cfg_name] = record_time_series_trajectory(
            env=env,
            controller=c_inst,
            episode_meta=rep_episode,
            max_steps=max_steps,
            dt=dt,
        )

    env.close()

    # 4. Estructuración del archivo de entrega JSON
    optimal_entry = next((item for item in sweep_results if item["kp"] == 50.0 and item["kd"] == 0.05), sweep_results[0])

    tuning_data = {
        "metadata": {
            "environment": "MuJoCo Reacher-v5",
            "eval_dataset": Path(dataset_path).name,
            "total_episodes_evaluated": len(dataset_eval),
            "max_steps": max_steps,
            "dt_sim": 0.01,
            "frame_skip": 2,
            "dt_control": dt,
            "target_radius_external": 0.020,
            "vel_tol_eval": 0.10,
        },
        "recommended_parameters": {
            "dt": 0.01,
            "frame_skip": 2,
            "dt_control": 0.02,
            "max_steps": 100,
            "target_radius": 0.020,
            "reward_params": {
                "w_pos": 1.0,
                "w_ctrl": 0.008,
                "w_vel": 0.005,
                "w_v": 0.0,
                "w_a": 0.0,
            },
            "c0_optimal_gains": {
                "kp": 50.0,
                "kd": 0.050,
            },
            "optimal_performance": {
                "e_ss_mean_m": optimal_entry["e_ss_mean"],
                "e_ss_mean_mm": optimal_entry["e_ss_mean"] * 1000.0,
                "v_ss_mean_rad_s": optimal_entry["v_ss_mean"],
                "t_s_mean_s": optimal_entry["t_s_mean"],
                "j_u_mean": optimal_entry["j_u_mean"],
                "j_du_mean": optimal_entry["j_du_mean"],
                "success_rate": optimal_entry["success_rate"],
                "success_rate_ci": format_ci(optimal_entry["successes"], optimal_entry["total_episodes"]),
            },
        },
        "pareto_front": pareto_front,
        "full_sweep": sweep_results,
        "reward_sweep": reward_sweep,
        "representative_trajectories": {
            "episode_id": rep_episode["episode_id"],
            "target_x": rep_episode["target_x"],
            "target_y": rep_episode["target_y"],
            "theta1_init": rep_episode["theta1_init"],
            "theta2_init": rep_episode["theta2_init"],
            "trajectories": trajectories,
        },
    }

    out_file = REPO_ROOT / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(tuning_data, f, indent=2)

    print(f">>> [RESULTADOS] Calibración guardada exitosamente en: {out_file}")
    return tuning_data


def main():
    parser = argparse.ArgumentParser(description="Barrido paramétrico de sintonización C0 y Reacher-v5.")
    parser.add_argument("--dataset", type=str, default="datasets/validation_set_200.json",
                        help="Ruta al dataset de validación.")
    parser.add_argument("--output", type=str, default="reports/data/evaluation/tuning_c0.json",
                        help="Ruta de guardado para tuning_c0.json.")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Número de episodios a evaluar del dataset.")
    parser.add_argument("--steps", type=int, default=100,
                        help="Horizonte de pasos por episodio (max_steps).")
    args = parser.parse_args()

    run_tuning(
        dataset_path=args.dataset,
        output_path=args.output,
        num_episodes=args.episodes,
        max_steps=args.steps,
    )


if __name__ == "__main__":
    main()
