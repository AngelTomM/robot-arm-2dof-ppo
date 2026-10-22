"""
Trayectorias cartesianas representativas para Fase D (MDP Continuo con Regulación LQR).

Simula agentes entrenados (PPO/SAC) bajo evaluación determinista y genera
fig5_representative_real_trajectories.png visualizando:
  - Trayectoria completa del efector final.
  - Ventana de estabilización de los últimos k=20 pasos (destacada).
  - Zona meta de radio r=0.06m y métricas duales (Opción 2 y Opción 3 J_LQR).
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from arm2d_env import Arm2DEnv
from src.evaluation.datasets import save_and_get_datasets
from src.evaluation.metrics import compute_stabilization_metrics, compute_lqr_canonical_cost


def simulate_trajectory(model, norm_stats_path, ep_meta, env_cfg=None):
    merged_cfg = env_cfg.copy() if env_cfg else {}
    if "env" not in merged_cfg:
        merged_cfg["env"] = {}
    merged_cfg["env"]["reward_type"] = "lyapunov_lqr"
    merged_cfg["env"]["success_mode"] = "none"
    merged_cfg["env"]["max_steps"] = 200

    env = Arm2DEnv(config=merged_cfg)

    norm_wrapper = None
    if norm_stats_path and os.path.exists(norm_stats_path):
        dummy = DummyVecEnv([lambda: Arm2DEnv(config=merged_cfg)])
        norm_wrapper = VecNormalize.load(norm_stats_path, dummy)
        norm_wrapper.training = False
        norm_wrapper.norm_reward = False

    obs = env.set_state_and_target(ep_meta["theta1_init"], ep_meta["theta2_init"], ep_meta["target_x"], ep_meta["target_y"])

    ee_positions = []
    elbow_positions = []
    distances = []
    joint_speeds = []
    actions = []

    for step in range(200):
        elbow, ee = env._forward_kinematics(env.theta1, env.theta2)
        ee_positions.append(np.copy(ee))
        elbow_positions.append(np.copy(elbow))
        distances.append(float(np.linalg.norm(env.target - ee)))
        joint_speeds.append(float(math.sqrt(env.dtheta1**2 + env.dtheta2**2)))

        policy_obs = norm_wrapper.normalize_obs(obs) if norm_wrapper else obs
        action, _ = model.predict(policy_obs, deterministic=True)
        actions.append(np.copy(action))
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs

        if terminated or truncated:
            elbow, ee = env._forward_kinematics(env.theta1, env.theta2)
            ee_positions.append(np.copy(ee))
            elbow_positions.append(np.copy(elbow))
            break

    env.close()

    stab = compute_stabilization_metrics(distances, joint_speeds, k=20)
    lqr = compute_lqr_canonical_cost(distances, joint_speeds, actions)

    return {
        "ee_traj": np.array(ee_positions),
        "elbow_traj": np.array(elbow_positions),
        "target": np.array(env.target),
        "target_radius": 0.06,
        "terminated": terminated,
        "truncated": truncated,
        "is_success": stab["is_success"],
        "mean_dist_20": stab["mean_dist_last_k"],
        "mean_vel_20": stab["mean_vel_last_k"],
        "j_lqr": lqr["j_lqr"],
        "info": info
    }


def plot_real_examples(output_dir: str = "reports/figures/phase_X"):
    os.makedirs(output_dir, exist_ok=True)
    _, final_dataset = save_and_get_datasets()

    # Buscar modelos disponibles en orden de prioridad
    candidate_paths = [
        ("PPO_1", "experiments/phase_X/prueba/PPO_1/seed_42/models/best_model/best_model.zip",
         "experiments/phase_X/prueba/PPO_1/seed_42/models/best_model/vecnormalize_stats.pkl"),
        ("SAC_1", "experiments/phase_X/prueba/SAC_1/seed_42/models/best_model/best_model.zip",
         None),
    ]

    selected_model = None
    selected_norm = None
    for label, m_path, n_path in candidate_paths:
        if os.path.exists(m_path):
            try:
                selected_model = PPO.load(m_path)
            except Exception:
                selected_model = SAC.load(m_path)
            selected_norm = n_path if (n_path and os.path.exists(n_path)) else None
            print(f">>> [TRAJECTORIES] Usando modelo {label} desde {m_path}")
            break

    if selected_model is None:
        print(">>> [TRAJECTORIES] Modelos Phase D aún no listos para trazado.")
        return

    # Simular 4 episodios representativos
    ep_indices = [0, 5, 10, 15]
    trajs = []
    for idx in ep_indices:
        t = simulate_trajectory(selected_model, selected_norm, final_dataset[idx])
        trajs.append(t)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
    flat_axes = axes.flatten()

    for ax, data, ep_idx in zip(flat_axes, trajs, ep_indices):
        circle_reach = patches.Circle((0, 0), 2.0, color="#ecf0f1", fill=True, zorder=0)
        ax.add_patch(circle_reach)
        ax.plot([0], [0], "ko", markersize=8, label="Base (0,0)")

        tgt = data["target"]
        tgt_circle = patches.Circle((tgt[0], tgt[1]), data["target_radius"], color="#27ae60", alpha=0.3, zorder=1)
        ax.add_patch(tgt_circle)
        ax.plot(tgt[0], tgt[1], "g*", markersize=12, label=f"Meta (r={data['target_radius']}m)")

        ee_tr = data["ee_traj"]
        # Trayectoria general (pasos iniciales)
        k_stab = min(20, len(ee_tr))
        if len(ee_tr) > k_stab:
            ax.plot(ee_tr[:-k_stab, 0], ee_tr[:-k_stab, 1], "-", color="#2980b9", linewidth=2.0, alpha=0.6, label="Aproximación")
        # Ventana de estabilización k=20 pasos finales destacada
        ax.plot(ee_tr[-k_stab:, 0], ee_tr[-k_stab:, 1], "-", color="#e74c3c" if not data["is_success"] else "#27ae60",
                linewidth=3.2, label=f"Últimos {k_stab} pasos (Estabilización)", zorder=4)

        ax.plot(ee_tr[0, 0], ee_tr[0, 1], "o", color="#34495e", markersize=8, label="Inicio Efector")
        ax.plot(ee_tr[-1, 0], ee_tr[-1, 1], "s", color="#2c3e50", markersize=8, label="Fin Efector")

        if len(ee_tr) > 0 and len(data["elbow_traj"]) > 0:
            elbow_fin = data["elbow_traj"][-1]
            ee_fin = ee_tr[-1]
            ax.plot([0, elbow_fin[0]], [0, elbow_fin[1]], color="#2980b9", linewidth=4, alpha=0.7)
            ax.plot([elbow_fin[0], ee_fin[0]], [elbow_fin[1], ee_fin[1]], color="#f39c12", linewidth=3, alpha=0.7)
            ax.plot(elbow_fin[0], elbow_fin[1], "o", color="#2c3e50", markersize=6)

        status_str = "ÉXITO" if data["is_success"] else "NO ESTABILIZADO"
        title = f"Episodio {ep_idx} ({status_str})\n" \
                f"d_20={data['mean_dist_20']*100:.1f}cm, v_20={data['mean_vel_20']:.2f}rad/s, J_LQR={data['j_lqr']:.1f}"

        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figX_representative_real_trajectories.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f">>> [OK] Figuras de trayectorias representativas guardadas en: {fig_path}")


def run(out_dir=None):
    """Punto de entrada del módulo."""
    plot_real_examples(out_dir if out_dir is not None else "reports/figures/phase_X")


if __name__ == "__main__":
    run()

