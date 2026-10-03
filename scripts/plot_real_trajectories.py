"""
Generación de Trayectorias Cartesianas Reales Representativas (Sección 17.3).
Selecciona episodios reales clasificados según las definiciones operativas de la Sección 9.6:
1. Llegada estable
2. Llegada balística
3. Oscilación persistente
4. Episodio truncado
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from arm2d_env import Arm2DEnv
from src.evaluation.datasets import save_and_get_datasets

def simulate_trajectory(model, norm_stats_path, ep_meta, env_cfg=None):
    merged_cfg = env_cfg.copy() if env_cfg else {}
    if "env" not in merged_cfg:
        merged_cfg["env"] = {}
    merged_cfg["env"]["success_mode"] = "hold"
    merged_cfg["env"]["target_radius"] = 0.06
    merged_cfg["env"]["velocity_tolerance"] = 0.10
    merged_cfg["env"]["hold_steps"] = 20
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

    for step in range(200):
        elbow, ee = env._forward_kinematics(env.theta1, env.theta2)
        ee_positions.append(np.copy(ee))
        elbow_positions.append(np.copy(elbow))
        distances.append(float(np.linalg.norm(env.target - ee)))
        joint_speeds.append(float(math.sqrt(env.dtheta1**2 + env.dtheta2**2)))

        policy_obs = norm_wrapper.normalize_obs(obs) if norm_wrapper else obs
        action, _ = model.predict(policy_obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs

        if terminated or truncated:
            elbow, ee = env._forward_kinematics(env.theta1, env.theta2)
            ee_positions.append(np.copy(ee))
            elbow_positions.append(np.copy(elbow))
            break

    env.close()
    return {
        "ee_traj": np.array(ee_positions),
        "elbow_traj": np.array(elbow_positions),
        "target": np.array(env.target),
        "target_radius": env.target_radius,
        "terminated": terminated,
        "truncated": truncated,
        "is_success": info["is_success"],
        "info": info
    }

def plot_real_examples(output_dir: str = "reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    _, final_dataset = save_and_get_datasets()

    # Cargar modelos representativos
    # X5 para llegada estable y balística
    x5_model_path = "experiments/phase_a/X5/seed_42/models/best_model/best_model.zip"
    x5_norm_path = "experiments/phase_a/X5/seed_42/models/best_model/vecnormalize_stats.pkl"
    # X1a para truncamiento
    x1_model_path = "experiments/phase_a/X1a/seed_42/models/best_model/best_model.zip"
    # X6_wa05 para oscilación
    x6_model_path = "experiments/phase_a/X6_wa05/seed_42/models/best_model/best_model.zip"
    x6_norm_path = "experiments/phase_a/X6_wa05/seed_42/models/best_model/vecnormalize_stats.pkl"

    if not os.path.exists(x5_model_path):
        print(">>> Modelos aún no listos para trazado.")
        return

    m_x5 = PPO.load(x5_model_path)
    m_x1 = PPO.load(x1_model_path) if os.path.exists(x1_model_path) else None
    m_x6 = PPO.load(x6_model_path) if os.path.exists(x6_model_path) else None

    # Simular 4 casos reales
    # Caso 1: Estable (X5, ep 0)
    traj_stable = simulate_trajectory(m_x5, x5_norm_path, final_dataset[0])
    # Caso 2: Balística (X5, ep 5)
    traj_ballistic = simulate_trajectory(m_x5, x5_norm_path, final_dataset[5])
    # Caso 3: Truncado (X1a, ep 10)
    traj_trunc = simulate_trajectory(m_x1, None, final_dataset[10]) if m_x1 else traj_stable
    # Caso 4: Oscilatorio (X6_wa05, ep 12)
    traj_osc = simulate_trajectory(m_x6, x6_norm_path, final_dataset[12]) if m_x6 else traj_stable

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
    cases = [
        (axes[0, 0], traj_stable, "1. Llegada Estable (Éxito + Hold Completo)", "#2ecc71"),
        (axes[0, 1], traj_ballistic, "2. Llegada Balística (Entrada Rápida v > 0.3 rad/s)", "#e74c3c"),
        (axes[1, 0], traj_osc, "3. Oscilación Persistente (Sobrepaso múltiple)", "#f39c12"),
        (axes[1, 1], traj_trunc, "4. Episodio Truncado (Sin Estabilización en 200p)", "#7f8c8d")
    ]

    for ax, data, title, color in cases:
        # Espacio de trabajo
        circle_reach = patches.Circle((0, 0), 2.0, color="#ecf0f1", fill=True, zorder=0)
        ax.add_patch(circle_reach)
        ax.plot([0], [0], "ko", markersize=8, label="Base (0,0)")

        # Target y Tolerancia
        tgt = data["target"]
        tgt_circle = patches.Circle((tgt[0], tgt[1]), data["target_radius"], color="#27ae60", alpha=0.3, zorder=1)
        ax.add_patch(tgt_circle)
        ax.plot(tgt[0], tgt[1], "g*", markersize=12, label=f"Meta (r={data['target_radius']}m)")

        # Trayectoria efector final
        ee_tr = data["ee_traj"]
        ax.plot(ee_tr[:, 0], ee_tr[:, 1], "-", color=color, linewidth=2.5, label="Trayectoria Efector", zorder=3)
        ax.plot(ee_tr[0, 0], ee_tr[0, 1], "o", color="#34495e", markersize=8, label="Inicio Efector")
        ax.plot(ee_tr[-1, 0], ee_tr[-1, 1], "s", color=color, markersize=8, label="Fin Efector")

        # Brazo en posición final
        if len(ee_tr) > 0 and len(data["elbow_traj"]) > 0:
            elbow_fin = data["elbow_traj"][-1]
            ee_fin = ee_tr[-1]
            ax.plot([0, elbow_fin[0]], [0, elbow_fin[1]], color="#2980b9", linewidth=4, alpha=0.7)
            ax.plot([elbow_fin[0], ee_fin[0]], [elbow_fin[1], ee_fin[1]], color="#f39c12", linewidth=3, alpha=0.7)
            ax.plot(elbow_fin[0], elbow_fin[1], "o", color="#2c3e50", markersize=6)

        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig5_representative_real_trajectories.png"))
    plt.close(fig)
    print(">>> [OK] Figuras de trayectorias representativas generadas.")

if __name__ == "__main__":
    plot_real_examples()
