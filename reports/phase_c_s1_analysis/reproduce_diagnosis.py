"""
Script de ejecución integral del diagnóstico técnico de SAC S1.
Calcula métricas agregadas por checkpoint, análisis espacial,
selecciona y traza trayectorias representativas, extrae logs de TensorBoard
y genera las 11 figuras obligatorias.
"""

import os
import sys
import json
import zipfile
import io
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Configuración visual
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#bbbbbb'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = str(SCRIPT_DIR.parents[1])
VAL_DATASET_PATH = os.path.join(REPO_ROOT, "data", "datasets", "phase_c", "validation_c_500.json")
OUTPUT_DIR = str(SCRIPT_DIR)
FIGURES_DIR = str(SCRIPT_DIR / "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# 1. Cargar Dataset de Validación Autorizado
with open(VAL_DATASET_PATH, "r", encoding="utf-8") as f:
    validation_dataset = json.load(f)

N_EPISODES = len(validation_dataset)
print(f"[HECHO] Dataset cargado: {VAL_DATASET_PATH} ({N_EPISODES} episodios)")

th1_init = np.array([ep["theta1_init"] for ep in validation_dataset], dtype=np.float32)
th2_init = np.array([ep["theta2_init"] for ep in validation_dataset], dtype=np.float32)
tgt_x = np.array([ep["target_x"] for ep in validation_dataset], dtype=np.float32)
tgt_y = np.array([ep["target_y"] for ep in validation_dataset], dtype=np.float32)
tgt_r = np.sqrt(tgt_x**2 + tgt_y**2)

region_masks = {
    "near": tgt_r < 0.30,
    "mid": (tgt_r >= 0.30) & (tgt_r <= 1.70),
    "far": tgt_r > 1.70
}

# Red Neuronal del Actor SAC S1
class S1Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_pi = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, 2)
        
    def forward(self, x):
        h = self.latent_pi(x)
        mu = self.mu(h)
        return torch.tanh(mu)

def load_actor_and_norm(model_zip_path, norm_pkl_path):
    with zipfile.ZipFile(model_zip_path, 'r') as z:
        policy_bytes = z.read("policy.pth")
        policy_dict = torch.load(io.BytesIO(policy_bytes), map_location="cpu")
    
    actor = S1Actor()
    state_dict = {
        "latent_pi.0.weight": policy_dict["actor.latent_pi.0.weight"],
        "latent_pi.0.bias": policy_dict["actor.latent_pi.0.bias"],
        "latent_pi.2.weight": policy_dict["actor.latent_pi.2.weight"],
        "latent_pi.2.bias": policy_dict["actor.latent_pi.2.bias"],
        "mu.weight": policy_dict["actor.mu.weight"],
        "mu.bias": policy_dict["actor.mu.bias"]
    }
    actor.load_state_dict(state_dict)
    actor.eval()
    
    with open(norm_pkl_path, "rb") as f:
        norm_obj = pickle.load(f)
    
    obs_mean = norm_obj.obs_rms.mean.astype(np.float32)
    obs_var = norm_obj.obs_rms.var.astype(np.float32)
    epsilon = 1e-8
    clip_obs = 10.0
    
    return actor, obs_mean, obs_var, epsilon, clip_obs

def simulate_validation(actor, obs_mean, obs_var, epsilon, clip_obs, record_episodes=None):
    dt = 0.05
    max_speed = 1.5
    target_radius = 0.06
    velocity_tolerance = 0.10
    hold_steps_req = 20
    max_steps = 200
    L1 = 1.0
    L2 = 1.0
    u_b = 0.40
    n_o = 3
    n_c = 3

    th1 = ((th1_init + np.pi) % (2 * np.pi) - np.pi).copy()
    th2 = ((th2_init + np.pi) % (2 * np.pi) - np.pi).copy()
    dth1 = np.zeros(N_EPISODES, dtype=np.float32)
    dth2 = np.zeros(N_EPISODES, dtype=np.float32)
    
    hold_counter = np.zeros(N_EPISODES, dtype=np.int32)
    hold_counter_max = np.zeros(N_EPISODES, dtype=np.int32)
    is_success = np.zeros(N_EPISODES, dtype=bool)
    steps_to_success = np.full(N_EPISODES, np.nan, dtype=np.float32)
    
    first_entry_step = np.full(N_EPISODES, np.nan, dtype=np.float32)
    joint_speed_at_first_entry = np.full(N_EPISODES, np.nan, dtype=np.float32)
    was_inside_target = np.zeros(N_EPISODES, dtype=bool)
    n_exits = np.zeros(N_EPISODES, dtype=np.int32)
    n_cross = np.zeros(N_EPISODES, dtype=np.int32)
    prev_error_x = np.full(N_EPISODES, np.nan, dtype=np.float32)
    prev_error_y = np.full(N_EPISODES, np.nan, dtype=np.float32)
    
    min_distance = np.full(N_EPISODES, np.inf, dtype=np.float32)
    prev_action = None
    action_delta_sums = np.zeros(N_EPISODES, dtype=np.float32)
    action_delta_counts = np.zeros(N_EPISODES, dtype=np.int32)
    
    episode_active = np.ones(N_EPISODES, dtype=bool)
    final_step_reached = np.zeros(N_EPISODES, dtype=np.int32)
    final_distance = np.zeros(N_EPISODES, dtype=np.float32)
    final_joint_speed = np.zeros(N_EPISODES, dtype=np.float32)

    step_traces = {ep: [] for ep in (record_episodes or [])}

    with torch.no_grad():
        for step in range(1, max_steps + 1):
            if not np.any(episode_active):
                break
                
            x1 = L1 * np.cos(th1)
            y1 = L1 * np.sin(th1)
            ee_x = x1 + L2 * np.cos(th1 + th2)
            ee_y = y1 + L2 * np.sin(th1 + th2)
            dist_x = tgt_x - ee_x
            dist_y = tgt_y - ee_y
            
            raw_obs = np.stack([
                np.cos(th1), np.sin(th1),
                np.cos(th2), np.sin(th2),
                dth1, dth2,
                tgt_x, tgt_y,
                dist_x, dist_y
            ], axis=1)
            
            norm_obs = (raw_obs - obs_mean) / np.sqrt(obs_var + epsilon)
            norm_obs = np.clip(norm_obs, -clip_obs, clip_obs)
            
            obs_tensor = torch.from_numpy(norm_obs)
            action = actor(obs_tensor).numpy()
            action = np.clip(action, -1.0, 1.0)
            
            if prev_action is not None:
                delta_a = action - prev_action
                delta_a_norm = np.linalg.norm(delta_a, axis=1)
                action_delta_sums += delta_a_norm * episode_active
                action_delta_counts += episode_active.astype(np.int32)
            else:
                delta_a_norm = np.zeros(N_EPISODES, dtype=np.float32)
                
            prev_action = action.copy()
            
            dth1 = action[:, 0] * max_speed
            dth2 = action[:, 1] * max_speed
            
            th1 = (th1 + dth1 * dt + np.pi) % (2 * np.pi) - np.pi
            th2 = (th2 + dth2 * dt + np.pi) % (2 * np.pi) - np.pi
            
            x1_new = L1 * np.cos(th1)
            y1_new = L1 * np.sin(th1)
            ee_x_new = x1_new + L2 * np.cos(th1 + th2)
            ee_y_new = y1_new + L2 * np.sin(th1 + th2)
            err_x = tgt_x - ee_x_new
            err_y = tgt_y - ee_y_new
            distance = np.sqrt(err_x**2 + err_y**2)
            joint_speed = np.sqrt(dth1**2 + dth2**2)
            
            for i in range(N_EPISODES):
                if not episode_active[i]:
                    continue
                    
                if distance[i] < min_distance[i]:
                    min_distance[i] = distance[i]
                    
                inside_tgt = (distance[i] < target_radius)
                slow = (joint_speed[i] < velocity_tolerance)
                
                if inside_tgt and np.isnan(first_entry_step[i]):
                    first_entry_step[i] = step
                    joint_speed_at_first_entry[i] = joint_speed[i]
                    
                if not np.isnan(first_entry_step[i]):
                    if was_inside_target[i] and not inside_tgt:
                        n_exits[i] += 1
                was_inside_target[i] = inside_tgt
                
                if distance[i] < 0.30 and not np.isnan(prev_error_x[i]):
                    dot_prod = err_x[i] * prev_error_x[i] + err_y[i] * prev_error_y[i]
                    if dot_prod < 0.0:
                        n_cross[i] += 1
                prev_error_x[i] = err_x[i]
                prev_error_y[i] = err_y[i]
                
                if inside_tgt and slow:
                    hold_counter[i] += 1
                else:
                    hold_counter[i] = 0
                    
                if hold_counter[i] > hold_counter_max[i]:
                    hold_counter_max[i] = hold_counter[i]
                    
                if record_episodes and (i in step_traces):
                    step_traces[i].append({
                        "step": step,
                        "ee_x": float(ee_x_new[i]),
                        "ee_y": float(ee_y_new[i]),
                        "elbow_x": float(x1_new[i]),
                        "elbow_y": float(y1_new[i]),
                        "tgt_x": float(tgt_x[i]),
                        "tgt_y": float(tgt_y[i]),
                        "distance": float(distance[i]),
                        "joint_speed": float(joint_speed[i]),
                        "th1": float(th1[i]),
                        "th2": float(th2[i]),
                        "a1": float(action[i, 0]),
                        "a2": float(action[i, 1]),
                        "delta_a": float(delta_a_norm[i]),
                        "hold_counter": int(hold_counter[i]),
                        "inside": bool(inside_tgt),
                        "slow": bool(slow)
                    })

                if hold_counter[i] >= hold_steps_req:
                    is_success[i] = True
                    steps_to_success[i] = step
                    episode_active[i] = False
                    final_step_reached[i] = step
                    final_distance[i] = distance[i]
                    final_joint_speed[i] = joint_speed[i]
                elif step == max_steps:
                    episode_active[i] = False
                    final_step_reached[i] = step
                    final_distance[i] = distance[i]
                    final_joint_speed[i] = joint_speed[i]

    mean_delta_act = np.where(action_delta_counts > 0, action_delta_sums / np.maximum(action_delta_counts, 1), np.nan)
    never_entered = min_distance >= target_radius
    entered_no_hold = (min_distance < target_radius) & (~is_success)
    truncated = ~is_success
    ballistic_flags = (joint_speed_at_first_entry > u_b) & (~np.isnan(joint_speed_at_first_entry))
    oscillatory_flags = (n_exits >= n_o) | (n_cross >= n_c)

    results_df = pd.DataFrame({
        "episode_id": np.arange(N_EPISODES),
        "target_r": tgt_r,
        "success": is_success,
        "truncated": truncated,
        "never_entered": never_entered,
        "entered_no_hold": entered_no_hold,
        "min_distance": min_distance,
        "final_distance": final_distance,
        "steps_to_first_entry": first_entry_step,
        "steps_to_success": steps_to_success,
        "hold_counter_max": hold_counter_max,
        "joint_speed_at_first_entry": joint_speed_at_first_entry,
        "final_joint_speed": final_joint_speed,
        "action_delta_mean": mean_delta_act,
        "n_exits": n_exits,
        "n_cross": n_cross,
        "ballistic": ballistic_flags,
        "oscillatory": oscillatory_flags,
        "steps_total": final_step_reached
    })
    
    return results_df, step_traces

# 4. Bucle principal de evaluación para todos los checkpoints
SEEDS = [42, 100, 2026]
STEPS = [40960 * i for i in range(1, 17)]

checkpoint_records = []
spatial_records = []
detailed_episodes_data = {} # (seed, step) -> step_traces

print("Iniciando evaluación exhaustiva de 48 configuraciones (3 semillas x 16 checkpoints x 500 episodios)...")

for seed in SEEDS:
    for step in STEPS:
        ckpt_dir = os.path.join(REPO_ROOT, f"experiments/phase_c/S1/seed_{seed}/checkpoints/ckpt_{step}")
        model_zip = os.path.join(ckpt_dir, "model.zip")
        norm_pkl = os.path.join(ckpt_dir, "vecnormalize_stats.pkl")
        
        actor, obs_mean, obs_var, epsilon, clip_obs = load_actor_and_norm(model_zip, norm_pkl)
        
        # Para el mejor checkpoint y el final de cada semilla, grabamos episodios candidatos
        record_eps = None
        if step in [286720, 409600, 655360]:
            # Guardamos episodios específicos para el análisis de trayectorias
            record_eps = [0, 4, 12, 18, 25, 42, 75, 100, 150, 200, 250, 300, 350, 400, 450, 499]
            
        df, traces = simulate_validation(actor, obs_mean, obs_var, epsilon, clip_obs, record_episodes=record_eps)
        if record_eps:
            detailed_episodes_data[(seed, step)] = (df, traces)

        # Métrica agregada checkpoint
        valid_entry = df["steps_to_first_entry"].dropna()
        valid_succ = df["steps_to_success"].dropna()
        valid_speed_entry = df["joint_speed_at_first_entry"].dropna()
        
        ckpt_row = {
            "seed": seed,
            "checkpoint_step": step,
            "n_episodes": N_EPISODES,
            "success_rate": float(df["success"].mean()),
            "truncated_rate": float(df["truncated"].mean()),
            "never_entered_rate": float(df["never_entered"].mean()),
            "entered_no_hold_rate": float(df["entered_no_hold"].mean()),
            "final_distance_mean": float(df["final_distance"].mean()),
            "final_distance_median": float(df["final_distance"].median()),
            "min_distance_mean": float(df["min_distance"].mean()),
            "min_distance_median": float(df["min_distance"].median()),
            "steps_to_first_entry_mean": float(valid_entry.mean()) if len(valid_entry) > 0 else np.nan,
            "steps_to_first_entry_median": float(valid_entry.median()) if len(valid_entry) > 0 else np.nan,
            "steps_to_success_mean": float(valid_succ.mean()) if len(valid_succ) > 0 else np.nan,
            "steps_to_success_median": float(valid_succ.median()) if len(valid_succ) > 0 else np.nan,
            "hold_counter_max_mean": float(df["hold_counter_max"].mean()),
            "hold_counter_max_median": float(df["hold_counter_max"].median()),
            "joint_speed_at_first_entry_mean": float(valid_speed_entry.mean()) if len(valid_speed_entry) > 0 else np.nan,
            "joint_speed_at_final_step_mean": float(df["final_joint_speed"].mean()),
            "action_delta_mean": float(df["action_delta_mean"].mean()),
            "n_exits_mean": float(df["n_exits"].mean()),
            "n_cross_mean": float(df["n_cross"].mean()),
            "ballistic_rate": float(df["ballistic"].mean()),
            "oscillatory_rate": float(df["oscillatory"].mean())
        }
        checkpoint_records.append(ckpt_row)
        
        # Métrica por región espacial
        for reg_name, mask in region_masks.items():
            sub_df = df[mask]
            sub_valid_entry = sub_df["steps_to_first_entry"].dropna()
            sub_valid_speed = sub_df["joint_speed_at_first_entry"].dropna()
            
            spat_row = {
                "seed": seed,
                "checkpoint_step": step,
                "region": reg_name,
                "n_episodes": int(np.sum(mask)),
                "success_rate": float(sub_df["success"].mean()),
                "truncated_rate": float(sub_df["truncated"].mean()),
                "never_entered_rate": float(sub_df["never_entered"].mean()),
                "entered_no_hold_rate": float(sub_df["entered_no_hold"].mean()),
                "min_distance_mean": float(sub_df["min_distance"].mean()),
                "final_distance_mean": float(sub_df["final_distance"].mean()),
                "steps_to_first_entry_mean": float(sub_valid_entry.mean()) if len(sub_valid_entry) > 0 else np.nan,
                "hold_counter_max_mean": float(sub_df["hold_counter_max"].mean()),
                "joint_speed_at_first_entry_mean": float(sub_valid_speed.mean()) if len(sub_valid_speed) > 0 else np.nan,
                "action_delta_mean": float(sub_df["action_delta_mean"].mean())
            }
            spatial_records.append(spat_row)

df_checkpoints = pd.DataFrame(checkpoint_records)
df_spatial = pd.DataFrame(spatial_records)

# Guardar CSVs
ckpt_csv_path = os.path.join(OUTPUT_DIR, "metricas_checkpoints.csv")
spat_csv_path = os.path.join(OUTPUT_DIR, "analisis_espacial.csv")
df_checkpoints.to_csv(ckpt_csv_path, index=False)
df_spatial.to_csv(spat_csv_path, index=False)
print(f"[HECHO] Guardado: {ckpt_csv_path} ({len(df_checkpoints)} filas)")
print(f"[HECHO] Guardado: {spat_csv_path} ({len(df_spatial)} filas)")

# 5. Selección y clasificación de episodios representativos
# Regla determinista de selección:
# Categorías:
# 1. Clean success: success=True, delta_a < 0.10, n_exits==0, n_cross==0
# 2. Pathological success: success=True, (ballistic or oscillatory or delta_a > 0.20)
# 3. Entered no hold: entered_no_hold=True, hold_counter_max between 1 and 19
# 4. No entry: never_entered=True, min_distance >= 0.06
# 5. Truncated with approach: min_distance < 0.20, success=False
# 6. Oscillation/overshoot: n_cross >= 2 or n_exits >= 2

# Buscaremos episodios del checkpoint final y del mejor checkpoint
trajectory_records = []

# Selección en Seed 42: Best (286720) y Final (655360)
# Selección en Seed 2026: Best (409600) y Final (655360)
# Selección en Seed 100: Final (655360)

pairs_to_analyze = [
    (42, 286720, "Best Seed 42"),
    (42, 655360, "Final Seed 42"),
    (100, 655360, "Final/Best Seed 100"),
    (2026, 409600, "Best Seed 2026"),
    (2026, 655360, "Final Seed 2026")
]

selected_traces_dict = {}

for seed, step, label in pairs_to_analyze:
    df_eval, traces = detailed_episodes_data[(seed, step)]
    
    # Clean success
    cand_clean = df_eval[df_eval["success"] & (df_eval["action_delta_mean"] < 0.12) & (df_eval["n_exits"] == 0) & (df_eval["n_cross"] == 0)]
    ep_clean = cand_clean.index[0] if len(cand_clean) > 0 else df_eval[df_eval["success"]].index[0]
    
    # Pathological success
    cand_path = df_eval[df_eval["success"] & ((df_eval["ballistic"]) | (df_eval["n_cross"] >= 1) | (df_eval["action_delta_mean"] > 0.18))]
    ep_path = cand_path.index[0] if len(cand_path) > 0 else None
    
    # Entered no hold
    cand_enh = df_eval[df_eval["entered_no_hold"]]
    ep_enh = cand_enh.index[0] if len(cand_enh) > 0 else None
    
    # No entry
    cand_no_entry = df_eval[df_eval["never_entered"]]
    ep_no_entry = cand_no_entry.index[0] if len(cand_no_entry) > 0 else None
    
    # Truncated approach
    cand_appr = df_eval[(~df_eval["success"]) & (df_eval["min_distance"] < 0.15)]
    ep_appr = cand_appr.index[0] if len(cand_appr) > 0 else None
    
    # Oscillation
    cand_osc = df_eval[(df_eval["n_cross"] >= 2) | (df_eval["n_exits"] >= 2)]
    ep_osc = cand_osc.index[0] if len(cand_osc) > 0 else None
    
    cats = [
        ("clean_success", ep_clean, "Aproximación suave y retención estable sin perturbaciones"),
        ("pathological_success", ep_path, "Éxito con overshoot/vibración o entrada balística"),
        ("entered_no_hold", ep_enh, "Entra al radio de 0.06m pero velocidad > 0.10 rad/s o rebote"),
        ("no_entry", ep_no_entry, "No alcanza el radio de 0.06m durante los 200 pasos"),
        ("truncated_with_approach", ep_appr, "Se aproxima (< 0.15m) pero termina por límite de pasos"),
        ("oscillation_or_overshoot", ep_osc, "Múltiples cruces de signo de error o salidas del radio")
    ]
    
    for cat_name, ep_idx, failure_desc in cats:
        if ep_idx is None:
            continue
        row_ep = df_eval.loc[ep_idx]
        trajectory_records.append({
            "seed": seed,
            "checkpoint_step": step,
            "checkpoint_role": label,
            "episode_id": int(ep_idx),
            "category": cat_name,
            "target_x": float(tgt_x[ep_idx]),
            "target_y": float(tgt_y[ep_idx]),
            "target_r": float(tgt_r[ep_idx]),
            "success": bool(row_ep["success"]),
            "steps_total": int(row_ep["steps_total"]),
            "min_distance": float(row_ep["min_distance"]),
            "final_distance": float(row_ep["final_distance"]),
            "entry_speed": float(row_ep["joint_speed_at_first_entry"]),
            "final_speed": float(row_ep["final_joint_speed"]),
            "action_delta_mean": float(row_ep["action_delta_mean"]),
            "hold_counter_max": int(row_ep["hold_counter_max"]),
            "n_exits": int(row_ep["n_exits"]),
            "n_cross": int(row_ep["n_cross"]),
            "failure_classification": failure_desc
        })

df_trajectories = pd.DataFrame(trajectory_records).drop_duplicates(subset=["seed", "checkpoint_step", "episode_id"])
traj_csv_path = os.path.join(OUTPUT_DIR, "metricas_trayectorias.csv")
df_trajectories.to_csv(traj_csv_path, index=False)
print(f"[HECHO] Guardado: {traj_csv_path} ({len(df_trajectories)} filas)")

# 6. Extracción de TensorBoard Logs
tb_data = {}
for seed in SEEDS:
    log_dir = os.path.join(REPO_ROOT, f"experiments/phase_c/S1/seed_{seed}/logs/SAC_1")
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags()['scalars']
    tb_data[seed] = {}
    for tag in ["train/actor_loss", "train/critic_loss", "train/ent_coef", "train/ent_coef_loss"]:
        if tag in tags:
            events = ea.Scalars(tag)
            tb_data[seed][tag] = {
                "step": [e.step for e in events],
                "val": [e.value for e in events]
            }
print("[HECHO] Datos de TensorBoard extraídos para todas las semillas.")

# 7. GENERACIÓN DE FIGURAS OBLIGATORIAS (1 a 11)

# Agrupación por step para media y desviación estándar entre semillas
step_groups = df_checkpoints.groupby("checkpoint_step")
mean_succ = step_groups["success_rate"].mean()
std_succ = step_groups["success_rate"].std(ddof=1)
mean_trunc = step_groups["truncated_rate"].mean()
std_trunc = step_groups["truncated_rate"].std(ddof=1)
mean_enh = step_groups["entered_no_hold_rate"].mean()
std_enh = step_groups["entered_no_hold_rate"].std(ddof=1)
mean_ne = step_groups["never_entered_rate"].mean()
std_ne = step_groups["never_entered_rate"].std(ddof=1)
mean_act = step_groups["action_delta_mean"].mean()
std_act = step_groups["action_delta_mean"].std(ddof=1)
steps_axis = mean_succ.index.values

# FIGURA 1: evolucion_exito.png
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
colors = {42: "#1f77b4", 100: "#2ca02c", 2026: "#ff7f0e"}
for seed in SEEDS:
    sub = df_checkpoints[df_checkpoints["seed"] == seed]
    ax.plot(sub["checkpoint_step"], sub["success_rate"] * 100, marker='o', markersize=4, label=f"Semilla {seed}", color=colors[seed], alpha=0.7)

ax.plot(steps_axis, mean_succ * 100, color="black", linewidth=2.5, label="Media entre semillas")
ax.fill_between(steps_axis, (mean_succ - std_succ) * 100, (mean_succ + std_succ) * 100, color="black", alpha=0.15, label="Desv. estándar (ddof=1)")
ax.set_title("Evolución Temporal del Éxito Estricto — SAC S1 (Validación C, 500 eps)", fontsize=12, fontweight="bold")
ax.set_xlabel("Pasos de Entrenamiento (Entorno)", fontsize=10)
ax.set_ylabel("Tasa de Éxito Estricto (%)", fontsize=10)
ax.set_ylim(0, 75)
ax.grid(True)
ax.legend(frameon=True, facecolor="white", loc="lower left")
plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, "evolucion_exito.png")
fig.savefig(fig1_path)
plt.close(fig)
print(f"[HECHO] Figura 1: {fig1_path}")

# FIGURA 2: evolucion_truncamiento.png
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
for seed in SEEDS:
    sub = df_checkpoints[df_checkpoints["seed"] == seed]
    ax.plot(sub["checkpoint_step"], sub["truncated_rate"] * 100, marker='s', markersize=4, label=f"Semilla {seed}", color=colors[seed], alpha=0.7)

ax.plot(steps_axis, mean_trunc * 100, color="black", linewidth=2.5, label="Media entre semillas")
ax.fill_between(steps_axis, (mean_trunc - std_trunc) * 100, (mean_trunc + std_trunc) * 100, color="black", alpha=0.15)
ax.set_title("Evolución del Truncamiento — SAC S1 (Validación C, 500 eps)", fontsize=12, fontweight="bold")
ax.set_xlabel("Pasos de Entrenamiento", fontsize=10)
ax.set_ylabel("Tasa de Truncamiento (%)", fontsize=10)
ax.set_ylim(25, 100)
ax.grid(True)
ax.legend(frameon=True, facecolor="white")
plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, "evolucion_truncamiento.png")
fig.savefig(fig2_path)
plt.close(fig)
print(f"[HECHO] Figura 2: {fig2_path}")

# FIGURA 3: comparacion_semillas.png (4 paneles)
fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
# (0,0) Exito
for seed in SEEDS:
    sub = df_checkpoints[df_checkpoints["seed"] == seed]
    axs[0, 0].plot(sub["checkpoint_step"], sub["success_rate"] * 100, marker='o', label=f"Semilla {seed}", color=colors[seed])
axs[0, 0].set_title("Tasa de Éxito Estricto (%)", fontweight="bold")
axs[0, 0].set_ylabel("Éxito (%)")
axs[0, 0].grid(True)
axs[0, 0].legend()

# (0,1) Entered No Hold
for seed in SEEDS:
    sub = df_checkpoints[df_checkpoints["seed"] == seed]
    axs[0, 1].plot(sub["checkpoint_step"], sub["entered_no_hold_rate"] * 100, marker='o', label=f"Semilla {seed}", color=colors[seed])
axs[0, 1].set_title("Entra al Radio sin Hold (%)", fontweight="bold")
axs[0, 1].set_ylabel("Entrada sin Hold (%)")
axs[0, 1].grid(True)
axs[0, 1].legend()

# (1,0) Never Entered
for seed in SEEDS:
    sub = df_checkpoints[df_checkpoints["seed"] == seed]
    axs[1, 0].plot(sub["checkpoint_step"], sub["never_entered_rate"] * 100, marker='o', label=f"Semilla {seed}", color=colors[seed])
axs[1, 0].set_title("Nunca Entra al Radio (%)", fontweight="bold")
axs[1, 0].set_ylabel("Nunca Entra (%)")
axs[1, 0].set_xlabel("Pasos de Entrenamiento")
axs[1, 0].grid(True)
axs[1, 0].legend()

# (1,1) Action Delta
for seed in SEEDS:
    sub = df_checkpoints[df_checkpoints["seed"] == seed]
    axs[1, 1].plot(sub["checkpoint_step"], sub["action_delta_mean"], marker='o', label=f"Semilla {seed}", color=colors[seed])
axs[1, 1].set_title("Variación Media de Acción ||Δa||", fontweight="bold")
axs[1, 1].set_ylabel("||Δa||")
axs[1, 1].set_xlabel("Pasos de Entrenamiento")
axs[1, 1].grid(True)
axs[1, 1].legend()

fig.suptitle("Comparativa de Desempeño entre las Tres Semillas de SAC S1", fontsize=14, fontweight="bold")
plt.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, "comparacion_semillas.png")
fig.savefig(fig3_path)
plt.close(fig)
print(f"[HECHO] Figura 3: {fig3_path}")

# FIGURA 4: comparacion_mejor_final.png
# Best checkpoints: Seed 42: 286720, Seed 100: 655360, Seed 2026: 409600
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
labels = ["Semilla 42", "Semilla 100", "Semilla 2026", "Media Semillas"]
best_vals = [
    df_checkpoints[(df_checkpoints["seed"]==42) & (df_checkpoints["checkpoint_step"]==286720)]["success_rate"].values[0] * 100,
    df_checkpoints[(df_checkpoints["seed"]==100) & (df_checkpoints["checkpoint_step"]==655360)]["success_rate"].values[0] * 100,
    df_checkpoints[(df_checkpoints["seed"]==2026) & (df_checkpoints["checkpoint_step"]==409600)]["success_rate"].values[0] * 100
]
best_vals.append(np.mean(best_vals))

final_vals = [
    df_checkpoints[(df_checkpoints["seed"]==42) & (df_checkpoints["checkpoint_step"]==655360)]["success_rate"].values[0] * 100,
    df_checkpoints[(df_checkpoints["seed"]==100) & (df_checkpoints["checkpoint_step"]==655360)]["success_rate"].values[0] * 100,
    df_checkpoints[(df_checkpoints["seed"]==2026) & (df_checkpoints["checkpoint_step"]==655360)]["success_rate"].values[0] * 100
]
final_vals.append(np.mean(final_vals))

x = np.arange(len(labels))
width = 0.35
rects1 = ax.bar(x - width/2, best_vals, width, label='Mejor Checkpoint (Pico)', color='#2b5c8f')
rects2 = ax.bar(x + width/2, final_vals, width, label='Checkpoint Final (Paso 655 360)', color='#d95f02')

ax.set_ylabel('Tasa de Éxito Estricto (%)', fontsize=11)
ax.set_title('Comparativa entre Mejor Checkpoint Histórico y Checkpoint Final', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.legend(frameon=True, facecolor="white")
ax.grid(axis='y')

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, "comparacion_mejor_final.png")
fig.savefig(fig4_path)
plt.close(fig)
print(f"[HECHO] Figura 4: {fig4_path}")

# FIGURA 5: rendimiento_espacial.png
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
spat_group = df_spatial.groupby(["checkpoint_step", "region"])["success_rate"].mean().unstack()
ax.plot(spat_group.index, spat_group["near"] * 100, marker='o', label="Cercano (r < 0.30 m, n=45)", color="#e41a1c", linewidth=2)
ax.plot(spat_group.index, spat_group["mid"] * 100, marker='s', label="Intermedio (0.30 ≤ r ≤ 1.70 m, n=401)", color="#377eb8", linewidth=2)
ax.plot(spat_group.index, spat_group["far"] * 100, marker='^', label="Periférico (r > 1.70 m, n=54)", color="#4daf4a", linewidth=2)

ax.set_title("Evolución Temporal del Éxito Estricto por Región Radial (Media 3 Semillas)", fontsize=12, fontweight="bold")
ax.set_xlabel("Pasos de Entrenamiento", fontsize=10)
ax.set_ylabel("Tasa de Éxito Estricto (%)", fontsize=10)
ax.grid(True)
ax.legend(frameon=True, facecolor="white")
plt.tight_layout()
fig5_path = os.path.join(FIGURES_DIR, "rendimiento_espacial.png")
fig.savefig(fig5_path)
plt.close(fig)
print(f"[HECHO] Figura 5: {fig5_path}")

# FIGURA 6: mapa_exitos_fallos.png (Scatter 2D de objetivos evaluados Best vs Final Seed 42)
df_eval_best, _ = detailed_episodes_data[(42, 286720)]
df_eval_final, _ = detailed_episodes_data[(42, 655360)]

fig, axs = plt.subplots(1, 2, figsize=(13, 6), dpi=150)

def plot_map(ax, df_e, title):
    # Círculos de límites de alcance
    circle_reach = patches.Circle((0, 0), 2.0, fill=False, color='grey', linestyle='--', label="Alcance max (2.0m)")
    circle_near = patches.Circle((0, 0), 0.3, fill=False, color='orange', linestyle=':', label="Zona cercana (0.3m)")
    circle_mid = patches.Circle((0, 0), 1.7, fill=False, color='blue', linestyle=':', label="Zona periferia (1.7m)")
    ax.add_patch(circle_reach)
    ax.add_patch(circle_near)
    ax.add_patch(circle_mid)
    
    succ = df_e["success"]
    enh = df_e["entered_no_hold"]
    ne = df_e["never_entered"]
    
    ax.scatter(tgt_x[succ], tgt_y[succ], c="#2ca02c", s=18, alpha=0.8, label=f"Éxito estricto ({succ.sum()})")
    ax.scatter(tgt_x[enh], tgt_y[enh], c="#ff7f0e", s=18, alpha=0.8, label=f"Entra sin Hold ({enh.sum()})")
    ax.scatter(tgt_x[ne], tgt_y[ne], c="#d62728", s=18, alpha=0.8, label=f"Nunca Entra ({ne.sum()})")
    
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal')
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=8)

plot_map(axs[0], df_eval_best, "Semilla 42 — Mejor Checkpoint (Paso 286 720, 60.8% éxito)")
plot_map(axs[1], df_eval_final, "Semilla 42 — Checkpoint Final (Paso 655 360, 40.6% éxito)")

plt.tight_layout()
fig6_path = os.path.join(FIGURES_DIR, "mapa_exitos_fallos.png")
fig.savefig(fig6_path)
plt.close(fig)
print(f"[HECHO] Figura 6: {fig6_path}")

# FIGURA 7: trayectorias_representativas.png (2D End-Effector Paths)
# Reconstruiremos 4 casos representativos desde Seed 42 Checkpoint Final:
# Ep 0 (clean success o intento), Ep 4 (no entry), Ep 18 (entered no hold), Ep 25 (overshoot/oscillation)
_, traces_final_42 = detailed_episodes_data[(42, 655360)]

fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=150)
eps_to_plot = [
    (0, "Episodio 0: Aproximación y estabilización", axs[0, 0]),
    (18, "Episodio 18: Entra al radio pero rebota/sale", axs[0, 1]),
    (4, "Episodio 4: No entra (periferia)", axs[1, 0]),
    (25, "Episodio 25: Oscilación y saturación de acción", axs[1, 1])
]

for ep_id, ep_title, ax in eps_to_plot:
    tr = traces_final_42[ep_id]
    x_steps = [s["ee_x"] for s in tr]
    y_steps = [s["ee_y"] for s in tr]
    tx, ty = tr[0]["tgt_x"], tr[0]["tgt_y"]
    
    # Dibujar radio objetivo
    tgt_circ = patches.Circle((tx, ty), 0.06, color='#2ca02c', alpha=0.3, label="Radio Objetivo (0.06m)")
    ax.add_patch(tgt_circ)
    ax.plot(x_steps, y_steps, color="#1f77b4", linewidth=1.5, label="Trayectoria Efector")
    ax.scatter([x_steps[0]], [y_steps[0]], color="black", s=30, label="Inicio")
    ax.scatter([x_steps[-1]], [y_steps[-1]], color="red", s=30, label="Final")
    ax.scatter([tx], [ty], color="green", marker="x", s=50, label="Objetivo")
    
    ax.set_title(ep_title, fontsize=10, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True)
    ax.legend(fontsize=7, loc="best")
    ax.set_aspect('equal', 'datalim')

plt.tight_layout()
fig7_path = os.path.join(FIGURES_DIR, "trayectorias_representativas.png")
fig.savefig(fig7_path)
plt.close(fig)
print(f"[HECHO] Figura 7: {fig7_path}")

# FIGURA 8: distancia_temporal.png (d(t) para episodios representativos)
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
for ep_id, label, col in [(0, "Episodio 0 (Éxito)", "#2ca02c"),
                          (18, "Episodio 18 (Entra sin hold)", "#ff7f0e"),
                          (4, "Episodio 4 (Nunca entra)", "#d62728"),
                          (25, "Episodio 25 (Oscilatorio)", "#9467bd")]:
    tr = traces_final_42[ep_id]
    t = [s["step"] for s in tr]
    d = [s["distance"] for s in tr]
    ax.plot(t, d, label=label, color=col, linewidth=1.8)

ax.axhline(0.06, color="black", linestyle="--", linewidth=1.2, label="Radio Objetivo (0.06 m)")
ax.set_title("Evolución Temporal de la Distancia al Objetivo d(t)", fontsize=12, fontweight="bold")
ax.set_xlabel("Paso Temporal (dt = 0.05 s)", fontsize=10)
ax.set_ylabel("Distancia al Objetivo (m)", fontsize=10)
ax.set_ylim(0, 1.5)
ax.grid(True)
ax.legend(frameon=True, facecolor="white")
plt.tight_layout()
fig8_path = os.path.join(FIGURES_DIR, "distancia_temporal.png")
fig.savefig(fig8_path)
plt.close(fig)
print(f"[HECHO] Figura 8: {fig8_path}")

# FIGURA 9: velocidad_articular.png (||dtheta(t)||)
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
for ep_id, label, col in [(0, "Episodio 0 (Éxito)", "#2ca02c"),
                          (18, "Episodio 18 (Entra sin hold)", "#ff7f0e"),
                          (25, "Episodio 25 (Oscilatorio)", "#9467bd")]:
    tr = traces_final_42[ep_id]
    t = [s["step"] for s in tr]
    v = [s["joint_speed"] for s in tr]
    ax.plot(t, v, label=label, color=col, linewidth=1.8)

ax.axhline(0.10, color="red", linestyle="--", linewidth=1.5, label="Tolerancia Estricta de Velocidad (0.10 rad/s)")
ax.axhline(0.30, color="orange", linestyle=":", linewidth=1.2, label="Tolerancia Entrenamiento (0.30 rad/s)")
ax.set_title("Evolución Temporal de la Velocidad Articular ||dθ/dt||", fontsize=12, fontweight="bold")
ax.set_xlabel("Paso Temporal", fontsize=10)
ax.set_ylabel("Velocidad Articular (rad/s)", fontsize=10)
ax.grid(True)
ax.legend(frameon=True, facecolor="white")
plt.tight_layout()
fig9_path = os.path.join(FIGURES_DIR, "velocidad_articular.png")
fig.savefig(fig9_path)
plt.close(fig)
print(f"[HECHO] Figura 9: {fig9_path}")

# FIGURA 10: acciones_temporales.png (Chattering / Acciones de la política)
fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True, dpi=150)
tr_25 = traces_final_42[25]
t_25 = [s["step"] for s in tr_25]
a1 = [s["a1"] for s in tr_25]
a2 = [s["a2"] for s in tr_25]
delta_a = [s["delta_a"] for s in tr_25]

axs[0].plot(t_25, a1, label="Acción a1 (Hombro)", color="#1f77b4")
axs[0].plot(t_25, a2, label="Acción a2 (Codo)", color="#ff7f0e")
axs[0].set_ylabel("Acción normalizada [-1, 1]")
axs[0].set_title("Acciones de la Política en Episodio con Deterioro (Episodio 25)", fontweight="bold")
axs[0].grid(True)
axs[0].legend(loc="upper right")

axs[1].plot(t_25, delta_a, label="||Δa_t|| = ||a_t - a_{t-1}||", color="#d62728")
axs[1].set_ylabel("Variación ||Δa||")
axs[1].set_xlabel("Paso Temporal")
axs[1].grid(True)
axs[1].legend(loc="upper right")

plt.tight_layout()
fig10_path = os.path.join(FIGURES_DIR, "acciones_temporales.png")
fig.savefig(fig10_path)
plt.close(fig)
print(f"[HECHO] Figura 10: {fig10_path}")

# FIGURA 11: indicadores_entrenamiento.png (TensorBoard scalars 4 paneles)
fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

# (0, 0) Actor Loss
for seed in SEEDS:
    if "train/actor_loss" in tb_data[seed]:
        d = tb_data[seed]["train/actor_loss"]
        axs[0, 0].plot(d["step"], d["val"], label=f"Semilla {seed}", color=colors[seed], alpha=0.6)
axs[0, 0].set_title("Pérdida del Actor (Actor Loss)", fontweight="bold")
axs[0, 0].set_ylabel("Loss")
axs[0, 0].grid(True)
axs[0, 0].legend()

# (0, 1) Critic Loss
for seed in SEEDS:
    if "train/critic_loss" in tb_data[seed]:
        d = tb_data[seed]["train/critic_loss"]
        axs[0, 1].plot(d["step"], d["val"], label=f"Semilla {seed}", color=colors[seed], alpha=0.6)
axs[0, 1].set_title("Pérdida de los Críticos (Critic Loss)", fontweight="bold")
axs[0, 1].set_ylabel("Loss")
axs[0, 1].grid(True)
axs[0, 1].legend()

# (1, 0) Coeficiente de Entropía Efectivo (alpha)
for seed in SEEDS:
    if "train/ent_coef" in tb_data[seed]:
        d = tb_data[seed]["train/ent_coef"]
        axs[1, 0].plot(d["step"], d["val"], label=f"Semilla {seed}", color=colors[seed], alpha=0.8)
axs[1, 0].set_title("Coeficiente de Entropía Efectivo (α)", fontweight="bold")
axs[1, 0].set_ylabel("α")
axs[1, 0].set_xlabel("Pasos de Entrenamiento")
axs[1, 0].grid(True)
axs[1, 0].legend()

# (1, 1) Ent Coef Loss
for seed in SEEDS:
    if "train/ent_coef_loss" in tb_data[seed]:
        d = tb_data[seed]["train/ent_coef_loss"]
        axs[1, 1].plot(d["step"], d["val"], label=f"Semilla {seed}", color=colors[seed], alpha=0.6)
axs[1, 1].set_title("Pérdida del Coeficiente de Entropía", fontweight="bold")
axs[1, 1].set_ylabel("Loss")
axs[1, 1].set_xlabel("Pasos de Entrenamiento")
axs[1, 1].grid(True)
axs[1, 1].legend()

fig.suptitle("Dinámica Interna del Entrenamiento de SAC S1 (TensorBoard)", fontsize=14, fontweight="bold")
plt.tight_layout()
fig11_path = os.path.join(FIGURES_DIR, "indicadores_entrenamiento.png")
fig.savefig(fig11_path)
plt.close(fig)
print(f"[HECHO] Figura 11: {fig11_path}")

print("=== DIAGNÓSTICO INTEGRAL COMPLETADO EXITOSAMENTE ===")
