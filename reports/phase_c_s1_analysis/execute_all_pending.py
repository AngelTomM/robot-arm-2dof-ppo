"""
Ejecucion exhaustiva y reproducible de los 5 analisis pendientes de SAC S1.
Solo lectura de artefactos originales. Salidas en reports/phase_c_s1_analysis/.
"""

import os
import json
import hashlib
import zipfile
import io
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "reports/phase_c_s1_analysis")
VAL_DATASET_PATH = os.path.join(REPO_ROOT, "data/datasets/phase_c/validation_c_500.json")

# Constantes fisicas
L1 = 1.0
L2 = 1.0
DT = 0.05
MAX_SPEED = 1.5
TARGET_RADIUS = 0.06
MAX_STEPS = 200

# Criterios de evaluacion
STRICT_HOLD_STEPS = 20
STRICT_VEL_TOL = 0.10

TRAIN_HOLD_STEPS = 5
TRAIN_VEL_TOL = 0.30

# Cargar dataset de validacion
with open(VAL_DATASET_PATH, 'r') as f:
    val_data = json.load(f)

N_EPISODES = len(val_data)
init_th1 = np.array([ep["theta1_init"] for ep in val_data], dtype=np.float32)
init_th2 = np.array([ep["theta2_init"] for ep in val_data], dtype=np.float32)
tgt_x = np.array([ep["target_x"] for ep in val_data], dtype=np.float32)
tgt_y = np.array([ep["target_y"] for ep in val_data], dtype=np.float32)

# --- Clases de modelos ---

class SACActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_pi = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, 2)
        self.log_std = nn.Linear(256, 2)

    def forward(self, obs):
        latent = self.latent_pi(obs)
        mean_actions = self.mu(latent)
        log_std = self.log_std(latent)
        log_std = torch.clamp(log_std, -20, 2)
        return torch.tanh(mean_actions), torch.exp(log_std)

def load_sac_model(model_zip_path):
    with zipfile.ZipFile(model_zip_path, 'r') as z:
        with z.open('policy.pth') as f:
            state_dict = torch.load(f, map_location='cpu')

    actor = SACActor()
    actor_state = {
        'latent_pi.0.weight': state_dict['actor.latent_pi.0.weight'],
        'latent_pi.0.bias': state_dict['actor.latent_pi.0.bias'],
        'latent_pi.2.weight': state_dict['actor.latent_pi.2.weight'],
        'latent_pi.2.bias': state_dict['actor.latent_pi.2.bias'],
        'mu.weight': state_dict['actor.mu.weight'],
        'mu.bias': state_dict['actor.mu.bias'],
        'log_std.weight': state_dict['actor.log_std.weight'],
        'log_std.bias': state_dict['actor.log_std.bias']
    }
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor

class PPOActor(nn.Module):
    def __init__(self, h1, h2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 2)
        )

    def forward(self, obs):
        return self.net(obs)

def load_ppo_model(model_zip_path):
    with zipfile.ZipFile(model_zip_path, 'r') as z:
        with z.open('policy.pth') as f:
            state_dict = torch.load(f, map_location='cpu')

    pi_w0 = state_dict['mlp_extractor.policy_net.0.weight']
    h1 = pi_w0.shape[0]
    pi_w2 = state_dict['mlp_extractor.policy_net.2.weight']
    h2 = pi_w2.shape[0]

    policy = PPOActor(h1, h2)
    p_state = {
        'net.0.weight': state_dict['mlp_extractor.policy_net.0.weight'],
        'net.0.bias': state_dict['mlp_extractor.policy_net.0.bias'],
        'net.2.weight': state_dict['mlp_extractor.policy_net.2.weight'],
        'net.2.bias': state_dict['mlp_extractor.policy_net.2.bias'],
        'net.4.weight': state_dict['action_net.weight'],
        'net.4.bias': state_dict['action_net.bias']
    }
    policy.load_state_dict(p_state)
    policy.eval()
    return policy

def load_norm(pkl_path):
    with open(pkl_path, 'rb') as f:
        vn = pickle.load(f)
    return vn.obs_rms.mean, vn.obs_rms.var, vn.epsilon, vn.clip_obs

def run_simulation(policy, obs_mean, obs_var, epsilon, clip_obs, is_sac=True,
                   hold_steps_req=20, vel_tol_req=0.10, collect_all_traces=False):
    obs_std = np.sqrt(obs_var + epsilon)

    th1 = init_th1.copy()
    th2 = init_th2.copy()
    dth1 = np.zeros(N_EPISODES, dtype=np.float32)
    dth2 = np.zeros(N_EPISODES, dtype=np.float32)

    is_success = np.zeros(N_EPISODES, dtype=bool)
    steps_to_success = np.full(N_EPISODES, np.nan)
    hold_counter = np.zeros(N_EPISODES, dtype=np.int32)
    hold_counter_max = np.zeros(N_EPISODES, dtype=np.int32)
    episode_active = np.ones(N_EPISODES, dtype=bool)

    min_distance = np.full(N_EPISODES, np.inf)
    first_entry_step = np.full(N_EPISODES, np.nan)

    prev_actions = np.zeros((N_EPISODES, 2), dtype=np.float32)
    has_prev_action = np.zeros(N_EPISODES, dtype=bool)
    action_delta_sums = np.zeros(N_EPISODES, dtype=np.float32)
    action_delta_counts = np.zeros(N_EPISODES, dtype=np.int32)

    n_exits = np.zeros(N_EPISODES, dtype=np.int32)
    was_inside = np.zeros(N_EPISODES, dtype=bool)

    # Sigmas registradas
    sigmas_history = []

    traces = {i: [] for i in range(N_EPISODES)} if collect_all_traces else None

    for step in range(1, MAX_STEPS + 1):
        if not np.any(episode_active):
            break

        x1 = L1 * np.cos(th1)
        y1 = L1 * np.sin(th1)
        ee_x = x1 + L2 * np.cos(th1 + th2)
        ee_y = y1 + L2 * np.sin(th1 + th2)

        dx = tgt_x - ee_x
        dy = tgt_y - ee_y
        dist = np.sqrt(dx**2 + dy**2)
        min_distance = np.minimum(min_distance, dist)

        obs_raw = np.column_stack([
            np.cos(th1), np.sin(th1),
            np.cos(th2), np.sin(th2),
            dth1, dth2,
            tgt_x, tgt_y,
            dx, dy
        ])
        obs_norm = np.clip((obs_raw - obs_mean) / obs_std, -clip_obs, clip_obs)

        with torch.no_grad():
            t_obs = torch.from_numpy(obs_norm.astype(np.float32))
            if is_sac:
                actions_t, std_t = policy(t_obs)
                action = actions_t.numpy()
                sigmas_history.append(std_t.numpy())
            else:
                action = policy(t_obs).numpy()
                action = np.clip(action, -1.0, 1.0)

        delta_act = np.linalg.norm(action - prev_actions, axis=1)
        valid_delta = has_prev_action & episode_active
        action_delta_sums[valid_delta] += delta_act[valid_delta]
        action_delta_counts[valid_delta] += 1
        prev_actions = action.copy()
        has_prev_action[episode_active] = True

        dth1 = action[:, 0] * MAX_SPEED
        dth2 = action[:, 1] * MAX_SPEED
        joint_speed = np.sqrt(dth1**2 + dth2**2)

        th1_new = (th1 + dth1 * DT + np.pi) % (2 * np.pi) - np.pi
        th2_new = (th2 + dth2 * DT + np.pi) % (2 * np.pi) - np.pi

        x1_new = L1 * np.cos(th1_new)
        y1_new = L1 * np.sin(th1_new)
        ee_x_new = x1_new + L2 * np.cos(th1_new + th2_new)
        ee_y_new = y1_new + L2 * np.sin(th1_new + th2_new)
        new_dist = np.sqrt((tgt_x - ee_x_new)**2 + (tgt_y - ee_y_new)**2)
        min_distance = np.minimum(min_distance, new_dist)

        inside_tgt = new_dist < TARGET_RADIUS
        slow_enough = joint_speed < vel_tol_req

        for i in range(N_EPISODES):
            if not episode_active[i]:
                continue

            if inside_tgt[i] and np.isnan(first_entry_step[i]):
                first_entry_step[i] = step

            if was_inside[i] and (not inside_tgt[i]):
                n_exits[i] += 1
            was_inside[i] = inside_tgt[i]

            if inside_tgt[i] and slow_enough[i]:
                hold_counter[i] += 1
            else:
                hold_counter[i] = 0

            hold_counter_max[i] = max(hold_counter_max[i], hold_counter[i])

            if collect_all_traces:
                traces[i].append({
                    "step": step,
                    "dist": float(new_dist[i]),
                    "joint_speed": float(joint_speed[i]),
                    "dth1": float(dth1[i]),
                    "dth2": float(dth2[i]),
                    "a1": float(action[i, 0]),
                    "a2": float(action[i, 1]),
                    "delta_a": float(delta_act[i]) if has_prev_action[i] else 0.0,
                    "hold": int(hold_counter[i]),
                    "inside": bool(inside_tgt[i]),
                    "slow": bool(slow_enough[i])
                })

            if hold_counter[i] >= hold_steps_req:
                is_success[i] = True
                steps_to_success[i] = step
                episode_active[i] = False
            elif step == MAX_STEPS:
                episode_active[i] = False

        th1 = th1_new
        th2 = th2_new

    never_entered = min_distance >= TARGET_RADIUS
    entered_no_hold = (min_distance < TARGET_RADIUS) & (~is_success)
    mean_delta_act = np.where(action_delta_counts > 0, action_delta_sums / np.maximum(action_delta_counts, 1), np.nan)

    res = {
        "success_rate": float(np.mean(is_success)),
        "truncated_rate": float(np.mean(~is_success)),
        "never_entered_rate": float(np.mean(never_entered)),
        "entered_no_hold_rate": float(np.mean(entered_no_hold)),
        "action_delta_mean": float(np.nanmean(mean_delta_act)),
        "n_exits_mean": float(np.mean(n_exits)),
        "min_distance_mean": float(np.mean(min_distance)),
        "is_success": is_success,
        "never_entered": never_entered,
        "entered_no_hold": entered_no_hold,
        "action_delta_per_ep": mean_delta_act,
        "n_exits": n_exits,
        "traces": traces,
        "sigmas": np.vstack(sigmas_history) if len(sigmas_history) > 0 else None
    }
    return res

print("[INICIO] Ejecutando los 5 analisis...")

# ==============================================================================
# 4.1. PERFILES TEMPORALES DE FALLOS
# ==============================================================================
print("\n--- 4.1. Perfiles Temporales de Fallos ---")
# Evaluamos Checkpoint 655360 con trazado completo para seleccionar con hash determinista
sample_profiles = []

for seed in [42, 100, 2026]:
    ckpt_dir = os.path.join(REPO_ROOT, f"experiments/phase_c/S1/seed_{seed}/checkpoints/ckpt_655360")
    policy = load_sac_model(os.path.join(ckpt_dir, "model.zip"))
    m, v, eps, clip_o = load_norm(os.path.join(ckpt_dir, "vecnormalize_stats.pkl"))
    sim = run_simulation(policy, m, v, eps, clip_o, is_sac=True,
                         hold_steps_req=STRICT_HOLD_STEPS, vel_tol_req=STRICT_VEL_TOL,
                         collect_all_traces=True)

    # Regla de seleccion fija antes de inspeccionar: menor hash sha256(seed_step_cat_epId)
    for cat, mask in [("entered_no_hold", sim["entered_no_hold"]), ("never_entered", sim["never_entered"])]:
        cands = np.where(mask)[0]
        if len(cands) == 0:
            continue
        # Seleccionar ep determinista con menor hash
        hashes = [hashlib.sha256(f"{seed}_655360_{cat}_{ep}".encode('utf-8')).hexdigest() for ep in cands]
        best_idx = np.argmin(hashes)
        sel_ep = int(cands[best_idx])
        sel_hash = hashes[best_idx][:8]

        trace = sim["traces"][sel_ep]
        for tr in trace:
            sample_profiles.append({
                "seed": seed,
                "checkpoint": 655360,
                "category": cat,
                "episode_id": sel_ep,
                "hash_id": sel_hash,
                "step": tr["step"],
                "distance": tr["dist"],
                "joint_speed": tr["joint_speed"],
                "action_1": tr["a1"],
                "action_2": tr["a2"],
                "delta_a": tr["delta_a"],
                "hold_counter": tr["hold"],
                "inside_target": tr["inside"],
                "slow_enough": tr["slow"]
            })
        print(f"  [Seleccion 4.1] Semilla {seed} | Cat: {cat:16s} | Ep: {sel_ep:3d} | Hash: {sel_hash} | Longitud: {len(trace)} pasos")

df_profiles = pd.DataFrame(sample_profiles)
df_profiles_path = os.path.join(OUTPUT_DIR, "perfiles_fallos_seleccionados.csv")
df_profiles.to_csv(df_profiles_path, index=False)
print(f"  -> Guardado: {df_profiles_path} ({len(df_profiles)} filas de pasos)")

# ==============================================================================
# 4.2. AMPLITUD DE OSCILACIÓN EN EL BORDE
# ==============================================================================
print("\n--- 4.2. Amplitud de Oscilacion en el Borde ---")
# Banda espacial: d in [0.04 m, 0.08 m] (r_tgt = 0.06 +- 0.02)
# Ventana temporal: episodios con >= 10 pasos dentro de la banda
BAND_MIN = 0.04
BAND_MAX = 0.08
MIN_BAND_STEPS = 10

edge_results = []

for seed in [42, 100, 2026]:
    for step in [327680, 655360]:
        ckpt_dir = os.path.join(REPO_ROOT, f"experiments/phase_c/S1/seed_{seed}/checkpoints/ckpt_{step}")
        policy = load_sac_model(os.path.join(ckpt_dir, "model.zip"))
        m, v, eps, clip_o = load_norm(os.path.join(ckpt_dir, "vecnormalize_stats.pkl"))
        sim = run_simulation(policy, m, v, eps, clip_o, is_sac=True,
                             hold_steps_req=STRICT_HOLD_STEPS, vel_tol_req=STRICT_VEL_TOL,
                             collect_all_traces=True)

        n_valid_episodes = 0
        amp_dist_list = []
        std_dist_list = []
        amp_act_list = []
        mean_speed_band_list = []
        crossings_list = []

        for ep in range(N_EPISODES):
            tr = sim["traces"][ep]
            band_steps = [s for s in tr if BAND_MIN <= s["dist"] <= BAND_MAX]
            if len(band_steps) >= MIN_BAND_STEPS:
                n_valid_episodes += 1
                dists = np.array([s["dist"] for s in band_steps])
                speeds = np.array([s["joint_speed"] for s in band_steps])
                delta_as = np.array([s["delta_a"] for s in band_steps[1:]]) if len(band_steps) > 1 else [0.0]

                # Cruces de frontera r=0.06 en la banda
                inside_flags = [s["dist"] < TARGET_RADIUS for s in band_steps]
                crossings = sum(inside_flags[i] != inside_flags[i-1] for i in range(1, len(inside_flags)))

                amp_dist_list.append(np.max(dists) - np.min(dists))
                std_dist_list.append(np.std(dists))
                amp_act_list.append(np.mean(delta_as) if len(delta_as) > 0 else 0.0)
                mean_speed_band_list.append(np.mean(speeds))
                crossings_list.append(crossings)

        edge_results.append({
            "seed": seed,
            "checkpoint_step": step,
            "n_episodes_analyzed": N_EPISODES,
            "n_episodes_in_band": n_valid_episodes,
            "pct_episodes_in_band": (n_valid_episodes / N_EPISODES) * 100,
            "pct_insufficient_data": ((N_EPISODES - n_valid_episodes) / N_EPISODES) * 100,
            "amp_radial_mean": float(np.mean(amp_dist_list)) if amp_dist_list else np.nan,
            "std_radial_mean": float(np.mean(std_dist_list)) if std_dist_list else np.nan,
            "delta_a_band_mean": float(np.mean(amp_act_list)) if amp_act_list else np.nan,
            "speed_band_mean": float(np.mean(mean_speed_band_list)) if mean_speed_band_list else np.nan,
            "crossings_band_mean": float(np.mean(crossings_list)) if crossings_list else np.nan
        })
        print(f"  [Borde 4.2] Semilla {seed} | Paso {step} | Eps en banda (>=10 steps): {n_valid_episodes}/500 ({n_valid_episodes/5:.1f}%) | Amp Radial Media: {np.mean(amp_dist_list):.4f} m | DeltaA en Banda: {np.mean(amp_act_list):.4f} | Cruces Medios: {np.mean(crossings_list):.2f}")

df_edge = pd.DataFrame(edge_results)
df_edge_path = os.path.join(OUTPUT_DIR, "oscilacion_borde.csv")
df_edge.to_csv(df_edge_path, index=False)
print(f"  -> Guardado: {df_edge_path}")

# ==============================================================================
# 4.3. SIGMA EFECTIVA Y ALPHA POR SEMILLA
# ==============================================================================
print("\n--- 4.3. Sigma Efectiva y Alpha por Semilla ---")
# 1. Alpha desde TensorBoard
alpha_records = []
for seed in [42, 100, 2026]:
    tb_path = os.path.join(REPO_ROOT, f"experiments/phase_c/S1/seed_{seed}/logs/SAC_1")
    ea = EventAccumulator(tb_path)
    ea.Reload()
    if 'train/ent_coef' in ea.Tags()['scalars']:
        events = ea.Scalars('train/ent_coef')
        steps = [e.step for e in events]
        alphas = [e.value for e in events]
        # Puntos clave
        a_init = alphas[0]
        # paso mas cercano a 327680
        idx_327 = np.argmin(np.abs(np.array(steps) - 327680))
        a_327 = alphas[idx_327]
        # paso final 655360
        idx_655 = np.argmin(np.abs(np.array(steps) - 655360))
        a_655 = alphas[idx_655]
        a_min = np.min(alphas)
        alpha_records.append({
            "seed": seed,
            "target_entropy": -2.0, # Target fijado en S1.yaml
            "alpha_initial": float(a_init),
            "alpha_step_327680": float(a_327),
            "alpha_step_655360": float(a_655),
            "alpha_min": float(a_min)
        })
        print(f"  [Alpha TB] Semilla {seed} | Init: {a_init:.4f} | 327k: {a_327:.4f} | 655k: {a_655:.4f} | Min: {a_min:.4f}")

# 2. Sigma efectiva desde la red neuronal del actor en checkpoints 327k y 655k
sigma_records = []
for seed in [42, 100, 2026]:
    for step in [327680, 655360]:
        ckpt_dir = os.path.join(REPO_ROOT, f"experiments/phase_c/S1/seed_{seed}/checkpoints/ckpt_{step}")
        policy = load_sac_model(os.path.join(ckpt_dir, "model.zip"))
        m, v, eps, clip_o = load_norm(os.path.join(ckpt_dir, "vecnormalize_stats.pkl"))
        sim = run_simulation(policy, m, v, eps, clip_o, is_sac=True,
                             hold_steps_req=STRICT_HOLD_STEPS, vel_tol_req=STRICT_VEL_TOL)
        sigmas = sim["sigmas"] # shape (N_eval_steps, 2)
        s1_mean = float(np.mean(sigmas[:, 0]))
        s1_std = float(np.std(sigmas[:, 0]))
        s2_mean = float(np.mean(sigmas[:, 1]))
        s2_std = float(np.std(sigmas[:, 1]))
        norm_sigma_mean = float(np.mean(np.linalg.norm(sigmas, axis=1)))
        sigma_records.append({
            "seed": seed,
            "checkpoint_step": step,
            "sigma1_mean": s1_mean,
            "sigma1_std": s1_std,
            "sigma2_mean": s2_mean,
            "sigma2_std": s2_std,
            "norm_sigma_mean": norm_sigma_mean
        })
        print(f"  [Sigma Actor] Semilla {seed} | Paso {step} | sigma_1: {s1_mean:.4f} +- {s1_std:.4f} | sigma_2: {s2_mean:.4f} +- {s2_std:.4f} | ||sigma||: {norm_sigma_mean:.4f}")

df_alpha = pd.DataFrame(alpha_records)
df_sigma = pd.DataFrame(sigma_records)
df_alpha.to_csv(os.path.join(OUTPUT_DIR, "alpha_tensorboard_por_semilla.csv"), index=False)
df_sigma.to_csv(os.path.join(OUTPUT_DIR, "sigma_efectiva_por_semilla.csv"), index=False)

# ==============================================================================
# 4.4. COMPARACIÓN CON X3 EN EPISODIOS IDÉNTICOS
# ==============================================================================
print("\n--- 4.4. Comparacion con X3 en Episodios Identicos ---")
x3_results = []
s1_results_comp = []

for seed in [42, 100, 2026]:
    # Checkpoint 655360 de X3 (Phase B)
    x3_ckpt_dir = os.path.join(REPO_ROOT, f"experiments/phase_b/X3/seed_{seed}/checkpoints/ckpt_655360")
    x3_policy = load_ppo_model(os.path.join(x3_ckpt_dir, "model.zip"))
    x3_m, x3_v, x3_eps, x3_clip = load_norm(os.path.join(x3_ckpt_dir, "vecnormalize_stats.pkl"))

    # Checkpoint 655360 de S1 (Phase C)
    s1_ckpt_dir = os.path.join(REPO_ROOT, f"experiments/phase_c/S1/seed_{seed}/checkpoints/ckpt_655360")
    s1_policy = load_sac_model(os.path.join(s1_ckpt_dir, "model.zip"))
    s1_m, s1_v, s1_eps, s1_clip = load_norm(os.path.join(s1_ckpt_dir, "vecnormalize_stats.pkl"))

    # Simular X3
    sim_x3 = run_simulation(x3_policy, x3_m, x3_v, x3_eps, x3_clip, is_sac=False,
                            hold_steps_req=STRICT_HOLD_STEPS, vel_tol_req=STRICT_VEL_TOL)
    # Simular S1
    sim_s1 = run_simulation(s1_policy, s1_m, s1_v, s1_eps, s1_clip, is_sac=True,
                            hold_steps_req=STRICT_HOLD_STEPS, vel_tol_req=STRICT_VEL_TOL)

    x3_results.append({
        "model": "X3 (PPO)",
        "seed": seed,
        "checkpoint": 655360,
        "success_rate": sim_x3["success_rate"] * 100,
        "truncated_rate": sim_x3["truncated_rate"] * 100,
        "entered_no_hold_rate": sim_x3["entered_no_hold_rate"] * 100,
        "never_entered_rate": sim_x3["never_entered_rate"] * 100,
        "action_delta_mean": sim_x3["action_delta_mean"],
        "n_exits_mean": sim_x3["n_exits_mean"],
        "min_dist_mean": sim_x3["min_distance_mean"]
    })

    s1_results_comp.append({
        "model": "S1 (SAC)",
        "seed": seed,
        "checkpoint": 655360,
        "success_rate": sim_s1["success_rate"] * 100,
        "truncated_rate": sim_s1["truncated_rate"] * 100,
        "entered_no_hold_rate": sim_s1["entered_no_hold_rate"] * 100,
        "never_entered_rate": sim_s1["never_entered_rate"] * 100,
        "action_delta_mean": sim_s1["action_delta_mean"],
        "n_exits_mean": sim_s1["n_exits_mean"],
        "min_dist_mean": sim_s1["min_distance_mean"]
    })

    print(f"  [Comp 4.4] Semilla {seed:4d} | X3 Exito: {sim_x3['success_rate']*100:5.2f}% (||DeltaA||={sim_x3['action_delta_mean']:.4f}, Salidas={sim_x3['n_exits_mean']:.2f}) | S1 Exito: {sim_s1['success_rate']*100:5.2f}% (||DeltaA||={sim_s1['action_delta_mean']:.4f}, Salidas={sim_s1['n_exits_mean']:.2f})")

df_comp = pd.DataFrame(x3_results + s1_results_comp)
df_comp_path = os.path.join(OUTPUT_DIR, "comparacion_x3_vs_s1_identicos.csv")
df_comp.to_csv(df_comp_path, index=False)
print(f"  -> Guardado: {df_comp_path}")

# ==============================================================================
# 4.5. ÉXITO SEGÚN EL CRITERIO DE ENTRENAMIENTO
# ==============================================================================
print("\n--- 4.5. Exito Segun Criterio de Entrenamiento ---")
# Criterio Entrenamiento: hold_steps = 5, vel_tol = 0.30 rad/s
# Criterio Estricto: hold_steps = 20, vel_tol = 0.10 rad/s
train_crit_records = []

eval_steps_list = [40960, 163840, 286720, 327680, 491520, 655360]

for seed in [42, 100, 2026]:
    for st in eval_steps_list:
        ckpt_dir = os.path.join(REPO_ROOT, f"experiments/phase_c/S1/seed_{seed}/checkpoints/ckpt_{st}")
        policy = load_sac_model(os.path.join(ckpt_dir, "model.zip"))
        m, v, eps, clip_o = load_norm(os.path.join(ckpt_dir, "vecnormalize_stats.pkl"))

        # Criterio estricto
        sim_strict = run_simulation(policy, m, v, eps, clip_o, is_sac=True,
                                    hold_steps_req=STRICT_HOLD_STEPS, vel_tol_req=STRICT_VEL_TOL)
        # Criterio de entrenamiento
        sim_train = run_simulation(policy, m, v, eps, clip_o, is_sac=True,
                                   hold_steps_req=TRAIN_HOLD_STEPS, vel_tol_req=TRAIN_VEL_TOL)

        train_crit_records.append({
            "seed": seed,
            "checkpoint_step": st,
            "success_strict": sim_strict["success_rate"] * 100,
            "success_train_criteria": sim_train["success_rate"] * 100,
            "difference_pp": (sim_train["success_rate"] - sim_strict["success_rate"]) * 100
        })
        print(f"  [Hold 4.5] Semilla {seed:4d} | Paso {st:6d} | Estricto (N=20, v=0.10): {sim_strict['success_rate']*100:5.2f}% | Entren (N=5, v=0.30): {sim_train['success_rate']*100:5.2f}% | Brecha: +{(sim_train['success_rate'] - sim_strict['success_rate'])*100:5.2f} pp")

df_train_crit = pd.DataFrame(train_crit_records)
df_train_crit_path = os.path.join(OUTPUT_DIR, "exito_criterio_entrenamiento_vs_estricto.csv")
df_train_crit.to_csv(df_train_crit_path, index=False)
print(f"  -> Guardado: {df_train_crit_path}")

print("\n[HECHO] Todos los 5 analisis ejecutados y exportados exitosamente.")
