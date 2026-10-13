"""
Pipeline Integral de Generación de Figuras para Fase B (Extensión):
- Curvas de aprendizaje comparativas (agentes principales y pilotos).
- Gráficos de barras de éxito estricto con intervalos de Wilson al 95%.
- Frontera de Pareto (Éxito vs Suavidad y Éxito vs Pasos).
- Trayectorias cartesianas y series temporales de acciones en episodios canónicos.
- Figuras individuales de selección de checkpoint por agente.
- 300 dpi con metadatos JSON para cada figura generada.
"""
import os
import sys
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import wilson_score_interval

AGENT_COLORS = {
    "X1a": "#7f7f7f",       # gris (base relu grande)
    "X1b": "#8c564b",       # marron (tanh pequeña)
    "X1c": "#e377c2",       # rosa (relu mediana)
    "X1s": "#17becf",       # cyan (ent_coef 0.01)
    "X2": "#ff7f0e",        # naranja (VecNormalize)
    "X2b": "#bcbd22",       # verde oliva (X1b + VecNormalize)
    "X3": "#1f77b4",        # azul (R_dist cuadrática + R_near)
    "X4": "#2ca02c",        # verde (batch_size 128)
    "X5": "#9467bd",        # morado (x5_vel)
    "X5_near": "#d62728",   # rojo (x5_near)
    "X6": "#8c564b",        # marron rojizo (x6_smooth)
    "C0_consistent_1_5": "#000000", # negro (C0 kp=3.0 vmax=1.5)
    "C0_historical_2_0": "#555555"  # gris oscuro (C0 kp=3.0 vmax=2.0)
}

def save_fig_with_metadata(fig, output_png_path: Path, metadata: dict):
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    json_path = output_png_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [FIG OK] Guardada: {output_png_path.name}")

def plot_learning_curves_main(agg_df: pd.DataFrame, out_dir: Path):
    main_agents = ["X1a", "X2", "X3", "X4", "X5", "X5_near", "X6"]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for agent in main_agents:
        sub = agg_df[agg_df["experiment"] == agent].sort_values("step")
        if sub.empty:
            continue
        x = sub["step"] / 1000.0
        y = sub["mean_strict_success"] * 100.0
        std = sub["std_strict_success"] * 100.0
        color = AGENT_COLORS.get(agent, "#333333")

        ax.plot(x, y, label=f"{agent} (n=3)", color=color, linewidth=2.0, marker="o", markersize=3.5)
        ax.fill_between(x, np.maximum(0, y - std), np.minimum(100, y + std), color=color, alpha=0.15)

    ax.set_title("Fase B (Extensión) — Curvas de Aprendizaje: 7 Agentes Principales (n=3)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Presupuesto de Interacción (miles de pasos, k)", fontsize=11)
    ax.set_ylabel("Tasa de Éxito Estricto (%) [Media ± Desv. Inter-Semilla]", fontsize=11)
    ax.set_ylim(-2, 102)
    ax.set_xlim(30, 670)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

    meta = {
        "title": "Curvas de aprendizaje de 7 agentes principales",
        "dataset": "validation_b2_500.json",
        "n_episodes": 500,
        "n_seeds": 3,
        "seeds": [42, 100, 2026],
        "eval_mode": "deterministic",
        "budget_limit": 655360,
        "primary_metric": "strict_success_rate",
        "aggregation": "mean +- std (ddof=1)"
    }
    save_fig_with_metadata(fig, out_dir / "fig_ext_learning_curves_main.png", meta)

def plot_learning_curves_pilots(agg_df: pd.DataFrame, out_dir: Path):
    pilot_agents = ["X1a", "X1b", "X1c", "X1s", "X2b"]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for agent in pilot_agents:
        sub = agg_df[agg_df["experiment"] == agent].sort_values("step")
        if sub.empty:
            continue
        x = sub["step"] / 1000.0
        y = sub["mean_strict_success"] * 100.0
        color = AGENT_COLORS.get(agent, "#333333")
        is_ref = (agent == "X1a")
        lbl = f"{agent} (ref, n=3)" if is_ref else f"{agent} (piloto n=1, s42)"
        ls = "-" if is_ref else "--"
        lw = 2.5 if is_ref else 1.8

        ax.plot(x, y, label=lbl, color=color, linewidth=lw, linestyle=ls, marker="s" if is_ref else "o", markersize=3.5)

    ax.set_title("Fase B (Extensión) — Curvas de Aprendizaje: Pilotos n=1 vs Referencia X1a", fontsize=13, fontweight="bold")
    ax.set_xlabel("Presupuesto de Interacción (miles de pasos, k)", fontsize=11)
    ax.set_ylabel("Tasa de Éxito Estricto (%) [Validación B2]", fontsize=11)
    ax.set_ylim(-2, 102)
    ax.set_xlim(30, 670)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

    meta = {
        "title": "Curvas de aprendizaje de pilotos n=1 vs referencia X1a",
        "dataset": "validation_b2_500.json",
        "n_episodes": 500,
        "eval_mode": "deterministic",
        "budget_limit": 655360,
        "note": "Pilotos informativos n=1 con semilla 42; no participan de la regla de elección 4.2"
    }
    save_fig_with_metadata(fig, out_dir / "fig_ext_learning_curves_pilots.png", meta)

def plot_success_barplot(selected_ckpts: list, eval_detail_df: pd.DataFrame, out_dir: Path):
    c0_cons = eval_detail_df[eval_detail_df["experiment"] == "C0_consistent_1_5"].iloc[0]
    c0_hist = eval_detail_df[eval_detail_df["experiment"] == "C0_historical_2_0"].iloc[0]

    items = []
    # C0 controllers
    items.append({
        "label": "C0 (consistente)\nmax_speed=1.5",
        "success": float(c0_cons["strict_success_rate"]) * 100.0,
        "ci_low": float(c0_cons["ci_low"]) * 100.0,
        "ci_high": float(c0_cons["ci_high"]) * 100.0,
        "color": AGENT_COLORS["C0_consistent_1_5"],
        "category": "Baseline Analítico"
    })
    items.append({
        "label": "C0 (histórico)\nmax_speed=2.0",
        "success": float(c0_hist["strict_success_rate"]) * 100.0,
        "ci_low": float(c0_hist["ci_low"]) * 100.0,
        "ci_high": float(c0_hist["ci_high"]) * 100.0,
        "color": AGENT_COLORS["C0_historical_2_0"],
        "category": "Baseline Analítico"
    })

    for sel in selected_ckpts:
        agent = sel["experiment"]
        succ = sel["chosen_raw_success"] * 100.0
        # Wilson al 95% en n=500 o n=1500
        n_eval = 1500 if agent in ["X1a", "X2", "X3", "X4", "X5", "X5_near", "X6"] else 500
        n_succ = int(round(sel["chosen_raw_success"] * n_eval))
        _, low_w, high_w = wilson_score_interval(n_succ, n_eval)
        is_main = agent in ["X1a", "X2", "X3", "X4", "X5", "X5_near", "X6"]

        items.append({
            "label": f"{agent}\n(p{sel['chosen_step']//1000}k)",
            "success": succ,
            "ci_low": low_w * 100.0,
            "ci_high": high_w * 100.0,
            "color": AGENT_COLORS.get(agent, "#333333"),
            "category": "Principal (n=3)" if is_main else "Piloto (n=1)"
        })

    # Ordenar por éxito descendente
    items = sorted(items, key=lambda x: x["success"], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    x_pos = np.arange(len(items))
    bars = []

    for i, it in enumerate(items):
        err_low = it["success"] - it["ci_low"]
        err_high = it["ci_high"] - it["success"]
        bar = ax.bar(i, it["success"], color=it["color"], alpha=0.85, edgecolor="black", linewidth=0.8,
                     yerr=[[err_low], [err_high]], capsize=4)
        bars.append(bar)
        ax.text(i, it["success"] + (err_high + 1.2), f"{it['success']:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([it["label"] for it in items], fontsize=9)
    ax.set_title("Fase B (Extensión) — Tasa de Éxito Estricto en Checkpoint Seleccionado (Validación B2)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Éxito Estricto (%) [con Barras de Error Wilson 95%]", fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    meta = {
        "title": "Comparativa de tasa de éxito estricto en checkpoints óptimos",
        "dataset": "validation_b2_500.json",
        "c0_consistent_max_speed": 1.5,
        "c0_historical_max_speed": 2.0,
        "n_episodes_c0": 500,
        "items": [{k: v for k, v in it.items() if k != "color"} for it in items]
    }
    save_fig_with_metadata(fig, out_dir / "fig_ext_success_barplot.png", meta)

def plot_pareto_fronts(selected_ckpts: list, eval_detail_df: pd.DataFrame, out_dir: Path):
    c0_cons = eval_detail_df[eval_detail_df["experiment"] == "C0_consistent_1_5"].iloc[0]
    c0_hist = eval_detail_df[eval_detail_df["experiment"] == "C0_historical_2_0"].iloc[0]

    pts = []
    pts.append({
        "name": "C0_cons (1.5)",
        "success": float(c0_cons["strict_success_rate"]) * 100.0,
        "smoothness": float(c0_cons["mean_action_delta"]),
        "steps": float(c0_cons["steps_to_success_median"]),
        "color": AGENT_COLORS["C0_consistent_1_5"],
        "is_main": False
    })
    pts.append({
        "name": "C0_hist (2.0)",
        "success": float(c0_hist["strict_success_rate"]) * 100.0,
        "smoothness": float(c0_hist["mean_action_delta"]),
        "steps": float(c0_hist["steps_to_success_median"]),
        "color": AGENT_COLORS["C0_historical_2_0"],
        "is_main": False
    })

    for sel in selected_ckpts:
        agent = sel["experiment"]
        is_main = agent in ["X1a", "X2", "X3", "X4", "X5", "X5_near", "X6"]
        pts.append({
            "name": agent,
            "success": sel["chosen_raw_success"] * 100.0,
            "smoothness": sel["chosen_action_delta"],
            "steps": sel["chosen_median_steps"],
            "color": AGENT_COLORS.get(agent, "#333333"),
            "is_main": is_main
        })

    # 1. Éxito vs Suavidad (Delta a, menor es mejor)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    for p in pts:
        m = "o" if p["is_main"] else ("s" if "C0" in p["name"] else "^")
        sz = 110 if p["is_main"] else (130 if "C0" in p["name"] else 80)
        ax.scatter(p["smoothness"], p["success"], color=p["color"], s=sz, marker=m, edgecolor="black", linewidth=1.0, zorder=4)
        offset_x = 0.0005
        offset_y = 1.0 if p["success"] < 95 else -2.5
        ax.text(p["smoothness"] + offset_x, p["success"] + offset_y, p["name"], fontsize=9, fontweight="bold" if p["is_main"] else "normal")

    ax.set_title("Fase B (Extensión) — Espacio de Pareto: Éxito vs Suavidad de Acción", fontsize=13, fontweight="bold")
    ax.set_xlabel(r"Variación Media de Acción ($\overline{\Delta a}$) [Menor = Más Suave]", fontsize=11)
    ax.set_ylabel("Tasa de Éxito Estricto (%) [Validación B2]", fontsize=11)
    ax.set_ylim(40, 105)
    ax.grid(True, linestyle="--", alpha=0.6)

    meta_smooth = {
        "title": "Espacio de Pareto Éxito vs Suavidad",
        "dataset": "validation_b2_500.json",
        "metrics": {"x": "mean_action_delta", "y": "strict_success_rate"}
    }
    save_fig_with_metadata(fig, out_dir / "fig_ext_pareto_smoothness.png", meta_smooth)

    # 2. Éxito vs Mediana de Pasos al Éxito (menor es más veloz)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    for p in pts:
        m = "o" if p["is_main"] else ("s" if "C0" in p["name"] else "^")
        sz = 110 if p["is_main"] else (130 if "C0" in p["name"] else 80)
        ax.scatter(p["steps"], p["success"], color=p["color"], s=sz, marker=m, edgecolor="black", linewidth=1.0, zorder=4)
        ax.text(p["steps"] + 0.4, p["success"] + (1.0 if p["success"] < 95 else -2.5), p["name"], fontsize=9, fontweight="bold" if p["is_main"] else "normal")

    ax.set_title("Fase B (Extensión) — Espacio de Pareto: Éxito vs Velocidad de Alcance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mediana de Pasos al Éxito (pasos) [Menor = Más Veloz]", fontsize=11)
    ax.set_ylabel("Tasa de Éxito Estricto (%) [Validación B2]", fontsize=11)
    ax.set_ylim(40, 105)
    ax.grid(True, linestyle="--", alpha=0.6)

    meta_steps = {
        "title": "Espacio de Pareto Éxito vs Pasos",
        "dataset": "validation_b2_500.json",
        "metrics": {"x": "steps_to_success_median", "y": "strict_success_rate"}
    }
    save_fig_with_metadata(fig, out_dir / "fig_ext_pareto_speed.png", meta_steps)

def plot_canonical_trajectories_and_actions(step_df: pd.DataFrame, out_dir: Path):
    agents_to_show = ["X3", "X4", "X5", "X6", "C0_consistent_1_5"]
    episodes = [311, 85, 167, 480]
    ep_titles = {
        311: "Q10 (Corto, r=0.31m)",
        85: "Q50 (Medio, r=1.01m)",
        167: "Q90 (Largo, r=1.73m)",
        480: "EDGE (Borde, r=1.90m)"
    }

    # Trayectorias cartesianas (2x2 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=300)
    axes = axes.flatten()

    for idx, ep_id in enumerate(episodes):
        ax = axes[idx]
        ep_data = step_df[step_df["episode_id"] == ep_id]
        tgt_x = ep_data["target_x"].iloc[0]
        tgt_y = ep_data["target_y"].iloc[0]

        # Dibujar base, espacio de trabajo y objetivo
        circle_ws = plt.Circle((0, 0), 2.0, color="#f0f0f0", fill=True, zorder=1)
        circle_tgt = plt.Circle((tgt_x, tgt_y), 0.06, color="gold", ec="darkgoldenrod", fill=True, lw=1.5, zorder=2)
        ax.add_patch(circle_ws)
        ax.add_patch(circle_tgt)
        ax.plot(tgt_x, tgt_y, "x", color="black", markersize=6, zorder=3)

        for ag in agents_to_show:
            ag_sub = ep_data[ep_data["agent"] == ag]
            if ag_sub.empty:
                continue
            color = AGENT_COLORS.get(ag, "#333333")
            lw = 2.0 if ag in ["X4", "C0_consistent_1_5"] else 1.5
            ls = "-" if ag != "C0_consistent_1_5" else "--"
            ax.plot(ag_sub["ee_x"], ag_sub["ee_y"], label=ag, color=color, linewidth=lw, linestyle=ls, zorder=4)

        ax.set_title(f"Episodio {ep_id}: {ep_titles[ep_id]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("x (m)", fontsize=10)
        ax.set_ylabel("y (m)", fontsize=10)
        ax.set_xlim(-0.5, 2.1)
        ax.set_ylim(-0.8, 2.1)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    fig.suptitle("Fase B (Extensión) — Trayectorias Cartesianas del Extremo (EE) en Episodios Canónicos", fontsize=13, fontweight="bold")
    meta_traj = {
        "title": "Trayectorias cartesianas del extremo en 4 episodios canónicos",
        "dataset": "validation_b2_500.json",
        "canonical_episodes": episodes,
        "agents": agents_to_show
    }
    save_fig_with_metadata(fig, out_dir / "fig_ext_canonical_trajectories.png", meta_traj)

    # Series temporales de acción (EDGE 480 y Q50 85)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    for row_idx, ep_id in enumerate([85, 480]):
        ep_data = step_df[step_df["episode_id"] == ep_id]
        ax_dist = axes[row_idx, 0]
        ax_act = axes[row_idx, 1]

        for ag in agents_to_show:
            ag_sub = ep_data[ep_data["agent"] == ag]
            if ag_sub.empty:
                continue
            color = AGENT_COLORS.get(ag, "#333333")
            lw = 1.8 if ag in ["X4", "C0_consistent_1_5"] else 1.3
            ls = "-" if ag != "C0_consistent_1_5" else "--"

            ax_dist.plot(ag_sub["step"], ag_sub["distance"], label=ag, color=color, lw=lw, ls=ls)
            ax_act.plot(ag_sub["step"], ag_sub["delta_action_norm"], label=ag, color=color, lw=lw, ls=ls)

        ax_dist.axhline(0.06, color="gold", linestyle=":", lw=1.5, label="Radio objetivo (0.06m)")
        ax_dist.set_title(f"Episodio {ep_id} ({ep_titles[ep_id]}): Distancia al Objetivo", fontsize=10.5, fontweight="bold")
        ax_dist.set_xlabel("Paso de simulación (t)", fontsize=9.5)
        ax_dist.set_ylabel("Distancia (m)", fontsize=9.5)
        ax_dist.set_ylim(-0.02, 2.0)
        ax_dist.grid(True, linestyle=":", alpha=0.6)

        ax_act.set_title(f"Episodio {ep_id}: Variación de Acción ||Δa||", fontsize=10.5, fontweight="bold")
        ax_act.set_xlabel("Paso de simulación (t)", fontsize=9.5)
        ax_act.set_ylabel("||Δa|| (rad/s)", fontsize=9.5)
        ax_act.set_ylim(-0.01, 0.45)
        ax_act.grid(True, linestyle=":", alpha=0.6)

        if row_idx == 0:
            ax_dist.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("Fase B (Extensión) — Dinámica Temporal y Suavidad de Acción en Episodios Canónicos", fontsize=13, fontweight="bold")
    meta_act = {
        "title": "Perfiles temporales de distancia y delta de acción",
        "dataset": "validation_b2_500.json",
        "episodes": [85, 480],
        "agents": agents_to_show
    }
    save_fig_with_metadata(fig, out_dir / "fig_ext_canonical_actions.png", meta_act)

def plot_agent_selection_figures(selected_ckpts: list, agg_df: pd.DataFrame, out_dir: Path):
    for sel in selected_ckpts:
        agent = sel["experiment"]
        sub = agg_df[agg_df["experiment"] == agent].sort_values("step")
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
        x = sub["step"] / 1000.0
        y_raw = sub["mean_strict_success"] * 100.0
        
        # Suavizado de 3 puntos
        y_smooth = []
        raw_list = list(sub["mean_strict_success"])
        for k in range(len(raw_list)):
            if k == 0:
                y_smooth.append((raw_list[0] + raw_list[1]) / 2.0 if len(raw_list) > 1 else raw_list[0])
            elif k == len(raw_list) - 1:
                y_smooth.append((raw_list[k - 1] + raw_list[k]) / 2.0)
            else:
                y_smooth.append((raw_list[k - 1] + raw_list[k] + raw_list[k + 1]) / 3.0)
        y_smooth = np.array(y_smooth) * 100.0

        color = AGENT_COLORS.get(agent, "#1f77b4")
        ax.plot(x, y_raw, "o--", color=color, alpha=0.45, label="Raw (por checkpoint)", markersize=4)
        ax.plot(x, y_smooth, "s-", color=color, linewidth=2.2, label="Suavizado (3 puntos)", markersize=5)

        # Banda de equivalencia U_equiv
        u_width = sel["u_equiv_width"] * 100.0
        max_smooth = sel["max_smoothed_success"] * 100.0
        thresh = sel["equiv_threshold"] * 100.0
        ax.axhline(thresh, color="firebrick", linestyle=":", linewidth=1.5, label=f"Umbral Indiferencia ({thresh:.1f}%)")
        ax.axhspan(thresh, max_smooth, color="firebrick", alpha=0.08, label=f"Banda U_equiv ({u_width:.1f}pp)")

        # Resaltar checkpoint elegido
        chosen_k = sel["chosen_step"] / 1000.0
        chosen_y = sel["chosen_smoothed_success"] * 100.0
        ax.scatter([chosen_k], [chosen_y], color="gold", edgecolors="black", s=140, zorder=6,
                   label=f"Elegido: Paso {sel['chosen_step']:,} ({chosen_y:.1f}%)")

        ax.set_title(f"Agente {agent} — Selección Formal de Checkpoint (Regla 4.1)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Presupuesto de Interacción (miles de pasos, k)", fontsize=10.5)
        ax.set_ylabel("Éxito Estricto (%) [Validación B2]", fontsize=10.5)
        ax.set_ylim(-2, 105)
        ax.set_xlim(30, 670)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

        meta_sel = {
            "agent": agent,
            "chosen_step": sel["chosen_step"],
            "chosen_smoothed_success": sel["chosen_smoothed_success"],
            "max_step": sel["max_step"],
            "max_smoothed_success": sel["max_smoothed_success"],
            "u_equiv_width": sel["u_equiv_width"],
            "final_slope_status": sel["final_slope_status"],
            "dataset": "validation_b2_500.json"
        }
        save_fig_with_metadata(fig, out_dir / f"fig_agent_{agent}_selection.png", meta_sel)

def run_generate_all_figures():
    out_dir = REPO_ROOT / "reports" / "figures" / "phase_b" / "extension"
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_csv = REPO_ROOT / "reports/data/phase_b/extension/checkpoints_eval_validation_aggregated.csv"
    detail_csv = REPO_ROOT / "reports/data/phase_b/extension/checkpoints_eval_validation.csv"
    sel_json = REPO_ROOT / "reports/data/phase_b/extension/selected_checkpoints.json"
    step_csv = REPO_ROOT / "reports/data/phase_b/extension/step_data_val_geometry.csv"

    print(">>> Cargando datos de evaluación y telemetría...")
    agg_df = pd.read_csv(agg_csv)
    detail_df = pd.read_csv(detail_csv)
    with open(sel_json, "r", encoding="utf-8") as f:
        selected_ckpts = json.load(f)
    step_df = pd.read_csv(step_csv)

    print("\n>>> [1/5] Generando curvas de aprendizaje globales...")
    plot_learning_curves_main(agg_df, out_dir)
    plot_learning_curves_pilots(agg_df, out_dir)

    print(">>> [2/5] Generando comparativa de barras de éxito estricto con Wilson...")
    plot_success_barplot(selected_ckpts, detail_df, out_dir)

    print(">>> [3/5] Generando espacios de Pareto (suavidad y velocidad)...")
    plot_pareto_fronts(selected_ckpts, detail_df, out_dir)

    print(">>> [4/5] Generando trayectorias y series temporales canónicas...")
    plot_canonical_trajectories_and_actions(step_df, out_dir)

    print(">>> [5/5] Generando figuras individuales de selección por agente...")
    plot_agent_selection_figures(selected_ckpts, agg_df, out_dir)

    print(f"\n================================================================================")
    print(f"=== TODAS LAS FIGURAS GENERADAS EXITOSAMENTE EN: {out_dir.relative_to(REPO_ROOT)} ===")
    print(f"================================================================================")

if __name__ == "__main__":
    run_generate_all_figures()
