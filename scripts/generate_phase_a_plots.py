"""
Generación de Gráficos Oficiales y Trayectorias Reales de la Fase A (Sección 17.3).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuración estética global
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def generate_all_plots(output_dir: str = "reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print(f" GENERANDO FIGURAS DE LA FASE A EN: {output_dir}")
    print("=" * 80)

    # 1. Cargar datos de evaluación final
    episodes_det_file = "reports/data/all_final_deterministic_episodes.csv"
    episodes_stoch_file = "reports/data/all_final_stochastic_episodes.csv"
    master_table_file = "reports/data/master_comparison_table.csv"

    if not os.path.exists(episodes_det_file):
        print(">>> [AVISO] all_final_deterministic_episodes.csv no encontrado aún. Esperando datos.")
        return

    df_det = pd.read_csv(episodes_det_file)
    df_stoch = pd.read_csv(episodes_stoch_file) if os.path.exists(episodes_stoch_file) else None

    exp_order = ["X0", "X1a", "X2", "X3", "X5_near", "X5", "X6", "C0"]
    colors = {
        "X0": "#7f8c8d",
        "X1a": "#e74c3c",
        "X2": "#e67e22",
        "X3": "#f1c40f",
        "X5_near": "#9b59b6",
        "X5": "#2ecc71",
        "X6": "#1abc9c",
        "C0": "#3498db"
    }

    # --- FIGURA 1: Tasa de Éxito Estricto e Intervalos de Wilson 95% ---
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    succ_rates = []
    labels = []
    err_lows = []
    err_highs = []

    for exp in exp_order:
        sub = df_det[df_det["experiment_id"] == exp]
        if len(sub) == 0:
            continue
        total = len(sub)
        succ = int(sub["success"].sum())
        p_hat = succ / total
        z = 1.95996
        den = 1.0 + (z**2) / total
        ctr = (p_hat + (z**2) / (2 * total)) / den
        spr = z * math.sqrt((p_hat * (1 - p_hat) / total) + ((z**2) / (4 * total**2))) / den
        low = max(0.0, ctr - spr)
        high = min(1.0, ctr + spr)

        labels.append(exp)
        succ_rates.append(p_hat * 100)
        err_lows.append((p_hat - low) * 100)
        err_highs.append((high - p_hat) * 100)

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, succ_rates, yerr=[err_lows, err_highs], capsize=5,
                  color=[colors.get(l, "#34495e") for l in labels], edgecolor="#2c3e50", alpha=0.9, width=0.6)

    ax.set_ylabel("Tasa de Éxito Estricto (%)", fontsize=12, fontweight="bold")
    ax.set_title("Tasa de Éxito Estricto (Hold 20 pasos, 0.1 rad/s) en Conjunto Final (300 ep)", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(95, color="#27ae60", linestyle="--", linewidth=1.5, label="Criterio de Aceptación (≥95%)")

    for bar, rate in zip(bars, succ_rates):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 3.0, f"{rate:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_strict_success_rates.png"))
    plt.close(fig)

    # --- FIGURA 2: Llegadas Balísticas y Episodios Oscilatorios ---
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ball_rates = []
    osc_rates = []
    for exp in labels:
        sub = df_det[df_det["experiment_id"] == exp]
        ball_rates.append(sub["ballistic_flag"].mean() * 100)
        osc_rates.append(sub["oscillatory_flag"].mean() * 100)

    w = 0.35
    ax.bar(x_pos - w/2, ball_rates, width=w, label="Llegadas Balísticas (v > 0.3 rad/s)", color="#e74c3c", alpha=0.85)
    ax.bar(x_pos + w/2, osc_rates, width=w, label="Episodios Oscilatorios (exits≥2 o cross≥3)", color="#f39c12", alpha=0.85)
    ax.set_ylabel("Porcentaje de Episodios (%)", fontsize=12, fontweight="bold")
    ax.set_title("Conductas Patológicas: Llegadas Balísticas y Oscilaciones", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.axhline(5.0, color="#27ae60", linestyle="--", linewidth=1.5, label="Límite Criterio Aceptación (≤5%)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_pathological_behaviors.png"))
    plt.close(fig)

    # --- FIGURA 3: Distribución de Distancia Final ---
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    dist_data = [df_det[df_det["experiment_id"] == exp]["final_distance"].clip(upper=0.5) for exp in labels]
    bplot = ax.boxplot(dist_data, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch, exp in zip(bplot["boxes"], labels):
        patch.set_facecolor(colors.get(exp, "#95a5a6"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Distancia Final al Objetivo (m)", fontsize=12, fontweight="bold")
    ax.set_title("Distribución de Distancia Final en Evaluación (Truncada en 0.5m)", fontsize=13, fontweight="bold")
    ax.axhline(0.06, color="#e74c3c", linestyle="--", label="Radio Meta (0.06 m)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_final_distance_distributions.png"))
    plt.close(fig)

    # --- FIGURA 4: Mediana de Pasos al Éxito y Suavidad ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    steps_data = []
    smooth_data = []
    for exp in labels:
        sub_succ = df_det[(df_det["experiment_id"] == exp) & (df_det["success"] == True)]
        med_s = sub_succ["steps_to_success"].median() if len(sub_succ) > 0 else np.nan
        steps_data.append(med_s)
        sub_all = df_det[df_det["experiment_id"] == exp]
        smooth_data.append(sub_all["mean_action_delta"].mean())

    ax1.bar(x_pos, steps_data, color=[colors.get(l) for l in labels], edgecolor="#2c3e50", alpha=0.85)
    ax1.set_ylabel("Mediana de Pasos al Éxito", fontsize=11, fontweight="bold")
    ax1.set_title("Tiempo de Convergencia (Pasos)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=10, fontweight="bold")

    ax2.bar(x_pos, smooth_data, color=[colors.get(l) for l in labels], edgecolor="#2c3e50", alpha=0.85)
    ax2.set_ylabel("Media ||Delta a||", fontsize=11, fontweight="bold")
    ax2.set_title("Suavidad de Acción (Menor es más suave)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_steps_and_smoothness.png"))
    plt.close(fig)

    print(f">>> [OK] Figuras generadas exitosamente en {output_dir}")

if __name__ == "__main__":
    generate_all_plots()
