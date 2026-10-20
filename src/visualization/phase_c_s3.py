"""
Figuras del análisis de cierre de Fase C (S3_combined) - 300 dpi + JSON.

FIGURAS GENERADAS
=================
fig1_curva_exito_245k, fig2_desglose_fallos_245k,
fig3_suavidad_y_salidas_245k, fig4_eficiencia_computacional_245k en
reports/phase_c_s3_analysis/figures/.

INSUMOS REQUERIDOS
==================
- reports/phase_c_s2_analysis/tabla_comparativa_245k_final.csv
- reports/phase_c_s3_analysis/metricas_checkpoints_s3_combined_agg.csv
- reports/phase_c_s3_analysis/tabla_comparativa_3seeds_245k.csv

USO
===
python scripts/generate_figures.py --phase c_s3
python scripts/generate_figures.py --phase c_s3 --out-dir <ruta_temporal>
python -m src.visualization.phase_c_s3                  # equivalente directo

PARÁMETROS CONFIGURABLES
========================
out_dir     : directorio de salida (por defecto
              reports/phase_c_s3_analysis/figures); aceptado por
              generate_all_s3_figures(out_dir) / run(out_dir) / --out-dir.
STEPS       : pasos evaluados (40.960 ... 245.760).
SEEDS       : semillas consideradas ("42", "100", "2026").
train_times / fps_vals (definidos en la fig4) : tiempos de entrenamiento
              (minutos, 3 semillas) y FPS medios por modelo, medidos en GPU.
COLORS      : paleta de los 5 modelos (PPO X3, SAC S1, SAC S2_freq,
              SAC S2_wa, SAC S3_combined).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

STEPS = [40960, 81920, 122880, 163840, 204800, 245760]
SEEDS = ["42", "100", "2026"]

# Estilos y colores (idénticos a los históricos)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

COLORS = {
    "PPO X3": "#1f77b4",          # Azul
    "SAC S1": "#7f7f7f",          # Gris
    "SAC S2_freq": "#2ca02c",     # Verde
    "SAC S2_wa": "#d62728",       # Rojo
    "SAC S3_combined": "#9467bd"  # Púrpura
}

def generate_all_s3_figures(out_dir=None):
    """
    Genera las 4 figuras y sus sidecars JSON.

    out_dir : ``None`` (por defecto reports/phase_c_s3_analysis/figures)
              o ruta alternativa de salida.
    """
    out = Path(out_dir) if out_dir is not None else REPO_ROOT / "reports" / "phase_c_s3_analysis" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR = out

    # 1. Cargar datos (originalmente a nivel de módulo)
    master_csv = REPO_ROOT / "reports" / "phase_c_s2_analysis" / "tabla_comparativa_245k_final.csv"
    df_master = pd.read_csv(master_csv)
    s3_agg_csv = REPO_ROOT / "reports" / "phase_c_s3_analysis" / "metricas_checkpoints_s3_combined_agg.csv"
    df_s3 = pd.read_csv(s3_agg_csv)

    # Preparar datos agregados de todos los modelos a lo largo de los pasos
    model_data = {}
    for m in ["PPO X3", "SAC S1", "SAC S2_freq", "SAC S2_wa"]:
        means, stds = [], []
        for st in STEPS:
            sub = df_master[(df_master["model"] == m) & (df_master["step"] == st) & (df_master["seed"].astype(str).isin(SEEDS))]
            means.append(sub["success_rate"].mean() * 100)
            stds.append(sub["success_rate"].std(ddof=1) * 100)
        model_data[m] = {"mean": np.array(means), "std": np.array(stds)}

    model_data["SAC S3_combined"] = {
        "mean": df_s3["success_rate_mean"].values * 100,
        "std": df_s3["success_rate_std"].values * 100
    }

    # ----------------------------------------------------------------------
    # FIGURA 1: Curvas de Éxito Estricto
    # ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    steps_k = np.array(STEPS) / 1000.0

    for m, vals in model_data.items():
        ls = "-" if m in ["PPO X3", "SAC S3_combined"] else "--"
        lw = 2.2 if m in ["PPO X3", "SAC S3_combined"] else 1.6
        ax.plot(steps_k, vals["mean"], label=m, color=COLORS[m], linestyle=ls, linewidth=lw, marker="o", markersize=4.5)
        ax.fill_between(steps_k, vals["mean"] - vals["std"], vals["mean"] + vals["std"], color=COLORS[m], alpha=0.15)

    ax.axhline(93.53, color=COLORS["PPO X3"], linestyle=":", alpha=0.6, label="PPO X3 final (93,53 %)")
    ax.set_title("Evolución de Tasa de Éxito Estricto (Validation C, 3 semillas)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Pasos de Entrenamiento (en miles)", fontsize=10)
    ax.set_ylabel("Éxito Estricto (%)", fontsize=10)
    ax.set_ylim(-2, 102)
    ax.set_xlim(35, 255)
    ax.set_xticks(steps_k)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9, loc="lower right")
    fig.tight_layout()

    fig1_path = OUTPUT_DIR / "fig1_curva_exito_245k.png"
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)

    with open(OUTPUT_DIR / "fig1_curva_exito_245k.json", "w", encoding="utf-8") as f:
        json.dump({
            "figure": "fig1_curva_exito_245k.png",
            "description": "Curvas de aprendizaje de éxito estricto desde 40.960 a 245.760 pasos (5 modelos, 3 semillas)",
            "dataset": "validation_c_500.json (500 episodios)",
            "protocol": "hold_steps: 20, velocity_tolerance: 0.10, target_radius: 0.06",
            "gatekeeper_reference_ppo": 93.53,
            "models": list(model_data.keys()),
            "steps": STEPS
        }, f, indent=4)

    # ----------------------------------------------------------------------
    # FIGURA 2: Desglose de Modos de Fallo al Paso 245.760
    # ----------------------------------------------------------------------
    comp_3seeds_csv = REPO_ROOT / "reports" / "phase_c_s3_analysis" / "tabla_comparativa_3seeds_245k.csv"
    df_comp = pd.read_csv(comp_3seeds_csv)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    models_order = ["PPO X3", "SAC S1", "SAC S2_freq", "SAC S2_wa", "SAC S3_combined"]
    df_plot = df_comp.set_index("model").loc[models_order].reset_index()

    succ = df_plot["success_rate_mean"].values * 100
    enh = df_plot["entered_no_hold_mean"].values * 100
    ne = df_plot["never_entered_mean"].values * 100
    # Residuos para sumar 100%
    other = np.maximum(0, 100 - (succ + enh + ne))

    x = np.arange(len(models_order))
    width = 0.55

    p1 = ax.bar(x, succ, width, label="Éxito Estricto", color="#2ca02c", alpha=0.85)
    p2 = ax.bar(x, enh, width, bottom=succ, label="Entered, No Hold", color="#ff7f0e", alpha=0.85)
    p3 = ax.bar(x, ne, width, bottom=succ+enh, label="Never Entered", color="#d62728", alpha=0.85)
    if np.sum(other) > 0.5:
        ax.bar(x, other, width, bottom=succ+enh+ne, label="Otros", color="#7f7f7f", alpha=0.5)

    ax.set_title("Desglose de Resultados y Modos de Fallo en Paso 245.760 (Validation C)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Proporción de Episodios (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9, loc="upper right")
    fig.tight_layout()

    fig2_path = OUTPUT_DIR / "fig2_desglose_fallos_245k.png"
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)

    with open(OUTPUT_DIR / "fig2_desglose_fallos_245k.json", "w", encoding="utf-8") as f:
        json.dump({
            "figure": "fig2_desglose_fallos_245k.png",
            "description": "Desglose apilado de modos de fallo en paso 245.760 sobre validation_c_500.json",
            "models": models_order,
            "success_rate": succ.tolist(),
            "entered_no_hold": enh.tolist(),
            "never_entered": ne.tolist()
        }, f, indent=4)

    # ----------------------------------------------------------------------
    # FIGURA 3: Suavidad de Acción y Salidas de Meta al Paso 245.760
    # ----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    delta_a = df_plot["action_delta_mean"].values
    delta_a_std = df_plot["action_delta_std"].values
    exits = df_plot["n_exits_mean"].values
    exits_std = df_plot["n_exits_std"].values

    colors_bar = [COLORS[m] for m in models_order]

    ax1.bar(x, delta_a, width=0.5, yerr=delta_a_std, capsize=4, color=colors_bar, alpha=0.8)
    ax1.set_title("Variación Media de Acción ||Δa||", fontsize=11, fontweight="bold")
    ax1.set_ylabel("||Δa|| (rad/s)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_order, rotation=20, ha="right", fontsize=8.5)

    ax2.bar(x, exits, width=0.5, yerr=exits_std, capsize=4, color=colors_bar, alpha=0.8)
    ax2.set_title("Salidas Medias del Radio de Meta (n_exits)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Eventos por Episodio", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_order, rotation=20, ha="right", fontsize=8.5)

    fig.suptitle("Estabilidad Cinemática y Anclaje en Meta (Paso 245.760)", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()

    fig3_path = OUTPUT_DIR / "fig3_suavidad_y_salidas_245k.png"
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)

    with open(OUTPUT_DIR / "fig3_suavidad_y_salidas_245k.json", "w", encoding="utf-8") as f:
        json.dump({
            "figure": "fig3_suavidad_y_salidas_245k.png",
            "description": "Comparativa de variación de acción ||Δa|| y salidas no deseadas de meta en paso 245.760",
            "models": models_order,
            "action_delta_mean": delta_a.tolist(),
            "n_exits_mean": exits.tolist()
        }, f, indent=4)

    # ----------------------------------------------------------------------
    # FIGURA 4: Eficiencia Computacional y Rendimiento
    # ----------------------------------------------------------------------
    train_times = {
        "PPO X3": 7.62,
        "SAC S1": 94.86,
        "SAC S2_freq": 28.24,
        "SAC S2_wa": 99.23,
        "SAC S3_combined": 30.40
    }
    fps_vals = {
        "PPO X3": 1614.3,
        "SAC S1": 129.7,
        "SAC S2_freq": 435.1,
        "SAC S2_wa": 124.1,
        "SAC S3_combined": 404.2
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)

    for m in models_order:
        t = train_times[m]
        s = df_plot[df_plot["model"] == m]["success_rate_mean"].values[0] * 100
        fps = fps_vals[m]
        ax.scatter(t, s, color=COLORS[m], s=140, zorder=5, label=f"{m} ({fps:.0f} FPS)")
        ax.annotate(f"{m}\n({s:.1f} %, {t:.1f} min)", (t, s), textcoords="offset points", xytext=(8, -8), fontsize=8.5)

    ax.set_title("Costo Computacional vs. Rendimiento al Paso 245.760", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Tiempo Total de Entrenamiento (3 semillas, minutos)", fontsize=10)
    ax.set_ylabel("Éxito Estricto Final (%)", fontsize=10)
    ax.set_ylim(35, 100)
    ax.set_xlim(-5, 115)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9, loc="lower right")
    fig.tight_layout()

    fig4_path = OUTPUT_DIR / "fig4_eficiencia_computacional_245k.png"
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)

    with open(OUTPUT_DIR / "fig4_eficiencia_computacional_245k.json", "w", encoding="utf-8") as f:
        json.dump({
            "figure": "fig4_eficiencia_computacional_245k.png",
            "description": "Comparativa de tiempo wall-clock de entrenamiento y FPS vs éxito estricto en paso 245.760 (5 modelos)",
            "train_times_minutes": train_times,
            "mean_fps": fps_vals,
            "success_rate_245k": succ.tolist(),
            "models": models_order
        }, f, indent=4)

    print(">>> [FIGURAS] Las 4 figuras y sidecars JSON se han generado en:", OUTPUT_DIR)


def run(out_dir=None):
    """Punto de entrada del módulo (ver ``generate_all_s3_figures``)."""
    generate_all_s3_figures(out_dir)


if __name__ == "__main__":
    run()
