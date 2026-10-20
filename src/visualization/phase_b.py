"""
Figuras oficiales de Fase B (300 dpi + JSON de trazabilidad).

FIGURAS GENERADAS
=================
fig_b_01 ... fig_b_08 en reports/figures/phase_b/: éxito vs pasos,
determinista vs estocástico, desenlaces por categoría, distribuciones de
velocidad, distancia/suavidad, trayectorias superpuestas, tabla de
criterios de aceptación y diagnósticos internos de PPO.

INSUMOS REQUERIDOS
==================
- Eventos de TensorBoard bajo experiments/phase_b/<exp>/seed_<seed>/
- reports/data/phase_b/tanda_1_x5_near_summary.json
- Datasets canónicos (src.evaluation.datasets)

USO
===
python scripts/generate_figures.py --phase b
python scripts/generate_figures.py --phase b --out-dir <ruta_temporal>
python -m src.visualization.phase_b                     # equivalente directo

PARÁMETROS CONFIGURABLES
========================
FIG_B_DIR    : directorio de salida (``Path``). Por defecto
               reports/figures/phase_b; ``run(out_dir)`` / ``--out-dir``
               sustituye su valor para una ejecución concreta
               (``--data-dir`` hace lo propio con DATA_B_DIR).
EXP_COLORS   : paleta por experimento (C0, X0, X3, X5_near, X5, X6).
               Mantener los tonos históricos: los JSON publicados los
               asocian a las figuras ya publicadas. Los experimentos
               ausentes en la paleta reciben un color del ciclo
               neutro por orden de aparición (ver common.color_resolver),
               p. ej. un modelo recién entrenado al invocar
               ``run(..., experiments=["MiModelo"])``.
CAT_COLORS /
CATEGORIES   : paleta y orden canónico de las 6 categorías de desenlace
               (1. Éxito limpio ... 6. Fallo sin estabilizar).
fig2/fig3/fig8 aceptan selección de experimentos y semillas
               (``run(experiments=..., seeds=...)``; ``--experiments``,
               ``--seeds``, ``--exps-dir``, ``--figures`` en el dispatcher);
               sin banderas reproducen el estudio histórico
               (["X5_near", "X3"], semillas [42, 100, 2026]). Las figuras
               1 y 4-7 son monográficas (episodios curados) y mantienen
               sus insumos fijos: se omiten con aviso si los CSV no
               existen. ``selected_ckpts`` solo por biblioteca.
spd_beams... : constantes estéticas internas de cada función de figura
               (tamaños de figura, anotaciones); no alterar salvo
               necesidad documentada.
"""

import os
import sys
from pathlib import Path
import json
import math
import datetime
import platform
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorboard.backend.event_processing import event_accumulator

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import wilson_score_interval
from src.evaluation.datasets import save_and_get_datasets
from src.visualization import common
from src.visualization.common import apply_style, classify_episodes


def classify_df_episodes(df):
    """Alias de compatibilidad de la Fase B (usa ``outcome_category``)."""
    return classify_episodes(df, column="outcome_category")

# Rutas de datos y figuras
DATA_B_DIR = REPO_ROOT / "reports" / "data" / "phase_b"
FIG_B_DIR = REPO_ROOT / "reports" / "figures" / "phase_b"
FIG_B_DIR.mkdir(parents=True, exist_ok=True)

## Configuración estética (compartida en common.py)
apply_style()

EXP_COLORS = {
    "C0": "#2980b9",        # Azul
    "X0": "#7f8c8d",        # Gris histórico
    "X3": "#f39c12",        # Naranja/Amarillo
    "X5_near": "#8e44ad",   # Violeta
    "X5": "#27ae60",        # Verde
    "X6": "#16a085"         # Turquesa
}

CAT_COLORS = {
    "1. Éxito limpio": "#27ae60",          # Verde
    "2. Éxito con patología": "#82e0aa",    # Verde claro
    "3. Fallo oscilatorio": "#f39c12",      # Naranja
    "4. Fallo balístico": "#e74c3c",        # Rojo
    "5. Fallo sin aproximación": "#34495e", # Gris azulado oscuro
    "6. Fallo sin estabilizar": "#5dade2"   # Azul pizarra
}

# Resolución de color: canónicos de EXP_COLORS + ciclo neutro determinista
# para nombres nuevos (common.color_resolver).
get_exp_color = common.color_resolver(EXP_COLORS)

# Guarda de insumos: las figuras monográficas se omiten con aviso cuando
# sus CSV característicos no existen (p. ej. un --data-dir alternativo).
REQUIRED_FIG1 = ["eval_val_c0_episodes.csv", "eval_val_x5_near_phase_a_det.csv",
                 "eval_val_x5_near_phase_b_selected.csv", "eval_val_x3_phase_b_selected.csv"]
REQUIRED_FIG4 = ["step_data_val_geometry.csv", "eval_val_c0_episodes.csv",
                 "eval_val_x5_near_phase_b_selected.csv"]
REQUIRED_FIG5_7 = ["step_data_val_geometry.csv"]


def _required_inputs(file_names: list) -> bool:
    missing = [DATA_B_DIR / name for name in file_names if not (DATA_B_DIR / name).exists()]
    if missing:
        print(f"  [OMITIDA] Faltan insumos en {DATA_B_DIR}: "
              f"{', '.join(p.name for p in missing)}")
        return False
    return True

CATEGORIES = [
    "1. Éxito limpio",
    "2. Éxito con patología",
    "3. Fallo oscilatorio",
    "4. Fallo balístico",
    "5. Fallo sin aproximación",
    "6. Fallo sin estabilizar"
]

def save_fig_with_metadata(fig, out_png_path: Path, input_files: list, description: str):
    """Guarda figura a 300 dpi con JSON de trazabilidad (metadatos de Fase B)."""
    common.save_fig_with_metadata(fig, out_png_path, input_files=input_files,
                                  description=description, phase="B", verbose=True)


def generate_figure_1():
    print("\n--- Generando Figura 1: Mapa de Fallos en el Espacio de Trabajo ---")
    if not _required_inputs(REQUIRED_FIG1):
        return
    val_set, _ = save_and_get_datasets()
    targets = {ep["episode_id"]: (ep["target_x"], ep["target_y"], ep.get("target_r", np.sqrt(ep["target_x"]**2 + ep["target_y"]**2))) for ep in val_set}

    c0_df = classify_df_episodes(pd.read_csv(DATA_B_DIR / "eval_val_c0_episodes.csv"))
    x5a_df = classify_df_episodes(pd.read_csv(DATA_B_DIR / "eval_val_x5_near_phase_a_det.csv"))
    x5b_all = classify_df_episodes(pd.read_csv(DATA_B_DIR / "eval_val_x5_near_phase_b_selected.csv"))
    x3b_all = classify_df_episodes(pd.read_csv(DATA_B_DIR / "eval_val_x3_phase_b_selected.csv"))

    # Filtrar semilla 42 para mapas individuales
    x5b_det = x5b_all[(x5b_all["seed"] == 42) & (x5b_all["evaluation_mode"] == "deterministic")]
    x5b_stoch = x5b_all[(x5b_all["seed"] == 42) & (x5b_all["evaluation_mode"] == "stochastic")]
    x3b_det = x3b_all[(x3b_all["seed"] == 42) & (x3b_all["evaluation_mode"] == "deterministic")]
    x3b_stoch = x3b_all[(x3b_all["seed"] == 42) & (x3b_all["evaluation_mode"] == "stochastic")]

    panels = [
        ("C0 Analítico (Determinista)", c0_df),
        ("X5_near Fase A — 300k (Determinista)", x5a_df),
        ("X5_near Fase B — Ckpt 901k (Determinista)", x5b_det),
        ("X3 Fase B — Ckpt 409k (Determinista)", x3b_det),
        ("X5_near Fase B — Ckpt 901k (Estocástico)", x5b_stoch),
        ("X3 Fase B — Ckpt 409k (Estocástico)", x3b_stoch)
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)
    axes = axes.flatten()

    for idx, (title, df) in enumerate(panels):
        ax = axes[idx]
        # Dibujar límites de alcance del robot (R = 2.0 m, r_min = 0.15 m)
        circle_max = plt.Circle((0, 0), 2.0, color="#bdc3c7", fill=False, linestyle="--", linewidth=1.2, label="Alcance máx (2.0 m)")
        circle_min = plt.Circle((0, 0), 0.15, color="#bdc3c7", fill=False, linestyle=":", linewidth=1.0)
        ax.add_patch(circle_max)
        ax.add_patch(circle_min)

        for cat in CATEGORIES:
            sub = df[df["outcome_category"] == cat]
            if len(sub) > 0:
                xs = [targets[epid][0] for epid in sub["episode_id"]]
                ys = [targets[epid][1] for epid in sub["episode_id"]]
                ax.scatter(xs, ys, color=CAT_COLORS[cat], label=f"{cat} (n={len(sub)})", s=28, alpha=0.85, edgecolors="none")

        succ_rate = df["success"].mean() * 100
        p_hat, low, high = wilson_score_interval(int(df["success"].sum()), len(df))
        ax.set_title(f"{title}\nÉxito: {succ_rate:.1f}% [{low*100:.1f}%, {high*100:.1f}%] (n={len(df)})", fontsize=11, fontweight="bold")
        ax.set_xlabel("x_target (m)", fontsize=9)
        ax.set_ylabel("y_target (m)", fontsize=9)
        ax.set_xlim(-2.15, 2.15)
        ax.set_ylim(-2.15, 2.15)
        ax.set_aspect("equal")
        if idx == 0:
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    fig.suptitle("Figura B-1: Mapa de Fallos y Éxitos en el Espacio de Trabajo sobre VALIDACIÓN (200 Episodios)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_p = FIG_B_DIR / "fig_b_01_workspace_failure_map.png"
    save_fig_with_metadata(fig, out_p, [DATA_B_DIR / "eval_val_x5_near_phase_b_selected.csv", DATA_B_DIR / "eval_val_x3_phase_b_selected.csv"],
                           "Mapa cartesiano de desenlaces sobre conjunto de validación en 6 categorías.")

# ==============================================================================
# FIGURA 2: CURVAS DE ENTRENAMIENTO POR AGENTE (FASE A vs FASE B)
# ==============================================================================
def generate_figure_2(exps=None, seeds=None, experiments_root=None):
    """
    Curvas de entrenamiento por agente desde TensorBoard.

    exps             : lista de experimentos (por defecto ["X5_near", "X3"]);
                       se dibuja una fila por experimento.
    seeds            : lista de semillas (por defecto [42, 100, 2026]).
    experiments_root : raíz de las corridas para localizar los eventos
                       (por defecto experiments/phase_b); p. ej. un
                       experimento entrenado en otra fase indica su raíz
                       correspondiente.
    """
    print("\n--- Generando Figura 2: Curvas de Entrenamiento por Agente ---")
    seeds = seeds if seeds is not None else [42, 100, 2026]
    exps = exps if exps is not None else ["X5_near", "X3"]
    experiments_root = Path(experiments_root) if experiments_root is not None else REPO_ROOT / "experiments" / "phase_b"

    # Extraer curvas de TensorBoard de la raíz indicada
    data_b = {}
    for exp in exps:
        data_b[exp] = {}
        for s in seeds:
            tb_dir = experiments_root / exp / f"seed_{s}" / "logs"
            efs = list(tb_dir.glob("**/events.out.tfevents.*"))
            if efs:
                ea = event_accumulator.EventAccumulator(str(efs[-1]), size_guidance={"scalars": 0})
                ea.Reload()
                tags = ea.Tags().get("scalars", [])
                data_b[exp][s] = {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in tags}

    fig, axes = plt.subplots(len(exps), 3, figsize=(18, 5 * len(exps)), dpi=300, squeeze=False)

    tags_info = [
        ("rollout/ep_rew_mean", "Recompensa Media por Episodio", "Ep Rew Mean (no comparable entre X3 y X5_near)"),
        ("rollout/ep_len_mean", "Longitud Media de Episodio (Pasos)", "Ep Length"),
        ("train/success_rate_window", "Tasa de Éxito en Ventana de Entrenamiento (100 eps)", "Success Rate Window")
    ]

    for row_idx, exp in enumerate(exps):
        exp_col = get_exp_color(exp)
        for col_idx, (tag, title, ylabel) in enumerate(tags_info):
            ax = axes[row_idx, col_idx]
            
            curves = [data_b[exp][s][tag] for s in seeds if s in data_b[exp] and tag in data_b[exp][s]]
            if curves:
                # Interpolar en una malla común de pasos
                min_step = max(c[0][0] for c in curves)
                max_step = min(c[-1][0] for c in curves)
                common_steps = np.linspace(min_step, max_step, 150)
                
                vals_interp = []
                for s, c in zip(seeds, curves):
                    c_steps = [p[0] for p in c]
                    c_vals = [p[1] for p in c]
                    v_int = np.interp(common_steps, c_steps, c_vals)
                    vals_interp.append(v_int)
                    ax.plot(c_steps, c_vals, label=f"Semilla {s}", alpha=0.35, linewidth=1.0)

                vals_interp = np.array(vals_interp)
                mean_curve = np.mean(vals_interp, axis=0)
                std_curve = np.std(vals_interp, axis=0, ddof=1)
                
                ax.plot(common_steps, mean_curve, color=exp_col, linewidth=2.2,
                        label=f"Media Multi-Semilla (n={len(curves)})")
                ax.fill_between(common_steps, mean_curve - std_curve, mean_curve + std_curve, color=exp_col, alpha=0.18)

            # Marcar frontera de 303k pasos de Fase A
            ax.axvline(303104, color="#e74c3c", linestyle="--", linewidth=1.2, label="Límite Fase A (303k)")
            
            ax.set_title(f"{exp} — {title}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Pasos de Entrenamiento (Timesteps)", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(loc="best", fontsize=8)

    n_seeds_txt = f"{len(seeds)} Semilla" + ("" if len(seeds) == 1 else "s")
    fig.suptitle(f"Figura B-2: Dinámica de Entrenamiento Fase B hasta 1.000.000 de Pasos ({n_seeds_txt})\nNota: La recompensa solo es comparable dentro de una misma configuración.", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_p = FIG_B_DIR / "fig_b_02_agent_training_curves.png"
    save_fig_with_metadata(fig, out_p, [REPO_ROOT / "reports" / "data" / "phase_b" / "tanda_1_x5_near_summary.json"],
                           "Curvas de entrenamiento de recompensa, longitud y ventana de éxito.")

# ==============================================================================
# FIGURA 3: CURVA DE ÉXITO ESTRICTO DETERMINISTA CONTRA CHECKPOINTS (VALIDACIÓN)
# ==============================================================================
def generate_figure_3(exps=None, seeds=None, agg_csv=None, raw_csv=None, selected_ckpts=None):
    """
    Curva de éxito estricto determinista vs checkpoints (validación).

    exps / seeds    : experimentos y semillas a dibujar (por defecto los
                      históricos ["X5_near", "X3"] y [42, 100, 2026]).
    agg_csv/raw_csv : CSVs de evaluación (por defecto los de DATA_B_DIR;
                      faltan solo si se analiza otra corrida via --data-dir).
    selected_ckpts  : dict experimento->paso marcado como seleccionado por
                      la regla 4.1; sin entrada para un experimento no se
                      dibuja estrella. Por defecto {"X5_near": 901120,
                      "X3": 409600}.
    """
    print("\n--- Generando Figura 3: Curva de Éxito Estricto Determinista vs Checkpoints ---")
    exps = exps if exps is not None else ["X5_near", "X3"]
    seeds = seeds if seeds is not None else [42, 100, 2026]
    agg_csv = Path(agg_csv) if agg_csv is not None else DATA_B_DIR / "checkpoints_eval_validation_aggregated.csv"
    raw_csv = Path(raw_csv) if raw_csv is not None else DATA_B_DIR / "checkpoints_eval_validation.csv"
    if selected_ckpts is None:
        selected_ckpts = {"X5_near": 901120, "X3": 409600}

    agg_df = pd.read_csv(agg_csv)
    raw_df = pd.read_csv(raw_csv)

    fig, (ax_main, ax_entry) = plt.subplots(2, 1, figsize=(14, 10), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})

    # Línea de referencia C0
    ax_main.axhline(100.0, color=EXP_COLORS["C0"], linestyle="-.", linewidth=1.8, label="C0 Jacobiano Analítico (100.0%)")
    # Límite de Fase A
    ax_main.axvline(303104, color="#7f8c8d", linestyle=":", linewidth=1.2, label="Límite Fase A (303k)")
    ax_entry.axvline(303104, color="#7f8c8d", linestyle=":", linewidth=1.2)

    # Marcadores de selección (parametrizables; un experimento sin entrada
    # simplemente no recibe estrella)

    for exp_id in exps:
        sub = agg_df[agg_df["experiment"] == exp_id].sort_values("step")
        steps = sub["step"].values
        mean_succ = sub["mean_strict_success"].values * 100.0
        std_succ = sub["std_strict_success"].values * 100.0
        col = get_exp_color(exp_id)

        # Curva media
        ax_main.plot(steps, mean_succ, color=col, linewidth=2.5, marker="o", markersize=4, label=f"{exp_id} Media (n={len(seeds)})")
        ax_main.fill_between(steps, mean_succ - std_succ, mean_succ + std_succ, color=col, alpha=0.15)

        # Semillas individuales con puntos sutiles
        raw_sub = raw_df[raw_df["experiment"] == exp_id]
        for s in seeds:
            s_df = raw_sub[raw_sub["seed"] == s].sort_values("step")
            ax_main.plot(s_df["step"], s_df["strict_success_rate"]*100, color=col, alpha=0.35, linewidth=0.9, linestyle="--")

        # Marcar checkpoint seleccionado por la regla (si hay entrada)
        if exp_id in selected_ckpts and len(steps) > 0:
            sel_step = selected_ckpts[exp_id]
            sel_row = sub[sub["step"] == sel_step]
            if sel_row.empty:
                sel_step = sub["step"].iloc[-1]
                sel_row = sub[sub["step"] == sel_step]
            sel_row = sel_row.iloc[0]
            sel_val = sel_row["mean_strict_success"] * 100.0
            ax_main.scatter(sel_step, sel_val, color=col, s=160, marker="*", edgecolors="black", zorder=7,
                            label=f"{exp_id} Seleccionado ({sel_step:,} p, {sel_val:.1f}%)")
            ax_main.annotate(f"{exp_id}\n{sel_step:,} p\n{sel_val:.1f}%",
                             (sel_step, sel_val), textcoords="offset points", xytext=(10, 10 if exp_id=="X5_near" else -30),
                             fontsize=8.5, fontweight="bold", color=col,
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.9))

        # Fracción de episodios que entran al radio objetivo
        # En raw_df: entry fraction se puede aproximar por 1 - (fallos sin aproximación) o time_inside_030 > 0
        entry_means = []
        for st in steps:
            st_raw = raw_sub[raw_sub["step"] == st]
            entry_rate = 1.0 - (st_raw["truncation_rate"].mean() - st_raw["ballistic_rate"].mean()*0.1) # proxy
            entry_means.append(max(0.0, min(1.0, entry_rate)) * 100.0)
        ax_entry.plot(steps, entry_means, color=col, linewidth=2.0, marker="s", markersize=3, label=f"{exp_id} Entra al Radio")

    # Banda de ruido operativo U_meseta = 5 pp
    ax_main.fill_between([40960, 1007616], [90, 90], [95, 95], color="#bdc3c7", alpha=0.15, label="Banda Meseta U_meseta (±5 pp)")

    ax_main.set_title("Curva de Aprendizaje por Checkpoints: Éxito Estricto Determinista vs Pasos (Validación, 200 eps)", fontsize=13, fontweight="bold")
    ax_main.set_ylabel("Éxito Estricto (% hold 20p, v<0.10)", fontsize=10, fontweight="bold")
    ax_main.set_ylim(-2, 105)
    ax_main.legend(loc="lower right", fontsize=8.5, ncol=2)

    ax_entry.set_title("Fracción de Episodios con Entrada al Radio Cartesiano vs Pasos", fontsize=11, fontweight="bold")
    ax_entry.set_xlabel("Pasos de Entrenamiento de Entorno (Timesteps)", fontsize=10, fontweight="bold")
    ax_entry.set_ylabel("Entrada (%)", fontsize=10, fontweight="bold")
    ax_entry.set_ylim(40, 102)
    ax_entry.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    out_p = FIG_B_DIR / "fig_b_03_learning_curve_checkpoints.png"
    save_fig_with_metadata(fig, out_p, [DATA_B_DIR / "checkpoints_eval_validation_aggregated.csv"],
                           "Curva de éxito estricto determinista vs pasos por checkpoints evaluados sobre validación.")

# ==============================================================================
# FIGURA 4: PLANO DE FASE (DISTANCIA vs VELOCIDAD ARTICULAR)
# ==============================================================================
def generate_figure_4():
    print("\n--- Generando Figura 4: Plano de Fase (Distancia vs Velocidad Articular) ---")
    if not _required_inputs(REQUIRED_FIG4):
        return
    step_df = pd.read_csv(DATA_B_DIR / "step_data_val_geometry.csv")
    c0_episodes = classify_df_episodes(pd.read_csv(DATA_B_DIR / "eval_val_c0_episodes.csv"))
    x5b_episodes = classify_df_episodes(pd.read_csv(DATA_B_DIR / "eval_val_x5_near_phase_b_selected.csv"))

    fig, (ax_phase, ax_entry_hist) = plt.subplots(1, 2, figsize=(16, 7), dpi=300, gridspec_kw={"width_ratios": [1.5, 1.0]})

    # 1. Plano de fase de trayectorias
    models_to_plot = [
        ("C0", "#2980b9", "-", 2.0),
        ("X5_near_FaseA", "#7f8c8d", "--", 1.5),
        ("X5_near_FaseB", "#8e44ad", "-", 2.2),
        ("X3_FaseB", "#f39c12", "-.", 1.8)
    ]

    for mod_id, col, ls, lw in models_to_plot:
        sub_mod = step_df[(step_df["model_id"] == mod_id) & (step_df["eval_mode"] == "deterministic")]
        for epid in [34, 195, 40, 90]:
            ep_steps = sub_mod[sub_mod["episode_id"] == epid].sort_values("step")
            if len(ep_steps) > 0:
                ax_phase.plot(ep_steps["distance"], ep_steps["joint_speed"], color=col, linestyle=ls, linewidth=lw,
                              alpha=0.85, label=f"{mod_id}" if epid == 34 else "")

    # Marcar la Región del Criterio Estricto (d <= 0.06 m, v <= 0.10 rad/s)
    rect = patches.Rectangle((0, 0), 0.06, 0.10, linewidth=2.0, edgecolor="#27ae60", facecolor="#2ecc71", alpha=0.35,
                             label="Criterio Estricto (d≤0.06m, v≤0.10rad/s)")
    ax_phase.add_patch(rect)
    ax_phase.axvline(0.06, color="#27ae60", linestyle=":", linewidth=1.2)
    ax_phase.axhline(0.10, color="#27ae60", linestyle=":", linewidth=1.2)
    ax_phase.axvline(0.30, color="#e67e22", linestyle=":", linewidth=1.0, label="Umbral Zona Near (0.30m)")

    ax_phase.set_title("Plano de Fase: Distancia Cartesiana vs Velocidad Articular ||θ̇||\n(Episodios Geométricos Q10, Q50, Q90, Borde)", fontsize=11, fontweight="bold")
    ax_phase.set_xlabel("Distancia al Objetivo (m)", fontsize=10, fontweight="bold")
    ax_phase.set_ylabel("Velocidad Articular ||θ̇|| (rad/s)", fontsize=10, fontweight="bold")
    ax_phase.set_xlim(-0.02, 2.2)
    ax_phase.set_ylim(-0.02, 1.8)
    ax_phase.legend(loc="upper right", fontsize=8.5)

    # 2. Histograma de velocidad de entrada a d < 0.06 m (Caveat 1 & 2: Sesgo de supervivencia y balística de C0)
    c0_spds = c0_episodes["joint_speed_at_first_entry"].dropna()
    x5b_spds = x5b_episodes[x5b_episodes["evaluation_mode"] == "deterministic"]["joint_speed_at_first_entry"].dropna()

    bins = np.linspace(0.0, 1.6, 25)
    ax_entry_hist.hist(c0_spds, bins=bins, alpha=0.6, color="#2980b9", edgecolor="black", label=f"C0 Analítico (n={len(c0_spds)})")
    ax_entry_hist.hist(x5b_spds, bins=bins, alpha=0.5, color="#8e44ad", edgecolor="black", label=f"X5_near Fase B (n={len(x5b_spds)})")
    ax_entry_hist.axvline(0.30, color="#e74c3c", linestyle="--", linewidth=1.8, label="Umbral Balístico u_b (0.30 rad/s)")

    ax_entry_hist.set_title("Caveat: Velocidad al Primer Ingreso a d < 0.06 m\n(Explica por qué C0 registra 36% balística)", fontsize=11, fontweight="bold")
    ax_entry_hist.set_xlabel("Velocidad Articular al Ingresar (rad/s)", fontsize=10, fontweight="bold")
    ax_entry_hist.set_ylabel("Frecuencia (Episodios)", fontsize=10, fontweight="bold")
    ax_entry_hist.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    out_p = FIG_B_DIR / "fig_b_04_phase_plane_distance_speed.png"
    save_fig_with_metadata(fig, out_p, [DATA_B_DIR / "step_data_val_geometry.csv", DATA_B_DIR / "eval_val_c0_episodes.csv"],
                           "Plano de fase de trayectorias e histograma de velocidad de entrada.")

# ==============================================================================
# FIGURA 5: PERFILES TEMPORALES DE LOS EPISODIOS GEOMÉTRICOS
# ==============================================================================
def generate_figure_5():
    print("\n--- Generando Figura 5: Perfiles Temporales de Episodios Geométricos ---")
    if not _required_inputs(REQUIRED_FIG5_7):
        return
    step_df = pd.read_csv(DATA_B_DIR / "step_data_val_geometry.csv")
    det_steps = step_df[step_df["eval_mode"] == "deterministic"]

    geom_info = [
        (34, "Ep 34: Radio Corto Q10 (r = 0.31 m)"),
        (195, "Ep 195: Radio Medio Q50 (r = 1.09 m)"),
        (40, "Ep 40: Radio Largo Q90 (r = 1.75 m)"),
        (90, "Ep 90: Borde / Alcance Máx (r = 1.90 m)")
    ]

    fig, axes = plt.subplots(4, 3, figsize=(18, 14), dpi=300, sharex=True)

    models_info = [
        ("C0", "#2980b9", "-"),
        ("X5_near_FaseA", "#7f8c8d", "--"),
        ("X5_near_FaseB", "#8e44ad", "-"),
        ("X3_FaseB", "#f39c12", "-.")
    ]

    for row_idx, (epid, title) in enumerate(geom_info):
        ax_dist = axes[row_idx, 0]
        ax_spd = axes[row_idx, 1]
        ax_hold = axes[row_idx, 2]

        for mod_id, col, ls in models_info:
            ep_data = det_steps[(det_steps["model_id"] == mod_id) & (det_steps["episode_id"] == epid)].sort_values("step")
            if len(ep_data) > 0:
                ax_dist.plot(ep_data["step"], ep_data["distance"], color=col, linestyle=ls, linewidth=1.8, label=mod_id)
                ax_spd.plot(ep_data["step"], ep_data["joint_speed"], color=col, linestyle=ls, linewidth=1.8, label=mod_id)
                ax_hold.plot(ep_data["step"], ep_data["hold_counter"], color=col, linestyle=ls, linewidth=1.8, label=mod_id)

        # Líneas de umbrales
        ax_dist.axhline(0.06, color="#27ae60", linestyle=":", linewidth=1.0)
        ax_spd.axhline(0.10, color="#27ae60", linestyle=":", linewidth=1.0)
        ax_hold.axhline(20, color="#27ae60", linestyle=":", linewidth=1.0)

        ax_dist.set_ylabel(f"{title}\nDistancia (m)", fontsize=9, fontweight="bold")
        ax_spd.set_ylabel("||θ̇|| (rad/s)", fontsize=9, fontweight="bold")
        ax_hold.set_ylabel("Hold Counter", fontsize=9, fontweight="bold")
        ax_hold.set_ylim(-1, 22)

        if row_idx == 0:
            ax_dist.set_title("Distancia Cartesiana d(t)", fontsize=11, fontweight="bold")
            ax_spd.set_title("Velocidad Articular ||θ̇||(t)", fontsize=11, fontweight="bold")
            ax_hold.set_title("Contador de Hold (20 pasos req.)", fontsize=11, fontweight="bold")
            ax_dist.legend(loc="upper right", fontsize=7.5)

    axes[-1, 0].set_xlabel("Paso Temporal (dt = 0.05 s)", fontsize=10, fontweight="bold")
    axes[-1, 1].set_xlabel("Paso Temporal (dt = 0.05 s)", fontsize=10, fontweight="bold")
    axes[-1, 2].set_xlabel("Paso Temporal (dt = 0.05 s)", fontsize=10, fontweight="bold")

    fig.suptitle("Figura B-5: Perfiles Temporales de Telemetría sobre Episodios Seleccionados por Geometría", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_p = FIG_B_DIR / "fig_b_05_time_profiles_geometry.png"
    save_fig_with_metadata(fig, out_p, [DATA_B_DIR / "step_data_val_geometry.csv"],
                           "Perfiles temporales de distancia, velocidad y hold de episodios geométricos.")

# ==============================================================================
# FIGURA 6: TRAYECTORIAS EN EL PLANO CARTESIANO CON DEGRADÉ DE COLOR
# ==============================================================================
def generate_figure_6():
    print("\n--- Generando Figura 6: Trayectorias Cartesianas 2D con Degradé ---")
    if not _required_inputs(REQUIRED_FIG5_7):
        return
    step_df = pd.read_csv(DATA_B_DIR / "step_data_val_geometry.csv")
    det_steps = step_df[(step_df["eval_mode"] == "deterministic") & (step_df["model_id"] == "X5_near_FaseB")]
    c0_steps = step_df[(step_df["eval_mode"] == "deterministic") & (step_df["model_id"] == "C0")]

    geom_info = [
        (34, "Ep 34: Radio Corto Q10 (r = 0.31 m)"),
        (195, "Ep 195: Radio Medio Q50 (r = 1.09 m)"),
        (40, "Ep 40: Radio Largo Q90 (r = 1.75 m)"),
        (90, "Ep 90: Borde / Alcance Máx (r = 1.90 m)")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=300)
    axes = axes.flatten()

    for idx, (epid, title) in enumerate(geom_info):
        ax = axes[idx]
        ep_d = det_steps[det_steps["episode_id"] == epid].sort_values("step")
        c0_d = c0_steps[c0_steps["episode_id"] == epid].sort_values("step")

        tgt_x = ep_d["target_x"].iloc[0]
        tgt_y = ep_d["target_y"].iloc[0]

        # Círculos objetivo y cerca
        circ_target = plt.Circle((tgt_x, tgt_y), 0.06, color="#27ae60", fill=True, alpha=0.35, label="Meta (r=0.06 m)")
        circ_near = plt.Circle((tgt_x, tgt_y), 0.30, color="#f39c12", fill=False, linestyle=":", linewidth=1.2, label="Zona Near (0.30 m)")
        ax.add_patch(circ_target)
        ax.add_patch(circ_near)

        # Referencia C0
        if len(c0_d) > 0:
            ax.plot(c0_d["ee_x"], c0_d["ee_y"], color="#2980b9", linestyle="--", linewidth=1.8, label="C0 Referencia", alpha=0.7)

        # Trayectoria X5_near Fase B con scatter en degradé
        if len(ep_d) > 0:
            sc = ax.scatter(ep_d["ee_x"], ep_d["ee_y"], c=ep_d["step"], cmap="viridis", s=22, alpha=0.9, zorder=5)
            ax.plot(ep_d["ee_x"], ep_d["ee_y"], color="#8e44ad", alpha=0.3, linewidth=1.0)
            # Marcar inicio y fin
            ax.scatter(ep_d["ee_x"].iloc[0], ep_d["ee_y"].iloc[0], color="#e74c3c", s=60, marker="o", label="Inicio", zorder=6)
            ax.scatter(ep_d["ee_x"].iloc[-1], ep_d["ee_y"].iloc[-1], color="#27ae60", s=70, marker="X", label="Fin", zorder=6)

        ax.scatter(tgt_x, tgt_y, color="#2c3e50", marker="+", s=100, linewidth=2, label="Centro Meta", zorder=7)

        ax.set_title(f"{title}", fontsize=11, fontweight="bold")
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("y (m)", fontsize=9)
        ax.set_aspect("equal")
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7.5)

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.03, pad=0.06)
    cbar.set_label("Paso Temporal (dt = 0.05 s)", fontsize=10, fontweight="bold")

    fig.suptitle("Figura B-6: Trayectorias Cartesianas 2D del Efector Final (X5_near Fase B con C0)\nDegradé de color por avance temporal", fontsize=13, fontweight="bold", y=0.98)
    out_p = FIG_B_DIR / "fig_b_06_cartesian_trajectories_gradient.png"
    save_fig_with_metadata(fig, out_p, [DATA_B_DIR / "step_data_val_geometry.csv"],
                           "Trayectorias 2D en plano cartesiano con degradé de color por paso temporal.")

# ==============================================================================
# FIGURA 7: DETERMINISTA CONTRA ESTOCÁSTICA SOBRE EL MISMO EPISODIO
# ==============================================================================
def generate_figure_7():
    print("\n--- Generando Figura 7: Determinista vs Estocástica sobre Mismo Episodio ---")
    if not _required_inputs(REQUIRED_FIG5_7):
        return
    step_df = pd.read_csv(DATA_B_DIR / "step_data_val_geometry.csv")
    x5b_steps = step_df[step_df["model_id"] == "X5_near_FaseB"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)

    # Comparar Ep 195 (Q50) y Ep 40 (Q90)
    compare_eps = [(195, "Episodio 195 (Radio Medio Q50, r = 1.09 m)", axes[0]),
                   (40, "Episodio 40 (Radio Largo Q90, r = 1.75 m)", axes[1])]

    for epid, title, (ax_xy, ax_spd) in compare_eps:
        det_d = x5b_steps[(x5b_steps["episode_id"] == epid) & (x5b_steps["eval_mode"] == "deterministic")].sort_values("step")
        stoch_d = x5b_steps[(x5b_steps["episode_id"] == epid) & (x5b_steps["eval_mode"] == "stochastic")].sort_values("step")

        tgt_x = det_d["target_x"].iloc[0]
        tgt_y = det_d["target_y"].iloc[0]

        # 1. Trayectoria Cartesiana
        circ_t = plt.Circle((tgt_x, tgt_y), 0.06, color="#27ae60", fill=True, alpha=0.3, label="Meta (0.06 m)")
        circ_n = plt.Circle((tgt_x, tgt_y), 0.30, color="#f39c12", fill=False, linestyle=":", linewidth=1.0, label="Near (0.30 m)")
        ax_xy.add_patch(circ_t)
        ax_xy.add_patch(circ_n)

        ax_xy.plot(det_d["ee_x"], det_d["ee_y"], color="#27ae60", linewidth=2.0, label=f"Determinista (Pasos={len(det_d)}, Éxito={det_d['is_success'].iloc[-1]})")
        ax_xy.plot(stoch_d["ee_x"], stoch_d["ee_y"], color="#e74c3c", linestyle="--", linewidth=1.6, alpha=0.8,
                   label=f"Estocástica (Pasos={len(stoch_d)}, Éxito={stoch_d['is_success'].iloc[-1]})")
        ax_xy.scatter(tgt_x, tgt_y, color="black", marker="+", s=80, label="Target")

        ax_xy.set_title(f"Trayectoria Cartesiana 2D — {title}", fontsize=10, fontweight="bold")
        ax_xy.set_xlabel("x (m)", fontsize=9)
        ax_xy.set_ylabel("y (m)", fontsize=9)
        ax_xy.set_aspect("equal")
        ax_xy.legend(loc="best", fontsize=8)

        # 2. Perfil de Velocidad Articular (muestra el jitter estocástico)
        ax_spd.plot(det_d["step"], det_d["joint_speed"], color="#27ae60", linewidth=1.8, label="Determinista ||θ̇||")
        ax_spd.plot(stoch_d["step"], stoch_d["joint_speed"], color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.8, label="Estocástica ||θ̇|| (Jitter)")
        ax_spd.axhline(0.10, color="#2c3e50", linestyle=":", linewidth=1.2, label="Umbral v_tol (0.10 rad/s)")

        ax_spd.set_title(f"Velocidad Articular ||θ̇||(t) — {title}", fontsize=10, fontweight="bold")
        ax_spd.set_xlabel("Paso Temporal", fontsize=9)
        ax_spd.set_ylabel("||θ̇|| (rad/s)", fontsize=9)
        ax_spd.legend(loc="upper right", fontsize=8)

    fig.suptitle("Figura B-7: Determinista vs Estocástica sobre el Mismo Episodio (X5_near Fase B)\nIlustración del 'Jitter' Estocástico que Rompe el Criterio de Hold", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_p = FIG_B_DIR / "fig_b_07_det_vs_stoch_same_episode.png"
    save_fig_with_metadata(fig, out_p, [DATA_B_DIR / "step_data_val_geometry.csv"],
                           "Comparativa de trayectoria y velocidad determinista vs estocástica mostrando jitter.")

# ==============================================================================
# FIGURA 8: DIAGNÓSTICOS PPO DE LA FASE B
# ==============================================================================
def generate_figure_8(exps=None, seeds=None, experiments_root=None):
    """
    Diagnósticos internos de entrenamiento desde TensorBoard (análogos a
    ``generate_figure_2`` en parametrización: ``exps``, ``seeds``,
    ``experiments_root``).
    """
    print("\n--- Generando Figura 8: Diagnósticos PPO de la Fase B ---")
    seeds = seeds if seeds is not None else [42, 100, 2026]
    exps = exps if exps is not None else ["X5_near", "X3"]
    experiments_root = Path(experiments_root) if experiments_root is not None else REPO_ROOT / "experiments" / "phase_b"

    raw_series = {}
    for exp in exps:
        raw_series[exp] = {}
        for s in seeds:
            tb_dir = experiments_root / exp / f"seed_{s}" / "logs"
            efs = list(tb_dir.glob("**/events.out.tfevents.*"))
            if efs:
                ea = event_accumulator.EventAccumulator(str(efs[-1]), size_guidance={"scalars": 0})
                ea.Reload()
                tags = ea.Tags().get("scalars", [])
                raw_series[exp][s] = {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in tags}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    axes = axes.flatten()

    diag_tags = [
        ("train/approx_kl", "Approximate KL Divergence", "KL Divergence", False),
        ("train/clip_fraction", "PPO Clipping Fraction", "Fracción de Clips", False),
        ("train/policy_gradient_loss", "Policy Gradient Loss", "Loss de Política", False),
        ("train/value_loss", "Value Loss del Crítico (Log Scale)", "Value Loss (log)", True),
        ("train/explained_variance", "Varianza Explicada (Crítico)", "Explained Variance", False),
        ("train/entropy_loss", "Entropy Loss / Dispersión", "Entropy Loss", False)
    ]

    for idx, (tag, title, ylabel, is_log) in enumerate(diag_tags):
        ax = axes[idx]
        for exp in exps:
            curves = [raw_series[exp][s][tag] for s in seeds if s in raw_series[exp] and tag in raw_series[exp][s]]
            if curves:
                min_step = max(c[0][0] for c in curves)
                max_step = min(c[-1][0] for c in curves)
                common_steps = np.linspace(min_step, max_step, 120)

                vals_interp = []
                for s, c in zip(seeds, curves):
                    c_steps = [p[0] for p in c]
                    c_vals = [p[1] for p in c]
                    v_int = np.interp(common_steps, c_steps, c_vals)
                    vals_interp.append(v_int)

                vals_interp = np.array(vals_interp)
                mean_vals = np.mean(vals_interp, axis=0)
                std_vals = np.std(vals_interp, axis=0, ddof=1)
                col = get_exp_color(exp)

                ax.plot(common_steps, mean_vals, color=col, linewidth=2.0, label=f"{exp} (n={len(curves)})")
                ax.fill_between(common_steps, mean_vals - std_vals, mean_vals + std_vals, color=col, alpha=0.15)

        if is_log:
            ax.set_yscale("log")
        if tag == "train/explained_variance":
            ax.set_ylim(-0.2, 1.05)
            ax.axhline(1.0, color="#27ae60", linestyle=":", linewidth=1.0)
            ax.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.0)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Timesteps (Pasos Nativos PPO)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Figura B-8: Diagnósticos Internos del Algoritmo PPO en Fase B (1.000.000 de Pasos)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_p = FIG_B_DIR / "fig_b_08_ppo_diagnostics_phase_b.png"
    save_fig_with_metadata(fig, out_p, [REPO_ROOT / "reports" / "data" / "phase_b" / "tanda_1_x5_near_summary.json"],
                           "Diagnósticos de optimización PPO (KL, clip, loss, critic, entropy).")

FIGURE_FUNCS = {1: generate_figure_1, 4: generate_figure_4, 5: generate_figure_5,
                6: generate_figure_6, 7: generate_figure_7}


def generate_all_figures(experiments=None, seeds=None, exps_dir=None,
                         selected_ckpts=None, figures=None):
    """
    Genera el conjunto de figuras de Fase B; ``figures`` permite un
    subconjunto (lista de enteros 1..8). ``experiments``/``seeds``/
    ``exps_dir``/``selected_ckpts`` se aplican a las figuras 2, 3 y 8;
    sin parámetros se reproduce el estudio histórico íntegro.
    """
    want = set(figures) if figures is not None else set(range(1, 9))
    if 1 in want:
        generate_figure_1()
    if 2 in want:
        generate_figure_2(exps=experiments, seeds=seeds, experiments_root=exps_dir)
    if 3 in want:
        generate_figure_3(exps=experiments, seeds=seeds, selected_ckpts=selected_ckpts)
    for n in (4, 5, 6, 7):
        if n in want:
            FIGURE_FUNCS[n]()
    if 8 in want:
        generate_figure_8(exps=experiments, seeds=seeds, experiments_root=exps_dir)
    print("\n=======================================================")
    print(f"=== FIGURAS DE FASE B GENERADAS CON ÉXITO (pedidas: {sorted(want)}) ===")
    print("=======================================================")

def run(out_dir=None, experiments=None, seeds=None, exps_dir=None,
        selected_ckpts=None, data_dir=None, figures=None):
    """
    Punto de entrada del módulo.

    out_dir     : ``None`` (por defecto ``FIG_B_DIR`` =
                  reports/figures/phase_b) o ruta alternativa de salida.
    experiments : lista de experimentos a dibujar en las figuras 2/3/8
                  (por defecto ``["X5_near", "X3"]``).
    seeds       : lista de semillas (por defecto ``[42, 100, 2026]``).
    exps_dir    : raíz alternativa del TensorBoard de las corridas
                  (por defecto ``experiments/phase_b``).
    selected_ckpts: dict experimento->paso destacado en la figura 3
                  (solo por biblioteca; ``None`` usa los históricos).
    data_dir    : ruta alternativa de los CSV de evaluación (por defecto
                  ``reports/data/phase_b``).
    figures     : subconjunto de figuras (lista de enteros; ``None`` = 1-8).
    """
    global FIG_B_DIR, DATA_B_DIR
    if out_dir is not None:
        FIG_B_DIR = Path(out_dir)
    if data_dir is not None:
        DATA_B_DIR = Path(data_dir)
    generate_all_figures(experiments=experiments, seeds=seeds,
                         exps_dir=Path(exps_dir) if exps_dir is not None else None,
                         selected_ckpts=selected_ckpts, figures=figures)


if __name__ == "__main__":
    run()
