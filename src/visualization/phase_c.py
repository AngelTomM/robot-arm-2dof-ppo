"""
Figuras de Fase C (SAC) - 300 dpi + JSON de trazabilidad.

FIGURAS GENERADAS
=================
fig_c_learning_curves_s1_vs_x3, fig_c_sac_training_diagnostics,
fig_c_failure_breakdown_final_ckpt, fig_c_canonical_geometric_trajectories,
fig_c_deterministic_vs_stochastic_profiles y
fig_c_compute_cost_and_runtime_comparison en reports/figures/phase_c/.

INSUMOS REQUERIDOS
==================
- reports/data/phase_c/{checkpoints_eval_validation_aggregated.csv,
  checkpoints_eval_validation_details.csv, step_data_val_geometry.csv}
- experiments/phase_c/S1/ (diagnósticos de entrenamiento SAC, opcional)

USO
===
python scripts/generate_figures.py --phase c
python scripts/generate_figures.py --phase c --out-dir <ruta_temporal>
python -m src.visualization.phase_c                      # equivalente directo

PARÁMETROS CONFIGURABLES
========================
data_dir / figures_dir : insumos y salida (por defecto
               "reports/data/phase_c" y "reports/figures/phase_c");
               ``run(out_dir)`` / ``--out-dir`` sustituye solo figures_dir
               (``--data-dir`` también admite insumos alternativos y
               ``--exps-dir`` la corrida SAC de la figura 2).
AGENT_COLORS : paleta (S1, X3, C0 y variantes estocásticas). Los agentes
               ausentes en la paleta reciben un color del ciclo neutro por
               orden de aparición (ver common.color_resolver), p. ej. un
               modelo recién entrenado al invocar
               ``run(..., experiments=["MiModelo"])``.
experiments  : agentes a dibujar (nombres tal como aparecen en la columna
               ``experiment`` de los CSV); sin bandera se usan los
               históricos ["X3", "S1"] en las figuras 1 y 3.
figures      : subconjunto de figuras (enteros 1..6; bandera ``--figures``).
step_filter  : filtro de checkpoint de la figura de desglose de fallos
               (plot_fig3_failure_breakdown). Valores posibles:
               655360 (por defecto, checkpoint final del estudio),
               245760 (último checkpoint del presupuesto del estudio S3)
               o None (sin filtro).
time_s1_sec / time_x3_sec : tiempos de entrenamiento de referencia
               (segundos, medidos en GPU) usados en la fig6;
               por defecto 4600.0 y 720.0.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.visualization import common
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

AGENT_COLORS = {
    "S1": "#d62728",       # Rojo para SAC S1
    "X3": "#1f77b4",       # Azul para PPO X3
    "C0": "#000000",       # Negro para Controlador Analítico C0
    "S1_sto": "#ff9896",   # Rojo claro para estocástico
    "X3_sto": "#aec7e8"    # Azul claro para estocástico
}

# Resolución de color: canónicos de AGENT_COLORS + ciclo neutro determinista
# para agentes nuevos (common.color_resolver).
get_agent_color = common.color_resolver(AGENT_COLORS)

def save_fig_with_metadata(fig, output_png_path: Path, metadata: dict):
    """Guarda figura a 300 dpi con JSON de metadatos libres (Fase C)."""
    common.save_fig_with_metadata(fig, output_png_path, metadata=metadata)


def plot_fig1_learning_curves(agg_df: pd.DataFrame, details_df: pd.DataFrame, out_dir: Path,
                              experiments=None):
    """Figura 1: Curva de éxito determinista y panel de truncamientos vs pasos.

    experiments : agentes a dibujar (por defecto los históricos
                  ["X3", "S1"]; nombres no canónicos usan el ciclo neutro)."""
    agents = experiments if experiments is not None else ["X3", "S1"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})

    # Panel superior: Éxito estricto
    for agent in agents:
        sub_agg = agg_df[agg_df["experiment"] == agent].sort_values("step")
        if sub_agg.empty:
            continue
        x = sub_agg["step"] / 1000.0
        y = sub_agg["mean_success_rate"] * 100.0
        std = sub_agg["std_success_rate"] * 100.0
        color = get_agent_color(agent)

        # Curva media con banda +- std
        ax1.plot(x, y, label=f"{agent} Media (n=3)", color=color, linewidth=2.5, marker="o", markersize=4.5)
        ax1.fill_between(x, np.maximum(0, y - std), np.minimum(100, y + std), color=color, alpha=0.18)

        # Semillas individuales tenues
        sub_det = details_df[details_df["experiment"] == agent]
        for s, s_grp in sub_det.groupby("seed"):
            s_sorted = s_grp.sort_values("step")
            ax1.plot(s_sorted["step"] / 1000.0, s_sorted["success_rate"] * 100.0,
                     color=color, linestyle=":", alpha=0.45, linewidth=1.0)

    # C0 Línea de referencia (etiqueta con su éxito observado)
    c0_sub = details_df[details_df["experiment"].str.startswith("C0")]
    if not c0_sub.empty:
        c0_succ = float(c0_sub["success_rate"].iloc[0]) * 100.0
        ax1.axhline(c0_succ, color=AGENT_COLORS["C0"], linestyle="--", linewidth=1.8, label=f"C0 Analítico ({c0_succ:.1f}%)")

    ax1.set_title("Fase C — Comparación Descriptiva de Aprendizaje: SAC (S1) vs PPO (X3)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Éxito Estricto (%) [Hold 20 pasos, v ≤ 0.10 rad/s]", fontsize=10.5)
    ax1.set_ylim(-2, 104)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

    # Panel inferior: Truncamientos
    for agent in agents:
        sub_agg = agg_df[agg_df["experiment"] == agent].sort_values("step")
        if sub_agg.empty:
            continue
        x = sub_agg["step"] / 1000.0
        y_trunc = sub_agg["mean_truncation_rate"] * 100.0
        color = get_agent_color(agent)
        ax2.plot(x, y_trunc, label=f"{agent} Truncamientos", color=color, linewidth=2.0, linestyle="-")

    ax2.set_xlabel("Pasos de Interacción en el Entorno (miles de pasos, k)", fontsize=10.5)
    ax2.set_ylabel("Truncamientos (%)", fontsize=10.5)
    ax2.set_ylim(-2, 102)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9.0)

    meta = {
        "figure_id": "fig_c_learning_curves_s1_vs_x3",
        "dataset": "validation_c_500.json",
        "eval_mode": "deterministic",
        "n_seeds": int(details_df[details_df["experiment"].isin(agents)]["seed"].nunique()),
        "x_units": "k timesteps",
        "y_units": "percentage (%)"
    }
    save_fig_with_metadata(fig, out_dir / "fig_c_learning_curves_s1_vs_x3.png", meta)

def plot_fig2_sac_training_diagnostics(exp_s1_dir: Path, out_dir: Path):
    """Figura 2: Diagnósticos internos de entrenamiento de SAC leídos desde TensorBoard."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        EventAccumulator = None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=300)
    ((ax_q, ax_pi), (ax_alpha, ax_rew)) = axes

    colors = {
        "seed_42": "#d62728",    # Rojo
        "seed_100": "#2ca02c",   # Verde
        "seed_2026": "#9467bd"   # Violeta
    }

    has_data = False
    metrics_summary = {}

    if EventAccumulator is not None:
        for seed_dir in sorted(exp_s1_dir.glob("seed_*")):
            sname = seed_dir.name
            tfevent_files = list((seed_dir / "logs" / "SAC_1").glob("events*"))
            if not tfevent_files:
                # Buscar en subdirectorios de logs
                tfevent_files = list(seed_dir.glob("**/events*"))
            if not tfevent_files:
                continue

            ea = EventAccumulator(str(tfevent_files[0]))
            ea.Reload()
            scalars = ea.Tags().get("scalars", [])
            c = colors.get(sname, "#333333")
            has_data = True
            metrics_summary[sname] = {}

            # 1. Pérdida de los Críticos (Critic Loss / MSE)
            if "train/critic_loss" in scalars:
                evs = ea.Scalars("train/critic_loss")
                df = pd.DataFrame({"step": [e.step for e in evs], "val": [e.value for e in evs]})
                df["roll"] = df["val"].rolling(window=20, min_periods=3).mean()
                ax_q.plot(df["step"] / 1000.0, df["val"], color=c, alpha=0.18, linewidth=0.8)
                ax_q.plot(df["step"] / 1000.0, df["roll"], color=c, label=f"{sname}", linewidth=1.8)
                metrics_summary[sname]["critic_loss_final"] = float(df["val"].iloc[-1])

            # 2. Pérdida del Actor (Policy Loss)
            if "train/actor_loss" in scalars:
                evs = ea.Scalars("train/actor_loss")
                df = pd.DataFrame({"step": [e.step for e in evs], "val": [e.value for e in evs]})
                df["roll"] = df["val"].rolling(window=20, min_periods=3).mean()
                ax_pi.plot(df["step"] / 1000.0, df["val"], color=c, alpha=0.18, linewidth=0.8)
                ax_pi.plot(df["step"] / 1000.0, df["roll"], color=c, label=f"{sname}", linewidth=1.8)
                metrics_summary[sname]["actor_loss_final"] = float(df["val"].iloc[-1])

            # 3. Coeficiente de Entropía / Temperatura (ent_coef / alpha)
            if "train/ent_coef" in scalars:
                evs = ea.Scalars("train/ent_coef")
                df = pd.DataFrame({"step": [e.step for e in evs], "val": [e.value for e in evs]})
                ax_alpha.plot(df["step"] / 1000.0, df["val"], color=c, label=f"{sname}", linewidth=1.8)
                metrics_summary[sname]["ent_coef_final"] = float(df["val"].iloc[-1])

            # 4. Retorno Acumulado por Episodio (rollout/ep_rew_mean)
            if "rollout/ep_rew_mean" in scalars:
                evs = ea.Scalars("rollout/ep_rew_mean")
                df = pd.DataFrame({"step": [e.step for e in evs], "val": [e.value for e in evs]})
                df["roll"] = df["val"].rolling(window=15, min_periods=3).mean()
                ax_rew.plot(df["step"] / 1000.0, df["val"], color=c, alpha=0.20, linewidth=0.8)
                ax_rew.plot(df["step"] / 1000.0, df["roll"], color=c, label=f"{sname}", linewidth=1.8)
                metrics_summary[sname]["ep_rew_mean_final"] = float(df["val"].iloc[-1])

    ax_q.set_title("Pérdida de los Críticos (Critic Loss / MSE)", fontsize=11, fontweight="bold")
    ax_q.set_xlabel("Pasos de Interacción (k)", fontsize=10)
    ax_q.set_ylabel("Pérdida MSE", fontsize=10)
    ax_q.set_ylim(-0.2, 12)
    ax_q.grid(True, linestyle="--", alpha=0.5)
    if has_data:
        ax_q.legend(fontsize=8.5, loc="upper right")

    ax_pi.set_title("Pérdida del Actor (Policy Loss)", fontsize=11, fontweight="bold")
    ax_pi.set_xlabel("Pasos de Interacción (k)", fontsize=10)
    ax_pi.set_ylabel("Pérdida", fontsize=10)
    ax_pi.grid(True, linestyle="--", alpha=0.5)
    if has_data:
        ax_pi.legend(fontsize=8.5, loc="upper right")

    ax_alpha.set_title("Temperatura de Entropía (ent_coef / α)", fontsize=11, fontweight="bold")
    ax_alpha.set_xlabel("Pasos de Interacción (k)", fontsize=10)
    ax_alpha.set_ylabel("Valor de α", fontsize=10)
    ax_alpha.grid(True, linestyle="--", alpha=0.5)
    if has_data:
        ax_alpha.legend(fontsize=8.5, loc="upper right")

    ax_rew.set_title("Retorno Acumulado por Episodio (ep_rew_mean)", fontsize=11, fontweight="bold")
    ax_rew.set_xlabel("Pasos de Interacción (k)", fontsize=10)
    ax_rew.set_ylabel("Retorno Medio", fontsize=10)
    ax_rew.grid(True, linestyle="--", alpha=0.5)
    if has_data:
        ax_rew.legend(fontsize=8.5, loc="lower right")

    fig.suptitle("Diagnósticos Internos de Entrenamiento — Soft Actor-Critic (SAC S1, 3 semillas)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    meta = {
        "figure_id": "fig_c_sac_training_diagnostics",
        "experiment": exp_s1_dir.name,
        "n_seeds": len(metrics_summary),
        "metrics_summary": metrics_summary,
        "note": "Diagnósticos extraídos de eventos TensorBoard de SAC. No comparables numéricamente con pérdidas PPO."
    }
    save_fig_with_metadata(fig, out_dir / "fig_c_sac_training_diagnostics.png", meta)

def plot_fig3_failure_breakdown(details_df: pd.DataFrame, out_dir: Path, step_filter: int = 655360,
                                experiments=None):
    """Figura 3: Partición exhaustiva de desenlaces en el checkpoint final.

    experiments : agentes a comparar (por defecto los históricos
                  ["X3", "S1"])."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    sub = details_df[(details_df["step"] == step_filter) & (details_df["seed"] != "analytic")]
    if sub.empty:
        # Usar el paso máximo disponible
        max_s = details_df[details_df["seed"] != "analytic"]["step"].max()
        sub = details_df[details_df["step"] == max_s]

    agents = experiments if experiments is not None else ["X3", "S1"]
    x = np.arange(len(agents))
    width = 0.55

    success_vals = []
    trunc_vals = []
    ball_vals = []
    osc_vals = []

    for agent in agents:
        ag_sub = sub[sub["experiment"] == agent]
        if not ag_sub.empty:
            success_vals.append(float(ag_sub["success_rate"].mean()) * 100.0)
            trunc_vals.append(float(ag_sub["truncation_rate"].mean()) * 100.0)
            ball_vals.append(float(ag_sub["ballistic_rate"].mean()) * 100.0)
            osc_vals.append(float(ag_sub["oscillatory_rate"].mean()) * 100.0)
        else:
            success_vals.append(0.0)
            trunc_vals.append(0.0)
            ball_vals.append(0.0)
            osc_vals.append(0.0)

    # Barras apiladas
    p1 = ax.bar(x, success_vals, width, label="Éxito Estricto (Hold 20)", color="#2ca02c")
    p2 = ax.bar(x, trunc_vals, width, bottom=success_vals, label="Truncamiento (Sin Hold)", color="#d62728", alpha=0.85)

    ax.set_title(f"Fase C — Distribución de Desenlaces en Checkpoint Final ({step_filter:,} pasos)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    if experiments is None:
        ax.set_xticklabels([f"PPO ({agents[0]})\n[Media n=3]", f"SAC ({agents[1]})\n[Media n=3]"], fontsize=11)
    else:
        ax.set_xticklabels([f"{a}\n[Media n={max(1, int(sub[sub['experiment'] == a]['seed'].nunique()))}]" for a in agents], fontsize=11)
    ax.set_ylabel("Porcentaje de Episodios (%) [Total = 100 %]", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    # Anotar valores numéricos
    for i in range(len(agents)):
        ax.text(x[i], success_vals[i] / 2.0, f"{success_vals[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=11)
        if trunc_vals[i] > 3.0:
            ax.text(x[i], success_vals[i] + trunc_vals[i] / 2.0, f"{trunc_vals[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    meta = {
        "figure_id": "fig_c_failure_breakdown_final_ckpt",
        "step": step_filter,
        "dataset": "validation_c_500.json",
        "n_seeds": max(1, int(sub["seed"].nunique())),
        "reconciliation": "Suma de categorías = 100.0%"
    }
    save_fig_with_metadata(fig, out_dir / "fig_c_failure_breakdown_final_ckpt.png", meta)

def plot_fig4_canonical_trajectories(step_df: pd.DataFrame, out_dir: Path):
    """Figura 4: Planos de fase y series temporales en los 4 episodios geométricos."""
    episodes = sorted(step_df["episode_id"].unique())
    n_eps = len(episodes)
    fig, axes = plt.subplots(n_eps, 2, figsize=(12, 3.2 * n_eps), dpi=300)

    for i, ep_id in enumerate(episodes):
        ax_phase = axes[i, 0] if n_eps > 1 else axes[0]
        ax_dist = axes[i, 1] if n_eps > 1 else axes[1]

        ep_sub = step_df[(step_df["episode_id"] == ep_id) & (step_df["eval_mode"] == "deterministic")]

        for agent, col in [("C0", AGENT_COLORS["C0"]), ("X3", AGENT_COLORS["X3"]), ("S1", AGENT_COLORS["S1"])]:
            ag_data = ep_sub[ep_sub["agent"] == agent].sort_values("step")
            if ag_data.empty:
                continue
            # Plano de fase: distancia vs derivada de distancia
            ax_phase.plot(ag_data["distance"], ag_data["distance_rate"], label=agent, color=col, linewidth=1.8)

            # Perfil temporal: distancia vs tiempo
            t = ag_data["step"] * 0.05
            ax_dist.plot(t, ag_data["distance"], label=agent, color=col, linewidth=1.8)

        # Marcar umbral de hold (r = 0.06 m)
        ax_phase.axvline(0.06, color="green", linestyle=":", label="r_tgt (0.06 m)")
        ax_dist.axhline(0.06, color="green", linestyle=":", label="r_tgt (0.06 m)")

        r_tgt = ep_sub["target_r"].iloc[0] if "target_r" in ep_sub else np.nan
        ax_phase.set_title(f"Plano de Fase (Ep {ep_id}, r = {r_tgt:.2f} m)", fontsize=10.5, fontweight="bold")
        ax_phase.set_xlabel("Distancia d (m)", fontsize=9.5)
        ax_phase.set_ylabel("Derivada dd/dt (m/s)", fontsize=9.5)
        ax_phase.grid(True, linestyle="--", alpha=0.5)

        ax_dist.set_title(f"Perfil Temporal de Distancia (Ep {ep_id})", fontsize=10.5, fontweight="bold")
        ax_dist.set_xlabel("Tiempo t (s)", fontsize=9.5)
        ax_dist.set_ylabel("Distancia d (m)", fontsize=9.5)
        ax_dist.grid(True, linestyle="--", alpha=0.5)

        if i == 0:
            ax_phase.legend(fontsize=8.5, loc="lower right")
            ax_dist.legend(fontsize=8.5, loc="upper right")

    plt.tight_layout()
    meta = {
        "figure_id": "fig_c_canonical_geometric_trajectories",
        "episodes": episodes,
        "mode": "deterministic",
        "dataset": "validation_c_500.json"
    }
    save_fig_with_metadata(fig, out_dir / "fig_c_canonical_geometric_trajectories.png", meta)

def plot_fig5_deterministic_vs_stochastic(step_df: pd.DataFrame, out_dir: Path):
    """Figura 5: Comparación determinista frente a estocástica de velocidad articular."""
    ep_id = step_df["episode_id"].iloc[0]
    ep_sub = step_df[step_df["episode_id"] == ep_id]

    fig, (ax_x3, ax_s1) = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)

    # Panel PPO X3
    x3_det = ep_sub[(ep_sub["agent"] == "X3") & (ep_sub["eval_mode"] == "deterministic")].sort_values("step")
    x3_sto = ep_sub[(ep_sub["agent"] == "X3") & (ep_sub["eval_mode"] == "stochastic")].sort_values("step")
    if not x3_det.empty:
        ax_x3.plot(x3_det["step"] * 0.05, x3_det["joint_speed"], label="Determinista", color=AGENT_COLORS["X3"], linewidth=2.0)
    if not x3_sto.empty:
        ax_x3.plot(x3_sto["step"] * 0.05, x3_sto["joint_speed"], label="Estocástico", color=AGENT_COLORS["X3_sto"], linewidth=1.5, linestyle="--")

    ax_x3.axhline(0.10, color="red", linestyle=":", linewidth=1.8, label="v_tol (0.10 rad/s)")
    ax_x3.set_title(f"PPO (X3) — Determinista vs Estocástico (Ep {ep_id})", fontsize=11, fontweight="bold")
    ax_x3.set_xlabel("Tiempo t (s)", fontsize=10)
    ax_x3.set_ylabel("Velocidad Articular ‖θ̇‖ (rad/s)", fontsize=10)
    ax_x3.set_ylim(-0.05, 1.6)
    ax_x3.grid(True, linestyle="--", alpha=0.5)
    ax_x3.legend(fontsize=9)

    # Panel SAC S1
    s1_det = ep_sub[(ep_sub["agent"] == "S1") & (ep_sub["eval_mode"] == "deterministic")].sort_values("step")
    s1_sto = ep_sub[(ep_sub["agent"] == "S1") & (ep_sub["eval_mode"] == "stochastic")].sort_values("step")
    if not s1_det.empty:
        ax_s1.plot(s1_det["step"] * 0.05, s1_det["joint_speed"], label="Determinista", color=AGENT_COLORS["S1"], linewidth=2.0)
    if not s1_sto.empty:
        ax_s1.plot(s1_sto["step"] * 0.05, s1_sto["joint_speed"], label="Estocástico", color=AGENT_COLORS["S1_sto"], linewidth=1.5, linestyle="--")

    ax_s1.axhline(0.10, color="red", linestyle=":", linewidth=1.8, label="v_tol (0.10 rad/s)")
    ax_s1.set_title(f"SAC (S1) — Determinista vs Estocástico (Ep {ep_id})", fontsize=11, fontweight="bold")
    ax_s1.set_xlabel("Tiempo t (s)", fontsize=10)
    ax_s1.grid(True, linestyle="--", alpha=0.5)
    ax_s1.legend(fontsize=9)

    fig.suptitle("Respuesta Dinámica bajo Inyección de Ruido Estocástico vs Tolerancia de Hold", fontsize=12.5, fontweight="bold")
    plt.tight_layout()

    meta = {
        "figure_id": "fig_c_deterministic_vs_stochastic_profiles",
        "episode_id": int(ep_id),
        "note": "Contraste entre política media determinista y muestreo estocástico"
    }
    save_fig_with_metadata(fig, out_dir / "fig_c_deterministic_vs_stochastic_profiles.png", meta)

def plot_fig6_compute_and_runtime(out_dir: Path, time_s1_sec: float = 4600.0, time_x3_sec: float = 720.0):
    """Figura 6: Diagrama comparativo de tiempo de reloj y pasos de gradiente."""
    fig, (ax_time, ax_grad) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    # Panel 1: Tiempo de reloj en minutos
    times_min = [time_x3_sec / 60.0, time_s1_sec / 60.0]
    labels = ["PPO (X3)", "SAC (S1)"]
    colors = [AGENT_COLORS["X3"], AGENT_COLORS["S1"]]

    bars1 = ax_time.bar(labels, times_min, color=colors, width=0.55)
    ax_time.set_title("Tiempo de Reloj por Corrida (655k pasos)", fontsize=11.5, fontweight="bold")
    ax_time.set_ylabel("Tiempo de Reloj (minutos)", fontsize=10.5)
    ax_time.grid(axis="y", linestyle="--", alpha=0.6)
    for b in bars1:
        h = b.get_height()
        ax_time.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"{h:.1f} min", ha="center", va="bottom", fontweight="bold")

    # Panel 2: Pasos de gradiente
    grad_steps = [2560, 650360]
    bars2 = ax_grad.bar(labels, [g / 1000.0 for g in grad_steps], color=colors, width=0.55)
    ax_grad.set_title("Actualizaciones de Gradiente Totales", fontsize=11.5, fontweight="bold")
    ax_grad.set_ylabel("Pasos de Gradiente (miles, k)", fontsize=10.5)
    ax_grad.grid(axis="y", linestyle="--", alpha=0.6)
    for b in bars2:
        h = b.get_height()
        ax_grad.text(b.get_x() + b.get_width() / 2.0, h + 10.0, f"{h:.1f}k", ha="center", va="bottom", fontweight="bold")

    fig.suptitle("Asimetría de Cómputo: SAC (Off-Policy 1:1) vs PPO (On-Policy)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    meta = {
        "figure_id": "fig_c_compute_cost_and_runtime_comparison",
        "time_s1_sec": time_s1_sec,
        "time_x3_sec": time_x3_sec,
        "grad_steps_s1": 650360,
        "grad_steps_x3": 2560
    }
    save_fig_with_metadata(fig, out_dir / "fig_c_compute_cost_and_runtime_comparison.png", meta)

def generate_all_phase_c_figures(data_dir=None,
                                 figures_dir=None,
                                 experiments=None,
                                 sac_exp_dir=None,
                                 figures=None):
    """
    Genera las figuras de Fase C; ``figures`` permite un subconjunto
    (lista de enteros 1..6). ``experiments`` filtra los agentes de las
    figuras 1 y 3; ``sac_exp_dir`` apunta la corrida SAC de la figura 2
    (por defecto experiments/phase_c/S1). Sin parámetros reproduce el
    estudio histórico íntegro.
    """
    in_path = Path(data_dir) if data_dir is not None else REPO_ROOT / "reports" / "data" / "phase_c"
    out_path = Path(figures_dir) if figures_dir is not None else REPO_ROOT / "reports" / "figures" / "phase_c"
    out_path.mkdir(parents=True, exist_ok=True)

    agg_csv = in_path / "checkpoints_eval_validation_aggregated.csv"
    det_csv = in_path / "checkpoints_eval_validation_details.csv"
    step_csv = in_path / "step_data_val_geometry.csv"
    want = set(figures) if figures is not None else {1, 2, 3, 4, 5, 6}

    print(f"\n{'='*75}")
    print(" GENERADOR DE FIGURAS — FASE C (300 DPI + JSON)")
    print(f"{'='*75}\n")

    if (agg_csv.exists() and det_csv.exists()) and (1 in want or 3 in want):
        agg_df = pd.read_csv(agg_csv)
        det_df = pd.read_csv(det_csv)
        if 1 in want:
            plot_fig1_learning_curves(agg_df, det_df, out_path, experiments=experiments)
        if 3 in want:
            plot_fig3_failure_breakdown(det_df, out_path, experiments=experiments)

    exp_s1_dir = Path(sac_exp_dir) if sac_exp_dir is not None else REPO_ROOT / "experiments" / "phase_c" / "S1"
    if 2 in want and exp_s1_dir.exists():
        plot_fig2_sac_training_diagnostics(exp_s1_dir, out_path)

    if step_csv.exists() and (4 in want or 5 in want):
        step_df = pd.read_csv(step_csv)
        if 4 in want:
            plot_fig4_canonical_trajectories(step_df, out_path)
        if 5 in want:
            plot_fig5_deterministic_vs_stochastic(step_df, out_path)

    # Figura 6 se puede generar siempre con los datos de cómputo medidos
    if 6 in want:
        plot_fig6_compute_and_runtime(out_path)

    print(f"\n>>> [FIGURAS] Proceso completado en: {out_path}\n")

def run(out_dir=None, data_dir=None, experiments=None, sac_exp_dir=None, figures=None):
    """
    Punto de entrada del módulo.

    out_dir     : ``None`` (por defecto reports/figures/phase_c) o ruta de
                  salida alternativa.
    data_dir    : ruta alternativa con los CSV de evaluación (por defecto
                  reports/data/phase_c; mismo esquema histórico de columnas).
    experiments : agentes a dibujar en las figuras 1 y 3 (por defecto
                  ["X3", "S1"]).
    sac_exp_dir : corrida SAC de la figura 2 (por defecto
                  experiments/phase_c/S1; p. ej. un modelo nuevo indica
                  experiments/<fase>/<exp>).
    figures     : subconjunto de figuras (lista de enteros; ``None`` = 1-6).
    """
    generate_all_phase_c_figures(
        data_dir=data_dir,
        figures_dir=str(out_dir) if out_dir is not None else None,
        experiments=experiments,
        sac_exp_dir=Path(sac_exp_dir) if sac_exp_dir is not None else None,
        figures=figures)


if __name__ == "__main__":
    run()
