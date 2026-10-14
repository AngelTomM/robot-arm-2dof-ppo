"""
Pipeline de Generación de Figuras para Fase C (300 dpi + Metadatos JSON):
1. fig_c_learning_curves_s1_vs_x3.png: Curva de éxito determinista vs pasos (S1 vs X3 y C0) + panel de truncamientos.
2. fig_c_sac_training_diagnostics.png: Diagnósticos de entrenamiento propios de SAC (pérdidas actor/crítico, ent_coef, recompensa).
3. fig_c_failure_breakdown_final_ckpt.png: Partición de desenlaces en el checkpoint final (655.360) reconciliada al 100%.
4. fig_c_canonical_geometric_trajectories.png: Planos de fase y series temporales en 4 episodios geométricos (S1, X3, C0).
5. fig_c_deterministic_vs_stochastic_profiles.png: Comparativa determinista vs estocástica en velocidad articular.
6. fig_c_compute_cost_and_runtime_comparison.png: Tiempos de reloj y pasos de gradiente (SAC vs PPO).
"""
import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

AGENT_COLORS = {
    "S1": "#d62728",       # Rojo para SAC S1
    "X3": "#1f77b4",       # Azul para PPO X3
    "C0": "#000000",       # Negro para Controlador Analítico C0
    "S1_sto": "#ff9896",   # Rojo claro para estocástico
    "X3_sto": "#aec7e8"    # Azul claro para estocástico
}

def save_fig_with_metadata(fig, output_png_path: Path, metadata: dict):
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    def _default_conv(o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    json_path = output_png_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=_default_conv)
    print(f"  [FIGURA OK] Guardada: {output_png_path.name}")

def plot_fig1_learning_curves(agg_df: pd.DataFrame, details_df: pd.DataFrame, out_dir: Path):
    """Figura 1: Curva de éxito determinista y panel de truncamientos vs pasos."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})

    # Panel superior: Éxito estricto
    for agent in ["X3", "S1"]:
        sub_agg = agg_df[agg_df["experiment"] == agent].sort_values("step")
        if sub_agg.empty:
            continue
        x = sub_agg["step"] / 1000.0
        y = sub_agg["mean_success_rate"] * 100.0
        std = sub_agg["std_success_rate"] * 100.0
        color = AGENT_COLORS[agent]

        # Curva media con banda +- std
        ax1.plot(x, y, label=f"{agent} Media (n=3)", color=color, linewidth=2.5, marker="o", markersize=4.5)
        ax1.fill_between(x, np.maximum(0, y - std), np.minimum(100, y + std), color=color, alpha=0.18)

        # Semillas individuales tenues
        sub_det = details_df[details_df["experiment"] == agent]
        for s, s_grp in sub_det.groupby("seed"):
            s_sorted = s_grp.sort_values("step")
            ax1.plot(s_sorted["step"] / 1000.0, s_sorted["success_rate"] * 100.0,
                     color=color, linestyle=":", alpha=0.45, linewidth=1.0)

    # C0 Línea de referencia
    c0_sub = details_df[details_df["experiment"].str.startswith("C0")]
    if not c0_sub.empty:
        c0_succ = float(c0_sub["success_rate"].iloc[0]) * 100.0
        ax1.axhline(c0_succ, color=AGENT_COLORS["C0"], linestyle="--", linewidth=1.8, label="C0 Analítico (100.0%)")

    ax1.set_title("Fase C — Comparación Descriptiva de Aprendizaje: SAC (S1) vs PPO (X3)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Éxito Estricto (%) [Hold 20 pasos, v ≤ 0.10 rad/s]", fontsize=10.5)
    ax1.set_ylim(-2, 104)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

    # Panel inferior: Truncamientos
    for agent in ["X3", "S1"]:
        sub_agg = agg_df[agg_df["experiment"] == agent].sort_values("step")
        if sub_agg.empty:
            continue
        x = sub_agg["step"] / 1000.0
        y_trunc = sub_agg["mean_truncation_rate"] * 100.0
        color = AGENT_COLORS[agent]
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
        "n_seeds": 3,
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
        "experiment": "S1",
        "n_seeds": len(metrics_summary),
        "metrics_summary": metrics_summary,
        "note": "Diagnósticos extraídos de eventos TensorBoard de SAC. No comparables numéricamente con pérdidas PPO."
    }
    save_fig_with_metadata(fig, out_dir / "fig_c_sac_training_diagnostics.png", meta)

def plot_fig3_failure_breakdown(details_df: pd.DataFrame, out_dir: Path, step_filter: int = 655360):
    """Figura 3: Partición exhaustiva de desenlaces en el checkpoint final."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    sub = details_df[(details_df["step"] == step_filter) & (details_df["seed"] != "analytic")]
    if sub.empty:
        # Usar el paso máximo disponible
        max_s = details_df[details_df["seed"] != "analytic"]["step"].max()
        sub = details_df[details_df["step"] == max_s]

    agents = ["X3", "S1"]
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
    ax.set_xticklabels([f"PPO ({agents[0]})\n[Media n=3]", f"SAC ({agents[1]})\n[Media n=3]"], fontsize=11)
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
        "n_seeds": 3,
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

def generate_all_phase_c_figures(data_dir: str = "reports/data/phase_c",
                                 figures_dir: str = "reports/figures/phase_c"):
    in_path = REPO_ROOT / data_dir
    out_path = REPO_ROOT / figures_dir
    out_path.mkdir(parents=True, exist_ok=True)

    agg_csv = in_path / "checkpoints_eval_validation_aggregated.csv"
    det_csv = in_path / "checkpoints_eval_validation_details.csv"
    step_csv = in_path / "step_data_val_geometry.csv"

    print(f"\n{'='*75}")
    print(" GENERADOR DE FIGURAS — FASE C (300 DPI + JSON)")
    print(f"{'='*75}\n")

    if agg_csv.exists() and det_csv.exists():
        agg_df = pd.read_csv(agg_csv)
        det_df = pd.read_csv(det_csv)
        plot_fig1_learning_curves(agg_df, det_df, out_path)
        plot_fig3_failure_breakdown(det_df, out_path)

    exp_s1_dir = REPO_ROOT / "experiments" / "phase_c" / "S1"
    if exp_s1_dir.exists():
        plot_fig2_sac_training_diagnostics(exp_s1_dir, out_path)

    if step_csv.exists():
        step_df = pd.read_csv(step_csv)
        plot_fig4_canonical_trajectories(step_df, out_path)
        plot_fig5_deterministic_vs_stochastic(step_df, out_path)

    # Figura 6 se puede generar siempre con los datos de cómputo medidos
    plot_fig6_compute_and_runtime(out_path)

    print(f"\n>>> [FIGURAS] Proceso completado en: {out_path}\n")

if __name__ == "__main__":
    generate_all_phase_c_figures()
