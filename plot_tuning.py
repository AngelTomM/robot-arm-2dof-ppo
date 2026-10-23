"""
Script de Generación de Figuras de Calibración Paramétrica con Matplotlib (Fase D).

Carga los datos estructurados de reports/data/evaluation/tuning_c0.json y genera:
  1. reports/figures/pareto_front_tuning.png:
     Frente de Pareto (e_ss vs. J_u) según variaciones de rigidez/amortiguamiento
     y análisis de ponderación del funcional LQR (w_ctrl, w_vel).
  2. reports/figures/temporal_convergence_profiles.png:
     Perfiles temporales continuos de convergencia: distancia cartesiana d(t),
     velocidad articular ||qvel(t)|| y esfuerzo de torque normalizado tau(t),
     comparando configuraciones subóptimas (subamortiguada y sobreamortiguada)
     frente a la configuración calibrada óptima.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO_ROOT = Path(__file__).resolve().parent
DATA_JSON = REPO_ROOT / "reports/data/evaluation/tuning_c0.json"
FIGS_DIR = REPO_ROOT / "reports/figures"


def apply_publication_style():
    """Aplica estilos tipográficos y estéticos formales para informes mecatrónicos."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 1.75,
        "figure.autolayout": False,
    })


def plot_pareto_front(tuning_data: dict, output_file: Path):
    """
    Genera la figura del Frente de Pareto e_ss vs. J_u y análisis de sensibilidad LQR.
    """
    pareto_pts = tuning_data.get("pareto_front", [])
    full_sweep = tuning_data.get("full_sweep", [])
    reward_sweep = tuning_data.get("reward_sweep", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # -------------------------------------------------------------
    # Panel Izquierdo: Espacio de Desempeño y Frente de Pareto
    # -------------------------------------------------------------
    all_ess = [p["e_ss_mean"] * 1000.0 for p in full_sweep]  # mm
    all_ju = [p["j_u_mean"] for p in full_sweep]
    kd_vals = [p["kd"] for p in full_sweep]

    sc = ax1.scatter(
        all_ju, all_ess, c=kd_vals, cmap="viridis",
        alpha=0.65, s=45, edgecolors="none", label="Configuraciones exploradas"
    )
    cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Amortiguamiento $K_d$ (s)", fontsize=10)

    # Trazar curva de Pareto
    pareto_ju = [p["j_u_mean"] for p in pareto_pts]
    pareto_ess = [p["e_ss_mean"] * 1000.0 for p in pareto_pts]
    # Ordenar por Ju para trazo limpio
    sorted_pairs = sorted(zip(pareto_ju, pareto_ess), key=lambda x: x[0])
    p_ju_sorted, p_ess_sorted = zip(*sorted_pairs)

    ax1.plot(
        p_ju_sorted, p_ess_sorted,
        color="#c0392b", linestyle="-", linewidth=2.2,
        marker="o", markersize=6, label="Frente de Pareto (No dominados)"
    )

    # Destacar configuraciones clave
    # 1. Calibrada Óptima
    opt = next((p for p in full_sweep if p["kp"] == 50.0 and p["kd"] == 0.05), full_sweep[0])
    ax1.scatter(
        [opt["j_u_mean"]], [opt["e_ss_mean"] * 1000.0],
        color="#27ae60", s=140, marker="*", edgecolors="black", linewidth=1.2,
        zorder=10, label=f"Calibrada Óptima ($K_p$=50, $K_d$=0.05)"
    )
    ax1.annotate(
        f"Óptimo Calibrado\n$e_{{ss}}$ = {opt['e_ss_mean']*1000:.1f} mm\n$J_u$ = {opt['j_u_mean']:.3f}",
        xy=(opt["j_u_mean"], opt["e_ss_mean"] * 1000.0),
        xytext=(opt["j_u_mean"] + 0.04, opt["e_ss_mean"] * 1000.0 + 8.0),
        arrowprops=dict(facecolor="#27ae60", shrink=0.08, width=1.0, headwidth=6),
        fontsize=8.5, fontweight="bold", color="#1e8449",
        bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#27ae60", alpha=0.9)
    )

    # 2. Subóptima Sobreamortiguada
    over = next((p for p in full_sweep if p["kp"] == 20.0 and p["kd"] == 0.18), full_sweep[-1])
    ax1.scatter(
        [over["j_u_mean"]], [over["e_ss_mean"] * 1000.0],
        color="#d35400", s=100, marker="s", edgecolors="black", linewidth=1.2,
        zorder=10, label="Sobreamortiguada ($K_p$=20, $K_d$=0.18)"
    )

    # Línea de umbral de éxito cartesiano
    ax1.axhline(20.0, color="#7f8c8d", linestyle=":", linewidth=1.4, label="Umbral $r_{target} = 20$ mm")

    ax1.set_xlabel("Esfuerzo de control acumulado $J_u$ [$\\mathrm{adimensional}$]")
    ax1.set_ylabel("Error cartesiano estacionario $e_{ss}$ [$\\mathrm{mm}$]")
    ax1.set_title("(a) Frente de Pareto y Espacio de Desempeño Dinámico", pad=10)
    ax1.set_ylim(-2, 60)
    ax1.set_xlim(-0.02, max(all_ju) * 1.05)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # -------------------------------------------------------------
    # Panel Derecho: Sensibilidad del Funcional LQR a w_ctrl y w_vel
    # -------------------------------------------------------------
    if reward_sweep:
        w_ctrls = sorted(list(set(r["w_ctrl"] for r in reward_sweep)))
        w_vels = sorted(list(set(r["w_vel"] for r in reward_sweep)))

        # Graficar curvas de costo LQR vs w_ctrl para diferentes w_vel
        palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(w_vels)))
        for i, wv in enumerate(w_vels):
            pts = [r for r in reward_sweep if r["w_vel"] == wv]
            pts.sort(key=lambda x: x["w_ctrl"])
            x_vals = [p["w_ctrl"] for p in pts]
            y_vals = [p["j_lqr_mean"] for p in pts]
            ax2.plot(
                x_vals, y_vals, marker="o", markersize=4,
                color=palette[i], label=f"$w_{{vel}} = {wv}$"
            )

        # Destacar el punto calibrado de recompensa (w_ctrl=0.008, w_vel=0.005)
        ax2.axvline(0.008, color="#27ae60", linestyle="--", linewidth=1.5, label="Calibrado $w_{ctrl} = 0.008$")

        ax2.set_xscale("log")
        ax2.set_xlabel("Ponderador de esfuerzo de control $w_{ctrl}$")
        ax2.set_ylabel("Costo acumulado del funcional cuadrático $J_{LQR}$")
        ax2.set_title("(b) Sensibilidad del Funcional LQR ante $w_{ctrl}$ y $w_{vel}$", pad=10)
        ax2.legend(loc="upper left", framealpha=0.88, fontsize=8)
    else:
        ax2.text(0.5, 0.5, "Datos de barrido de recompensa no disponibles", ha="center", va="center")

    fig.tight_layout(pad=2.0)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f">>> [FIGURA 1] Guardada exitosamente: {output_file}")


def plot_temporal_convergence_profiles(tuning_data: dict, output_file: Path):
    """
    Genera la figura de 3 subplots con perfiles temporales de convergencia:
      - Distancia cartesiana d(t) efector-objetivo
      - Norma de velocidad articular ||qvel(t)||
      - Señales de torque articular tau_1(t) y tau_2(t)
    """
    rep = tuning_data.get("representative_trajectories", {})
    trajs = rep.get("trajectories", {})

    under = trajs.get("suboptimal_underdamped")
    over = trajs.get("suboptimal_overdamped")
    opt = trajs.get("calibrated_optimal")

    if not (under and over and opt):
        print(">>> [ADVERTENCIA] No se encontraron todas las trayectorias temporales necesarias.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True, dpi=300)

    t = np.array(opt["time"])

    # Colores normalizados para coherencia gráfica
    c_under = "#c0392b"  # Rojo: Oscilatorio / Subamortiguado
    c_over = "#d35400"   # Naranja: Sobreamortiguado / Lento
    c_opt = "#27ae60"    # Verde: Calibrado Óptimo

    # -------------------------------------------------------------
    # 1. Distancia Cartesiana d(t) [mm]
    # -------------------------------------------------------------
    ax1.plot(t, np.array(under["distance"]) * 1000.0, color=c_under, linestyle="--", label="Subóptima Subamortiguada ($K_d$=0.005)")
    ax1.plot(t, np.array(over["distance"]) * 1000.0, color=c_over, linestyle="-.", label="Subóptima Sobreamortiguada ($K_d$=0.180)")
    ax1.plot(t, np.array(opt["distance"]) * 1000.0, color=c_opt, linestyle="-", linewidth=2.2, label="Calibrada Óptima ($K_p$=50, $K_d$=0.05)")

    # Umbral de éxito externo
    ax1.axhline(20.0, color="#2c3e50", linestyle=":", linewidth=1.5, label="Umbral de contacto $r_{target} = 20$ mm")
    # Banda estacionaria final (últimos 0.4 s / 20 pasos)
    ax1.axvspan(1.6, 2.0, color="#bdc3c7", alpha=0.25, label="Ventana estacionaria final ($k=20$)")

    ax1.set_ylabel("Distancia $d(t)$ [$\\mathrm{mm}$]")
    ax1.set_title("Respuesta Dinámica Comparativa del Manipulador Planar 2-DOF (MuJoCo Reacher-v5)", pad=12, fontsize=12)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax1.set_ylim(-2, 180)

    # -------------------------------------------------------------
    # 2. Norma de Velocidad Articular ||qvel(t)|| [rad/s]
    # -------------------------------------------------------------
    ax2.plot(t, under["qvel_norm"], color=c_under, linestyle="--", label="Subamortiguada (oscilaciones de velocidad)")
    ax2.plot(t, over["qvel_norm"], color=c_over, linestyle="-.", label="Sobreamortiguada (transitorio estrangulado)")
    ax2.plot(t, opt["qvel_norm"], color=c_opt, linestyle="-", linewidth=2.2, label="Calibrada Óptima (disipación progresiva)")

    ax2.axhline(0.10, color="#2c3e50", linestyle=":", linewidth=1.5, label="Tolerancia de velocidad $v_{tol} = 0.10$ rad/s")
    ax2.axvspan(1.6, 2.0, color="#bdc3c7", alpha=0.25)

    ax2.set_ylabel("Velocidad $\\|q_{vel}(t)\\|$ [$\\mathrm{rad/s}$]")
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax2.set_ylim(-0.2, 10.0)

    # -------------------------------------------------------------
    # 3. Acciones de Torque tau_1 y tau_2 [-1.0, 1.0]
    # -------------------------------------------------------------
    # Se grafican tau_1 (articulación proximal) y tau_2 (articulación distal)
    ax3.plot(t, under["tau_1"], color=c_under, linestyle="--", alpha=0.75, label="Subamortiguada $\\tau_1$ (chattering/saturación)")
    ax3.plot(t, over["tau_1"], color=c_over, linestyle="-.", alpha=0.75, label="Sobreamortiguada $\\tau_1$ (esfuerzo de frenado)")
    ax3.plot(t, opt["tau_1"], color=c_opt, linestyle="-", linewidth=2.0, label="Calibrada Óptima $\\tau_1$ (esfuerzo suave)")
    ax3.plot(t, opt["tau_2"], color="#16a085", linestyle=":", linewidth=1.8, label="Calibrada Óptima $\\tau_2$")

    ax3.axhline(1.0, color="#7f8c8d", linestyle="-", linewidth=0.8, alpha=0.6)
    ax3.axhline(-1.0, color="#7f8c8d", linestyle="-", linewidth=0.8, alpha=0.6)
    ax3.axvspan(1.6, 2.0, color="#bdc3c7", alpha=0.25)

    ax3.set_xlabel("Tiempo físico $t$ [$\\mathrm{s}$]")
    ax3.set_ylabel("Torque normalizado $\\tau$ [$\\mathrm{adimensional}$]")
    ax3.set_xlim(0.0, 2.0)
    ax3.set_ylim(-1.15, 1.15)
    ax3.legend(loc="lower right", framealpha=0.9, fontsize=8.0, ncol=2)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f">>> [FIGURA 2] Guardada exitosamente: {output_file}")


def main():
    apply_publication_style()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_JSON.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos de calibración: {DATA_JSON}.\n"
            "Ejecute previamente 'python tune_c0.py' para generar los resultados."
        )

    with open(DATA_JSON, "r", encoding="utf-8") as f:
        tuning_data = json.load(f)

    fig_pareto = FIGS_DIR / "pareto_front_tuning.png"
    fig_temporal = FIGS_DIR / "temporal_convergence_profiles.png"

    plot_pareto_front(tuning_data, fig_pareto)
    plot_temporal_convergence_profiles(tuning_data, fig_temporal)


if __name__ == "__main__":
    main()
