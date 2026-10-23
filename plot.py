"""
Generador Modular de Figuras y Gráficas de Análisis Mecatrónico (Fase D).

Organiza los entregables gráficos de forma desacoplada, estructurada e independiente:
  1. Gráficas por modelo en carpetas exclusivas: reports/figures/<modelo>/
     - success_rate.png: Tasa de estabilización terminal con IC Wilson al 95%.
     - steady_state_error.png: Distribución de error cartesiano en régimen permanente (e_ss).
     - residual_velocity.png: Distribución de velocidad angular residual (v_ss).
     - settling_time.png: Distribución de tiempo de establecimiento (t_s).
     - control_effort.png: Distribución de esfuerzo de torque acumulado (J_u).
     - reward_components.png: Curvas de las 3 componentes del costo LQR (J_pos, J_vel, J_ctrl).
     - temporal_trajectory_episode.png: Respuesta temporal de estado y torque en episodio crítico.
     - pareto_tradeoff.png: Frente de Pareto y dispersión bi-objetivo (e_ss vs J_u).
     - workspace_spatial_error.png: Mapeo espacial del error cartesiano en el espacio de trabajo.
     - learning_curve_return.png: Retorno episódico medio vs timesteps (ajuste multisemilla).
     - learning_curve_success_rate.png: Tasa de éxito vs timesteps (ajuste multisemilla).
     - multiseed_learning_curves.png: Comparativa consolidada de semillas y dispersión estadística.
     - optimizer_total_loss.png: Pérdida total del optimizador.
     - optimizer_policy_loss.png: Pérdida del gradiente de la política.
     - optimizer_value_loss.png: Pérdida cuadrática de la función de valor (crítico).
     - optimizer_entropy.png: Evolución de la entropía de la política.
     - optimizer_approx_kl.png: Divergencia KL aproximada por época.
     - optimizer_explained_variance.png: Varianza explicada de la función de valor.
     - optimizer_convergence_summary.png: Panel resumen 2x3 con todas las pérdidas del optimizador.
  2. Gráficas comparativas consolidadas en:
     - reports/figures/comparative/ (por defecto para todos los modelos)
     - reports/figures/<M1>_<M2>_.../ (para combinaciones específicas con --compare M1,M2,...)
     - comparative_success_rate.png
     - comparative_steady_state_error.png
     - comparative_residual_velocity.png
     - comparative_settling_time.png
     - comparative_control_effort.png
     - comparative_reward_components.png
     - comparative_pareto_tradeoff.png
     - comparative_spatial_error.png
     - comparative_temporal_trajectories.png
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent


def apply_plot_style() -> None:
    """Configura parámetros tipográficos y estéticos formales para informes mecatrónicos."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
        "figure.autolayout": False,
    })


def load_evaluation_data(data_dir: Path, files: Optional[List[str]] = None) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Carga los resultados tabulares buscando tanto en subdirectorios
    (reports/data/evaluation/<modelo>/episodes.csv) como en archivos planos (*_episodes.csv).
    Retorna un diccionario {nombre_modelo: df} y un DataFrame concatenado.
    """
    dfs_by_model: Dict[str, pd.DataFrame] = {}

    if files:
        csv_paths = [Path(f) for f in files]
    else:
        subfolder_csvs = list(data_dir.glob("*/episodes.csv"))
        flat_csvs = list(data_dir.glob("*_episodes.csv"))
        csv_paths = sorted(list(set(subfolder_csvs + flat_csvs)))

    if not csv_paths:
        raise FileNotFoundError(f"No se encontraron archivos de evaluación en {data_dir}")

    # Cargar coordenadas del dataset de referencia si se requieren para completar target_x, target_y
    ref_dataset_path = REPO_ROOT / "data/datasets/validation_set_200.json"
    ref_coords = {}
    if ref_dataset_path.exists():
        try:
            with open(ref_dataset_path, "r", encoding="utf-8") as f:
                ds = json.load(f)
                for item in ds:
                    ref_coords[int(item["episode_id"])] = (float(item["target_x"]), float(item["target_y"]))
        except Exception:
            pass

    for p in csv_paths:
        if not p.exists() or p.is_dir():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue

        if "agent" in df.columns and len(df["agent"].dropna().unique()) > 0:
            model_name = str(df["agent"].iloc[0])
        elif p.name == "episodes.csv":
            model_name = p.parent.name
            df["agent"] = model_name
        else:
            model_name = p.stem.replace("_episodes", "")
            df["agent"] = model_name

        # Asegurar coordenadas target_x, target_y si faltasen
        if "target_x" not in df.columns and ref_coords:
            df["target_x"] = df["episode_id"].map(lambda ep: ref_coords.get(int(ep), (0.0, 0.0))[0])
            df["target_y"] = df["episode_id"].map(lambda ep: ref_coords.get(int(ep), (0.0, 0.0))[1])

        if model_name not in dfs_by_model:
            dfs_by_model[model_name] = df

    if not dfs_by_model:
        raise ValueError("No se pudieron cargar datos válidos de evaluación.")

    all_concat = pd.concat(list(dfs_by_model.values()), ignore_index=True)
    return dfs_by_model, all_concat


def load_sample_trajectories(data_dir: Path, model_name: str) -> Optional[pd.DataFrame]:
    """Carga sample_trajectories.csv si existe para el modelo especificado."""
    traj_path = data_dir / model_name / "sample_trajectories.csv"
    if traj_path.exists():
        try:
            return pd.read_csv(traj_path)
        except Exception:
            return None
    return None


def extract_tensorboard_scalars(log_dirs: List[Path], tags: List[str]) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Extrae series temporales de escalares (steps, values) desde archivos de eventos de TensorBoard.
    Retorna un diccionario {tag: [(steps_seed1, values_seed1), (steps_seed2, values_seed2), ...]}.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    results: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {t: [] for t in tags}

    for ld in log_dirs:
        event_files = sorted(list(ld.glob("**/events.out.tfevents*")))
        if not event_files:
            continue
        for ef in event_files:
            try:
                ea = EventAccumulator(str(ef))
                ea.Reload()
                available_scalars = ea.Tags().get("scalars", [])
                for t in tags:
                    if t in available_scalars:
                        events = ea.Scalars(t)
                        steps = np.array([e.step for e in events])
                        vals = np.array([e.value for e in events])
                        if len(steps) > 0:
                            results[t].append((steps, vals))
            except Exception:
                continue

    return {k: v for k, v in results.items() if len(v) > 0}


# ==============================================================================
# 1. GENERACIÓN DE FIGURAS INDEPENDIENTES POR MODELO
# ==============================================================================

def plot_model_success_rate(df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    """Genera gráfica independiente de la tasa de éxito con IC Wilson al 95%."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)

    succ = int(df["success"].sum()) if "success" in df else 0
    total = len(df)
    p_hat = succ / total if total > 0 else 0.0

    z = 1.95996
    z2 = z ** 2
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    spread = z * np.sqrt((p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * (total ** 2)))) / denom
    low = max(0.0, center - spread)
    high = min(1.0, center + spread)

    err_low = (p_hat - low) * 100
    err_high = (high - p_hat) * 100

    ax.bar([0], [p_hat * 100], yerr=[[err_low], [err_high]], capsize=6,
           color="#2b5c8f", edgecolor="#1a365d", width=0.35, alpha=0.88)

    ax.text(0, p_hat * 100 + err_high + 2.5, f"{p_hat * 100:.1f}%\n({succ}/{total})",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks([0])
    ax.set_xticklabels([model_name], fontweight="bold", fontsize=11)
    ax.set_ylabel("Tasa de Estabilización (%) [IC Wilson 95%]")
    ax.set_title(f"Tasa de Éxito de Estabilización — {model_name}", pad=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {out_path.name}")


def plot_model_steady_state_error(df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    """Genera gráfica independiente de error cartesiano e_ss."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)

    e_ss = df["e_ss"].values * 1000.0  # en mm
    bp = ax.boxplot([e_ss], patch_artist=True, showfliers=False, widths=0.35)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.75)

    ax.axhline(20.0, color="#c0392b", linestyle="--", linewidth=1.4, label="Tolerancia r_target = 20 mm")

    mean_val = float(np.mean(e_ss))
    median_val = float(np.median(e_ss))
    ax.text(0.98, 0.95, f"Media: {mean_val:.2f} mm\nMediana: {median_val:.2f} mm",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f4f6f7", ec="#bdc3c7"))

    ax.set_xticks([1])
    ax.set_xticklabels([model_name], fontweight="bold", fontsize=11)
    ax.set_ylabel("Error Cartesiano Estacionario $e_{ss}$ [mm]")
    ax.set_title(f"Precisión Cartesiana Terminal — {model_name}", pad=10)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {out_path.name}")


def plot_model_residual_velocity(df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    """Genera gráfica independiente de velocidad residual v_ss."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)

    v_ss = df["v_ss"].values
    bp = ax.boxplot([v_ss], patch_artist=True, showfliers=False, widths=0.35)
    for patch in bp["boxes"]:
        patch.set_facecolor("#1abc9c")
        patch.set_alpha(0.75)

    ax.axhline(0.10, color="#c0392b", linestyle="--", linewidth=1.4, label="Tolerancia v_tol = 0.10 rad/s")

    mean_val = float(np.mean(v_ss))
    median_val = float(np.median(v_ss))
    ax.text(0.98, 0.95, f"Media: {mean_val:.4f} rad/s\nMediana: {median_val:.4f} rad/s",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f4f6f7", ec="#bdc3c7"))

    ax.set_xticks([1])
    ax.set_xticklabels([model_name], fontweight="bold", fontsize=11)
    ax.set_ylabel("Velocidad Residual $v_{ss}$ [rad/s]")
    ax.set_title(f"Disipación Cinética Terminal — {model_name}", pad=10)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {out_path.name}")


def plot_model_settling_time(df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    """Genera gráfica independiente de tiempo de establecimiento t_s."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)

    ts = df["t_s"].values
    bp = ax.boxplot([ts], patch_artist=True, showfliers=False, widths=0.35)
    for patch in bp["boxes"]:
        patch.set_facecolor("#f39c12")
        patch.set_alpha(0.75)

    mean_val = float(np.mean(ts))
    median_val = float(np.median(ts))
    ax.text(0.98, 0.95, f"Media: {mean_val:.3f} s\nMediana: {median_val:.3f} s",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f4f6f7", ec="#bdc3c7"))

    ax.set_xticks([1])
    ax.set_xticklabels([model_name], fontweight="bold", fontsize=11)
    ax.set_ylabel("Tiempo de Establecimiento $t_s$ [s]")
    ax.set_title(f"Rapidez de Convergencia Dinámica — {model_name}", pad=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {out_path.name}")


def plot_model_control_effort(df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    """Genera gráfica independiente de esfuerzo de control acumulado J_u."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)

    ju = df["j_u"].values
    bp = ax.boxplot([ju], patch_artist=True, showfliers=False, widths=0.35)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9b59b6")
        patch.set_alpha(0.75)

    mean_val = float(np.mean(ju))
    median_val = float(np.median(ju))
    ax.text(0.98, 0.95, f"Media: {mean_val:.4f}\nMediana: {median_val:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f4f6f7", ec="#bdc3c7"))

    ax.set_xticks([1])
    ax.set_xticklabels([model_name], fontweight="bold", fontsize=11)
    ax.set_ylabel("Esfuerzo de Control $J_u$ [adimensional]")
    ax.set_title(f"Gasto Energético Acumulado — {model_name}", pad=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {out_path.name}")


def plot_model_reward_components(df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    """
    Genera gráfica independiente mostrando las 3 curvas de componentes de costo/recompensa
    (J_pos: posición, J_vel: velocidad, J_ctrl: esfuerzo de control) a lo largo de los episodios.
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    episodes = np.arange(len(df))
    j_pos = df["j_pos"].values if "j_pos" in df else np.zeros(len(df))
    j_vel = df["j_vel"].values if "j_vel" in df else np.zeros(len(df))
    j_ctrl = df["j_ctrl"].values if "j_ctrl" in df else np.zeros(len(df))

    window = min(15, max(3, len(df) // 10))
    pos_smooth = pd.Series(j_pos).rolling(window=window, min_periods=1).mean().values
    vel_smooth = pd.Series(j_vel).rolling(window=window, min_periods=1).mean().values
    ctrl_smooth = pd.Series(j_ctrl).rolling(window=window, min_periods=1).mean().values

    ax.plot(episodes, pos_smooth, color="#2980b9", linewidth=2.0,
            label=f"Posición $J_{{pos}}$ (Media: {np.mean(j_pos):.2f})")
    ax.fill_between(episodes, pos_smooth, color="#2980b9", alpha=0.15)

    ax.plot(episodes, vel_smooth, color="#27ae60", linewidth=2.0,
            label=f"Velocidad $J_{{vel}}$ (Media: {np.mean(j_vel):.2f})")
    ax.fill_between(episodes, vel_smooth, color="#27ae60", alpha=0.15)

    ax.plot(episodes, ctrl_smooth, color="#e67e22", linewidth=2.0,
            label=f"Control $J_{{ctrl}}$ (Media: {np.mean(j_ctrl):.4f})")
    ax.fill_between(episodes, ctrl_smooth, color="#e67e22", alpha=0.15)

    ax.set_xlabel("Índice de Episodio de Evaluación")
    ax.set_ylabel("Costo Cuadrático Acumulado")
    ax.set_title(f"Componentes del Funcional de Recompensa LQR — {model_name}", pad=10)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, max(1, len(df) - 1))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {out_path.name}")


def plot_model_temporal_trajectory(traj_df: pd.DataFrame, model_name: str, out_path: Path, episode_id: int = 0) -> None:
    """
    Genera figura de respuesta temporal de 3 paneles sincronizados en el tiempo:
      1. Error euclidiano ||e(t)|| frente a la cota r_target = 20 mm.
      2. Coordenadas articulares q1, q2 y velocidades articulares.
      3. Señales de torque tau1, tau2 frente a límites mecánicos de saturación (+- 0.05 N*m).
    """
    apply_plot_style()
    sub_df = traj_df[traj_df["episode_id"] == episode_id]
    if sub_df.empty:
        sub_df = traj_df.iloc[:100]

    t = sub_df["time"].values
    dist_mm = sub_df["distance"].values * 1000.0  # en mm
    th1 = sub_df["theta1"].values
    th2 = sub_df["theta2"].values
    dth1 = sub_df["dtheta1"].values
    dth2 = sub_df["dtheta2"].values
    tau1 = sub_df["tau1"].values
    tau2 = sub_df["tau2"].values

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.0, 7.8), dpi=300, sharex=True)

    # Panel 1: Distancia Cartesiana
    ax1.plot(t, dist_mm, color="#2980b9", linewidth=2.0, label="Distancia cartesiana $\|e(t)\|$")
    ax1.axhline(20.0, color="#c0392b", linestyle="--", linewidth=1.4, label="Tolerancia $r_{target} = 20$ mm")
    ax1.axhspan(0, 20.0, color="#2ecc71", alpha=0.12, label="Banda de estabilización")
    ax1.set_ylabel("Error Cartesiano [mm]")
    ax1.set_title(f"Respuesta Temporal en Episodio {episode_id} — {model_name}", pad=8)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Panel 2: Variables Articulares
    ax2.plot(t, th1, color="#8e44ad", linewidth=1.8, label="$q_1(t)$ [rad]")
    ax2.plot(t, th2, color="#9b59b6", linestyle="--", linewidth=1.8, label="$q_2(t)$ [rad]")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, dth1, color="#16a085", linewidth=1.2, alpha=0.7, label="$\dot{q}_1(t)$ [rad/s]")
    ax2_twin.plot(t, dth2, color="#1abc9c", linestyle=":", linewidth=1.2, alpha=0.7, label="$\dot{q}_2(t)$ [rad/s]")
    ax2.set_ylabel("Posición [rad]")
    ax2_twin.set_ylabel("Velocidad [rad/s]")
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_2t, labels_2t = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_2 + lines_2t, labels_2 + labels_2t, loc="upper right", fontsize=8.0)

    # Panel 3: Torques de Control
    ax3.plot(t, tau1, color="#d35400", linewidth=1.8, label="Torque $\\tau_1(t)$")
    ax3.plot(t, tau2, color="#e67e22", linestyle="--", linewidth=1.8, label="Torque $\\tau_2(t)$")
    ax3.axhline(0.05, color="#c0392b", linestyle=":", linewidth=1.2, label="Límite $\\pm \\tau_{max} = \\pm 0.05$ N·m")
    ax3.axhline(-0.05, color="#c0392b", linestyle=":", linewidth=1.2)
    ax3.set_ylabel("Torque [N·m]")
    ax3.set_xlabel("Tiempo $t$ [s]")
    ax3.set_ylim(-0.065, 0.065)
    ax3.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {out_path.name}")


def plot_model_pareto_tradeoff(df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    """Genera diagrama de dispersión y frontera de Pareto entre error estacionario e_ss y esfuerzo J_u."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)

    e_ss = df["e_ss"].values * 1000.0  # mm
    ju = df["j_u"].values

    # Identificar frontera empírica de Pareto (mínimo e_ss y mínimo j_u)
    sorted_indices = np.argsort(e_ss)
    pareto_e = []
    pareto_ju = []
    min_ju_so_far = float("inf")
    for idx in sorted_indices:
        if ju[idx] < min_ju_so_far:
            pareto_e.append(e_ss[idx])
            pareto_ju.append(ju[idx])
            min_ju_so_far = ju[idx]

    # Dispersión general de episodios
    succ = df["success"].values if "success" in df else np.zeros(len(df), dtype=bool)
    ax.scatter(e_ss[~succ], ju[~succ], color="#7f8c8d", alpha=0.45, s=28, label="No estabilizado")
    ax.scatter(e_ss[succ], ju[succ], color="#2980b9", alpha=0.75, s=34, edgecolors="#1b4f72", label="Estabilizado")

    # Frente de Pareto
    if pareto_e:
        ax.step(pareto_e, pareto_ju, where="post", color="#c0392b", linewidth=1.8, label="Frontera de Pareto empírica")
        ax.scatter(pareto_e, pareto_ju, color="#c0392b", marker="*", s=80, zorder=5)

    ax.axvline(20.0, color="#e74c3c", linestyle="--", linewidth=1.2, label="Tolerancia $r_{target} = 20$ mm")
    ax.set_xlabel("Error Estacionario $e_{ss}$ [mm]")
    ax.set_ylabel("Esfuerzo de Control $J_u$ [adimensional]")
    ax.set_title(f"Espacio de Compromiso Bi-Objetivo — {model_name}", pad=10)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {out_path.name}")


def plot_model_workspace_spatial_error(df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    """Genera mapa espacial cartesiano (x, y) de error terminal en el espacio de trabajo."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)

    tgt_x = df["target_x"].values if "target_x" in df else np.zeros(len(df))
    tgt_y = df["target_y"].values if "target_y" in df else np.zeros(len(df))
    e_ss = df["e_ss"].values * 1000.0  # mm
    succ = df["success"].values if "success" in df else (e_ss <= 20.0)

    # Dibujar límites geométricos del brazo planar (L1=0.10 m, L2=0.11 m)
    r_max = 0.21
    r_min = 0.01
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r_max * np.cos(theta), r_max * np.sin(theta), color="#34495e", linestyle="--", linewidth=1.4, label="Límite cinemático $R_{max} = 0.21$ m")
    ax.plot(r_min * np.cos(theta), r_min * np.sin(theta), color="#7f8c8d", linestyle=":", linewidth=1.0)

    # Mapeo de puntos
    sc = ax.scatter(tgt_x, tgt_y, c=e_ss, cmap="viridis", s=40, edgecolors=["#145a32" if s else "#922b21" for s in succ],
                    linewidths=1.2, alpha=0.88, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Error Estacionario $e_{ss}$ [mm]")

    ax.set_xlabel("Coordenada Cartesiana $X$ [m]")
    ax.set_ylabel("Coordenada Cartesiana $Y$ [m]")
    ax.set_title(f"Distribución Espacial del Error Cartesiano — {model_name}", pad=10)
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {out_path.name}")


def plot_model_learning_curves(eval_data_list: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
                               model_name: str, model_fig_dir: Path) -> None:
    """
    Genera gráficas de aprendizaje (retorno y tasa de éxito) ajustadas automáticamente a una o varias semillas.
    Cada elemento en eval_data_list es (seed, timesteps, results, successes).
    """
    apply_plot_style()
    if not eval_data_list:
        return

    # 1. Gráfica de Retorno Episódico
    fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=300)
    all_means = []
    common_t = eval_data_list[0][1]

    for seed, t, res, _ in eval_data_list:
        m = np.mean(res, axis=1)
        all_means.append(m)
        if len(eval_data_list) > 1:
            ax.plot(t, m, alpha=0.45, linewidth=1.2, label=f"Semilla {seed}")

    all_means_arr = np.array(all_means)
    grand_mean = np.mean(all_means_arr, axis=0)
    grand_std = np.std(all_means_arr, axis=0) if len(eval_data_list) > 1 else np.std(eval_data_list[0][2], axis=1)

    label_mean = f"Retorno Medio ($\mu$) ({len(eval_data_list)} semillas)" if len(eval_data_list) > 1 else "Retorno Medio"
    ax.plot(common_t, grand_mean, color="#27ae60", linewidth=2.2, label=label_mean)
    ax.fill_between(common_t, grand_mean - grand_std, grand_mean + grand_std, color="#27ae60", alpha=0.22,
                    label="Dispersión $\pm 1\sigma$")

    ax.set_xlabel("Pasos de Interacción (Timesteps)")
    ax.set_ylabel("Retorno Acumulado LQR")
    ax.set_title(f"Curva de Aprendizaje: Retorno Episódico — {model_name}", pad=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    p_ret = model_fig_dir / "learning_curve_return.png"
    fig.savefig(p_ret)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {p_ret.name}")

    # 2. Gráfica de Tasa de Éxito
    fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=300)
    all_succ = []
    has_successes = any(s[3] is not None for s in eval_data_list)

    if has_successes:
        for seed, t, _, succ in eval_data_list:
            if succ is not None:
                sr = np.mean(succ, axis=1) * 100.0
                all_succ.append(sr)
                if len(eval_data_list) > 1:
                    ax.plot(t, sr, alpha=0.45, linewidth=1.2, label=f"Semilla {seed}")

        if all_succ:
            all_succ_arr = np.array(all_succ)
            mean_succ = np.mean(all_succ_arr, axis=0)
            std_succ = np.std(all_succ_arr, axis=0)
            label_s = f"Tasa de Estabilización ($\mu$) ({len(eval_data_list)} semillas)" if len(eval_data_list) > 1 else "Tasa de Estabilización"
            ax.plot(common_t, mean_succ, color="#2b5c8f", marker="o", markersize=3.5, linewidth=2.0, label=label_s)
            if len(eval_data_list) > 1:
                ax.fill_between(common_t, np.maximum(0, mean_succ - std_succ), np.minimum(100, mean_succ + std_succ),
                                color="#2b5c8f", alpha=0.20, label="Dispersión $\pm 1\sigma$")

        ax.set_ylim(-2, 105)
        ax.set_ylabel("Tasa de Estabilización (%)")
        ax.set_xlabel("Pasos de Interacción (Timesteps)")
        ax.set_title(f"Curva de Aprendizaje: Tasa de Estabilización — {model_name}", pad=10)
        ax.legend(loc="upper left")
        fig.tight_layout()
        p_succ = model_fig_dir / "learning_curve_success_rate.png"
        fig.savefig(p_succ)
        plt.close(fig)
        print(f">>> [FIGURA] Guardada: {p_succ.name}")

    # 3. Multisemilla consolidada (2 paneles)
    if len(eval_data_list) > 1 and has_successes and all_succ:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.5), dpi=300, sharex=True)
        ax1.plot(common_t, grand_mean, color="#27ae60", linewidth=2.0, label=f"Media $\mu$ ({len(eval_data_list)} semillas)")
        ax1.fill_between(common_t, grand_mean - grand_std, grand_mean + grand_std, color="#27ae60", alpha=0.22)
        ax1.set_ylabel("Retorno Acumulado LQR")
        ax1.set_title(f"Convergencia Multisemilla con Bandas Estadísticas — {model_name}", pad=8)
        ax1.legend(loc="lower right")

        ax2.plot(common_t, mean_succ, color="#2b5c8f", linewidth=2.0, label="Tasa de Éxito $\mu$")
        ax2.fill_between(common_t, np.maximum(0, mean_succ - std_succ), np.minimum(100, mean_succ + std_succ),
                         color="#2b5c8f", alpha=0.20)
        ax2.set_ylabel("Tasa Éxito (%)")
        ax2.set_xlabel("Pasos de Interacción (Timesteps)")
        ax2.set_ylim(-2, 105)
        ax2.legend(loc="upper left")

        fig.tight_layout()
        p_multi = model_fig_dir / "multiseed_learning_curves.png"
        fig.savefig(p_multi)
        plt.close(fig)
        print(f">>> [FIGURA] Guardada: {p_multi.name}")


def plot_model_optimizer_curves(tb_data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
                                model_name: str, model_fig_dir: Path) -> None:
    """
    Genera gráficas estáticas independientes de las pérdidas del optimizador:
      - optimizer_total_loss.png
      - optimizer_policy_loss.png
      - optimizer_value_loss.png
      - optimizer_entropy.png
      - optimizer_approx_kl.png
      - optimizer_explained_variance.png
      - optimizer_convergence_summary.png (panel 2x3)
    """
    apply_plot_style()
    if not tb_data:
        return

    tag_meta = {
        "train/loss": ("optimizer_total_loss.png", "Pérdida Total del Optimizador", "Pérdida Total ($L_{total}$)", "#c0392b"),
        "train/policy_gradient_loss": ("optimizer_policy_loss.png", "Pérdida del Gradiente de la Política", "Pérdida de Política ($L_{clip}$)", "#e67e22"),
        "train/value_loss": ("optimizer_value_loss.png", "Pérdida de la Función de Valor (Crítico)", "Pérdida de Valor ($L_V$)", "#2980b9"),
        "train/entropy_loss": ("optimizer_entropy.png", "Evolución de la Entropía de la Política", "Entropía ($H$)", "#8e44ad"),
        "train/approx_kl": ("optimizer_approx_kl.png", "Divergencia KL Aproximada", "Divergencia KL ($D_{KL}$)", "#16a085"),
        "train/explained_variance": ("optimizer_explained_variance.png", "Varianza Explicada de la Función de Valor", "Varianza Explicada", "#27ae60"),
    }

    # 1. Gráficas independientes por escalar
    for tag, (fname, title, ylabel, color) in tag_meta.items():
        if tag not in tb_data:
            continue
        series_list = tb_data[tag]
        fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)

        for idx, (steps, vals) in enumerate(series_list):
            lbl = f"Semilla {idx + 1}" if len(series_list) > 1 else None
            # Suavizado de media móvil
            w = min(15, max(3, len(vals) // 8))
            smooth_vals = pd.Series(vals).rolling(window=w, min_periods=1).mean().values
            ax.plot(steps, smooth_vals, color=color, alpha=0.85 if len(series_list) == 1 else 0.5,
                    linewidth=1.8, label=lbl)

        ax.set_xlabel("Pasos de Entrenamiento (Timesteps)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {model_name}", pad=10)
        if len(series_list) > 1:
            ax.legend(loc="upper right")

        fig.tight_layout()
        out_f = model_fig_dir / fname
        fig.savefig(out_f)
        plt.close(fig)
        print(f">>> [FIGURA] Guardada: {out_f.name}")

    # 2. Panel Resumen 2x3
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
    axes_flat = axes.flatten()

    for idx, (tag, (_, title, ylabel, color)) in enumerate(tag_meta.items()):
        ax = axes_flat[idx]
        if tag in tb_data:
            for s_idx, (steps, vals) in enumerate(tb_data[tag]):
                w = min(15, max(3, len(vals) // 8))
                smooth_vals = pd.Series(vals).rolling(window=w, min_periods=1).mean().values
                ax.plot(steps, smooth_vals, color=color, linewidth=1.6, alpha=0.85)
            ax.set_title(title, fontsize=10, pad=6)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlabel("Timesteps", fontsize=9)
        else:
            ax.text(0.5, 0.5, f"Sin datos: {tag}", ha="center", va="center", fontsize=9)

    fig.suptitle(f"Panel Consolidado de Convergencia del Optimizador — {model_name}", fontsize=13, y=0.99)
    fig.tight_layout()
    summary_path = model_fig_dir / "optimizer_convergence_summary.png"
    fig.savefig(summary_path)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {summary_path.name}")


# ==============================================================================
# 2. GENERACIÓN DE FIGURAS COMPARATIVAS CONSOLIDADAS
# ==============================================================================

def plot_comparative_success_rates(all_df: pd.DataFrame, out_path: Path) -> None:
    """Genera comparativa de tasa de éxito entre modelos con IC Wilson."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    agents = all_df["agent"].unique()
    x = np.arange(len(agents))
    rates, err_low, err_high = [], [], []

    for a in agents:
        sub = all_df[all_df["agent"] == a]
        succ = int(sub["success"].sum()) if "success" in sub else 0
        total = len(sub)
        p_hat = succ / total if total > 0 else 0.0

        z = 1.95996
        z2 = z ** 2
        denom = 1.0 + z2 / total
        center = (p_hat + z2 / (2.0 * total)) / denom
        spread = z * np.sqrt((p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * (total ** 2)))) / denom
        low = max(0.0, center - spread)
        high = min(1.0, center + spread)

        rates.append(p_hat * 100)
        err_low.append((p_hat - low) * 100)
        err_high.append((high - p_hat) * 100)

    bars = ax.bar(x, rates, yerr=[err_low, err_high], capsize=5,
                  color="#2b5c8f", edgecolor="#1a365d", alpha=0.85, width=0.45)

    ax.set_ylabel("Tasa de Éxito (%) [IC Wilson 95%]")
    ax.set_title("Comparativa: Tasa de Estabilización Asintótica", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y")

    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5,
                f"{r:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [COMPARATIVA] Guardada: {out_path.name}")


def plot_comparative_steady_state_error(all_df: pd.DataFrame, out_path: Path) -> None:
    """Genera comparativa de error cartesiano e_ss entre modelos."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    agents = all_df["agent"].unique()
    data_e = [all_df[all_df["agent"] == a]["e_ss"].values * 1000.0 for a in agents]

    bp = ax.boxplot(data_e, patch_artist=True, showfliers=False, widths=0.45)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.75)

    ax.axhline(20.0, color="#c0392b", linestyle="--", linewidth=1.4, label="Tolerancia r_target = 20 mm")
    ax.set_xticks(range(1, len(agents) + 1))
    ax.set_xticklabels(agents, fontweight="bold")
    ax.set_ylabel("Error Estacionario $e_{ss}$ [mm]")
    ax.set_title("Comparativa: Precisión Cartesiana Terminal", pad=10)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [COMPARATIVA] Guardada: {out_path.name}")


def plot_comparative_residual_velocity(all_df: pd.DataFrame, out_path: Path) -> None:
    """Genera comparativa de velocidad residual v_ss entre modelos."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    agents = all_df["agent"].unique()
    data_v = [all_df[all_df["agent"] == a]["v_ss"].values for a in agents]

    bp = ax.boxplot(data_v, patch_artist=True, showfliers=False, widths=0.45)
    for patch in bp["boxes"]:
        patch.set_facecolor("#1abc9c")
        patch.set_alpha(0.75)

    ax.axhline(0.10, color="#c0392b", linestyle="--", linewidth=1.4, label="Tolerancia v_tol = 0.10 rad/s")
    ax.set_xticks(range(1, len(agents) + 1))
    ax.set_xticklabels(agents, fontweight="bold")
    ax.set_ylabel("Velocidad Residual $v_{ss}$ [rad/s]")
    ax.set_title("Comparativa: Disipación Cinética Terminal", pad=10)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [COMPARATIVA] Guardada: {out_path.name}")


def plot_comparative_settling_time(all_df: pd.DataFrame, out_path: Path) -> None:
    """Genera comparativa de tiempo de establecimiento t_s."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    agents = all_df["agent"].unique()
    data_ts = [all_df[all_df["agent"] == a]["t_s"].values for a in agents]

    bp = ax.boxplot(data_ts, patch_artist=True, showfliers=False, widths=0.45)
    for patch in bp["boxes"]:
        patch.set_facecolor("#f39c12")
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(agents) + 1))
    ax.set_xticklabels(agents, fontweight="bold")
    ax.set_ylabel("Tiempo de Establecimiento $t_s$ [s]")
    ax.set_title("Comparativa: Rapidez de Respuesta Temporal", pad=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [COMPARATIVA] Guardada: {out_path.name}")


def plot_comparative_control_effort(all_df: pd.DataFrame, out_path: Path) -> None:
    """Genera comparativa de esfuerzo de control acumulado J_u."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    agents = all_df["agent"].unique()
    data_ju = [all_df[all_df["agent"] == a]["j_u"].values for a in agents]

    bp = ax.boxplot(data_ju, patch_artist=True, showfliers=False, widths=0.45)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9b59b6")
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(agents) + 1))
    ax.set_xticklabels(agents, fontweight="bold")
    ax.set_ylabel("Esfuerzo de Control $J_u$ [adimensional]")
    ax.set_title("Comparativa: Gasto Energético de Control", pad=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [COMPARATIVA] Guardada: {out_path.name}")


def plot_comparative_reward_components(all_df: pd.DataFrame, out_path: Path) -> None:
    """Genera barras comparativas con las medias de las 3 componentes de costo por agente."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)

    agents = all_df["agent"].unique()
    x = np.arange(len(agents))
    width = 0.25

    mean_pos = [float(all_df[all_df["agent"] == a]["j_pos"].mean()) if "j_pos" in all_df else 0.0 for a in agents]
    mean_vel = [float(all_df[all_df["agent"] == a]["j_vel"].mean()) if "j_vel" in all_df else 0.0 for a in agents]
    mean_ctrl = [float(all_df[all_df["agent"] == a]["j_ctrl"].mean()) if "j_ctrl" in all_df else 0.0 for a in agents]

    ax.bar(x - width, mean_pos, width, label="Costo Posición $J_{pos}$", color="#2980b9", alpha=0.85)
    ax.bar(x, mean_vel, width, label="Costo Velocidad $J_{vel}$", color="#27ae60", alpha=0.85)
    ax.bar(x + width, mean_ctrl, width, label="Costo Control $J_{ctrl}$", color="#e67e22", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontweight="bold")
    ax.set_ylabel("Costo Cuadrático Medio por Episodio")
    ax.set_title("Comparativa: Desglose de Componentes de Costo LQR", pad=10)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [COMPARATIVA] Guardada: {out_path.name}")


def plot_comparative_pareto_tradeoff(all_df: pd.DataFrame, out_path: Path) -> None:
    """Genera dispersión comparativa y frente de Pareto conjunto entre modelos."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)

    agents = all_df["agent"].unique()
    palette = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#16a085", "#c0392b"]
    markers = ["o", "s", "^", "D", "v", "P"]

    for idx, a in enumerate(agents):
        sub = all_df[all_df["agent"] == a]
        e = sub["e_ss"].values * 1000.0
        ju = sub["j_u"].values
        c = palette[idx % len(palette)]
        m = markers[idx % len(markers)]

        ax.scatter(e, ju, color=c, marker=m, alpha=0.4, s=26)
        # Centroide y desviación estándar
        mean_e = np.mean(e)
        mean_ju = np.mean(ju)
        std_e = np.std(e)
        std_ju = np.std(ju)
        ax.errorbar([mean_e], [mean_ju], xerr=[std_e], yerr=[std_ju],
                    fmt=m, color=c, markeredgecolor="black", markeredgewidth=1.2,
                    markersize=9, capsize=4, elinewidth=1.6, label=f"Centroide {a}")

    ax.axvline(20.0, color="#c0392b", linestyle="--", linewidth=1.2, label="Tolerancia $r_{target} = 20$ mm")
    ax.set_xlabel("Error Estacionario $e_{ss}$ [mm]")
    ax.set_ylabel("Esfuerzo de Control $J_u$ [adimensional]")
    ax.set_title("Comparativa: Compromiso Bi-Objetivo y Frontera de Pareto", pad=10)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [COMPARATIVA] Guardada: {out_path.name}")


def plot_comparative_spatial_error(dfs_by_model: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """Genera subplots espaciales comparativos del error en el espacio de trabajo."""
    apply_plot_style()
    agents = list(dfs_by_model.keys())
    n = len(agents)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2), dpi=300, sharey=True)
    if n == 1:
        axes = [axes]

    r_max = 0.21
    r_min = 0.01
    theta = np.linspace(0, 2 * np.pi, 200)

    for idx, a in enumerate(agents):
        ax = axes[idx]
        df = dfs_by_model[a]
        tgt_x = df["target_x"].values if "target_x" in df else np.zeros(len(df))
        tgt_y = df["target_y"].values if "target_y" in df else np.zeros(len(df))
        e_ss = df["e_ss"].values * 1000.0

        ax.plot(r_max * np.cos(theta), r_max * np.sin(theta), color="#34495e", linestyle="--", linewidth=1.2)
        ax.plot(r_min * np.cos(theta), r_min * np.sin(theta), color="#7f8c8d", linestyle=":", linewidth=0.9)

        sc = ax.scatter(tgt_x, tgt_y, c=e_ss, cmap="viridis", vmin=0, vmax=250, s=28, alpha=0.85)
        ax.set_title(a, fontweight="bold", pad=8)
        ax.set_xlabel("X [m]")
        if idx == 0:
            ax.set_ylabel("Y [m]")
        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.25, 0.25)
        ax.set_aspect("equal")

    cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.03)
    cbar.set_label("Error Estacionario $e_{ss}$ [mm]")
    fig.suptitle("Comparativa Espacial del Error Cartesiano en el Espacio de Trabajo", y=1.02)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f">>> [COMPARATIVA] Guardada: {out_path.name}")


def plot_comparative_temporal_trajectories(traj_by_model: Dict[str, pd.DataFrame], out_path: Path, episode_id: int = 0) -> None:
    """Superpone las respuestas temporales de los modelos comparados sobre el mismo episodio."""
    apply_plot_style()
    valid_trajs = {k: v[v["episode_id"] == episode_id] for k, v in traj_by_model.items() if v is not None}
    if not valid_trajs:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 5.8), dpi=300, sharex=True)
    palette = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#16a085"]

    for idx, (m_name, df_t) in enumerate(valid_trajs.items()):
        c = palette[idx % len(palette)]
        t = df_t["time"].values
        d = df_t["distance"].values * 1000.0
        tau_mag = np.sqrt(df_t["tau1"].values**2 + df_t["tau2"].values**2)

        ax1.plot(t, d, color=c, linewidth=1.8, label=m_name)
        ax2.plot(t, tau_mag, color=c, linewidth=1.8, label=m_name)

    ax1.axhline(20.0, color="#c0392b", linestyle="--", linewidth=1.2, label="Tolerancia 20 mm")
    ax1.set_ylabel("Distancia [mm]")
    ax1.set_title(f"Comparativa de Trayectoria Temporal — Episodio {episode_id}", pad=8)
    ax1.legend(loc="upper right", fontsize=8.5)

    ax2.axhline(0.05 * np.sqrt(2), color="#c0392b", linestyle=":", linewidth=1.1, label="Límite magnitud torque")
    ax2.set_ylabel("Magnitud Torque $\|\\tau\|$ [N·m]")
    ax2.set_xlabel("Tiempo $t$ [s]")
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [COMPARATIVA] Guardada: {out_path.name}")


# ==============================================================================
# 3. PIPELINE DE EJECUCIÓN PRINCIPAL
# ==============================================================================

def generate_all_plots(data_dir: Path, out_dir: Path,
                       specific_model: Optional[str] = None,
                       compare_models: Optional[List[str]] = None,
                       eval_log: Optional[Path] = None,
                       files: Optional[List[str]] = None) -> None:
    """Ejecuta el protocolo completo de generación modular de gráficas independientes."""
    dfs_by_model, all_df = load_evaluation_data(data_dir, files)

    # Filtrar según modelos a comparar o procesar
    if compare_models:
        selected_models = [m for m in compare_models if m in dfs_by_model]
        if not selected_models:
            raise ValueError(f"Ninguno de los modelos en {compare_models} existe en {data_dir}. Disponibles: {list(dfs_by_model.keys())}")
        target_comparative_dir = out_dir / "_".join(selected_models)
        models_to_process = selected_models
    elif specific_model:
        models_to_process = [specific_model] if specific_model in dfs_by_model else []
        target_comparative_dir = None
    else:
        models_to_process = list(dfs_by_model.keys())
        target_comparative_dir = out_dir / "comparative"

    print(f"\n{'='*75}")
    print(f" GENERADOR GRÁFICO MODULAR (FASE D)")
    print(f" Modelos seleccionados: {models_to_process}")
    print(f" Directorio base de figuras: {out_dir}")
    print(f"{'='*75}\n")

    # Cargar trayectorias de muestra
    trajs_by_model = {m: load_sample_trajectories(data_dir, m) for m in models_to_process}

    # 1. Gráficas por modelo en carpetas dedicadas
    for model_name in models_to_process:
        df_m = dfs_by_model[model_name]
        model_fig_dir = out_dir / model_name
        model_fig_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Generando figuras individuales para: {model_name} ---")
        plot_model_success_rate(df_m, model_name, model_fig_dir / "success_rate.png")
        plot_model_steady_state_error(df_m, model_name, model_fig_dir / "steady_state_error.png")
        plot_model_residual_velocity(df_m, model_name, model_fig_dir / "residual_velocity.png")
        plot_model_settling_time(df_m, model_name, model_fig_dir / "settling_time.png")
        plot_model_control_effort(df_m, model_name, model_fig_dir / "control_effort.png")
        plot_model_reward_components(df_m, model_name, model_fig_dir / "reward_components.png")
        plot_model_pareto_tradeoff(df_m, model_name, model_fig_dir / "pareto_tradeoff.png")
        plot_model_workspace_spatial_error(df_m, model_name, model_fig_dir / "workspace_spatial_error.png")

        # Gráfica de trayectoria temporal
        if trajs_by_model.get(model_name) is not None:
            plot_model_temporal_trajectory(trajs_by_model[model_name], model_name,
                                          model_fig_dir / "temporal_trajectory_episode.png", episode_id=0)

        # Buscar logs de entrenamiento (multisemilla o semilla única)
        seed_dirs = sorted(list((REPO_ROOT / "experiments").glob(f"**/{model_name}/seed_*")))
        eval_data_list = []
        log_dirs = []

        if eval_log and eval_log.exists():
            data = np.load(eval_log)
            eval_data_list.append((0, data["timesteps"], data["results"], data["successes"] if "successes" in data else None))
            log_dirs.append(eval_log.parent)

        for s_dir in seed_dirs:
            try:
                seed_num = int(s_dir.name.replace("seed_", ""))
            except ValueError:
                seed_num = 0
            npz_candidates = list(s_dir.glob("**/evaluations.npz"))
            if npz_candidates and not eval_log:
                d = np.load(npz_candidates[0])
                eval_data_list.append((seed_num, d["timesteps"], d["results"], d["successes"] if "successes" in d else None))
            l_dir = s_dir / "logs"
            if l_dir.exists():
                log_dirs.append(l_dir)

        # Curvas de evaluación durante entrenamiento
        if eval_data_list:
            print(f">>> Incorporando curvas de evaluación ({len(eval_data_list)} fuentes) para {model_name}")
            plot_model_learning_curves(eval_data_list, model_name, model_fig_dir)

        # Curvas de pérdidas del optimizador desde TensorBoard
        if log_dirs:
            tags_to_extract = [
                "train/loss", "train/policy_gradient_loss", "train/value_loss",
                "train/entropy_loss", "train/approx_kl", "train/explained_variance"
            ]
            tb_scalars = extract_tensorboard_scalars(log_dirs, tags_to_extract)
            if tb_scalars:
                print(f">>> Incorporando telemetría del optimizador desde TensorBoard para {model_name}")
                plot_model_optimizer_curves(tb_scalars, model_name, model_fig_dir)

    # 2. Gráficas comparativas consolidadas
    if target_comparative_dir and len(models_to_process) > 1:
        target_comparative_dir.mkdir(parents=True, exist_ok=True)
        sub_all_df = all_df[all_df["agent"].isin(models_to_process)]
        sub_dfs = {m: dfs_by_model[m] for m in models_to_process}

        print(f"\n--- Generando figuras comparativas consolidadas en: {target_comparative_dir} ---")
        plot_comparative_success_rates(sub_all_df, target_comparative_dir / "comparative_success_rate.png")
        plot_comparative_steady_state_error(sub_all_df, target_comparative_dir / "comparative_steady_state_error.png")
        plot_comparative_residual_velocity(sub_all_df, target_comparative_dir / "comparative_residual_velocity.png")
        plot_comparative_settling_time(sub_all_df, target_comparative_dir / "comparative_settling_time.png")
        plot_comparative_control_effort(sub_all_df, target_comparative_dir / "comparative_control_effort.png")
        plot_comparative_reward_components(sub_all_df, target_comparative_dir / "comparative_reward_components.png")
        plot_comparative_pareto_tradeoff(sub_all_df, target_comparative_dir / "comparative_pareto_tradeoff.png")
        plot_comparative_spatial_error(sub_dfs, target_comparative_dir / "comparative_spatial_error.png")

        valid_traj_comp = {m: trajs_by_model[m] for m in models_to_process if trajs_by_model.get(m) is not None}
        if len(valid_traj_comp) > 1:
            plot_comparative_temporal_trajectories(valid_traj_comp, target_comparative_dir / "comparative_temporal_trajectories.png", episode_id=0)

    print(f"\n>>> [PLOT OK] Proceso completado exitosamente.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="plot.py",
        description="Generación modular e independiente de figuras de evaluación y análisis mecatrónico.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=str, default="reports/data/evaluation",
                        help="Directorio base que contiene las evaluaciones (subcarpetas o archivos CSV).")
    parser.add_argument("--model", type=str, default=None,
                        help="Nombre del modelo específico para graficar únicamente sus figuras (ej. PPO_2).")
    parser.add_argument("--compare", type=str, default=None,
                        help="Lista de modelos separados por comas para generar comparativa combinada exclusiva (ej. PPO_1,PPO_2,SAC_1).")
    parser.add_argument("--files", type=str, default=None,
                        help="Lista opcional de rutas CSV separadas por comas.")
    parser.add_argument("--eval-log", type=str, default=None,
                        help="Ruta al archivo evaluations.npz del log de entrenamiento.")
    parser.add_argument("--out-dir", type=str, default="reports/figures",
                        help="Directorio base para la salida gráfica (se crean subcarpetas por modelo y combinaciones).")
    args = parser.parse_args()

    data_dir = REPO_ROOT / args.data_dir
    out_dir = REPO_ROOT / args.out_dir
    eval_log_path = Path(args.eval_log) if args.eval_log else None
    file_list = [s.strip() for s in args.files.split(",")] if args.files else None
    compare_list = [s.strip() for s in args.compare.split(",") if s.strip()] if args.compare else None

    generate_all_plots(
        data_dir=data_dir,
        out_dir=out_dir,
        specific_model=args.model,
        compare_models=compare_list,
        eval_log=eval_log_path,
        files=file_list,
    )


if __name__ == "__main__":
    main()
