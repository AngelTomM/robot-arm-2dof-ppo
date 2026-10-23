"""
Generador de Figuras y Gráficas de Análisis Comparativo (Fase D).

Opera exclusivamente sobre los resultados estructurados procesados (CSV / JSON)
generados por evaluate.py y train.py, sin depender del simulador físico ni de Gymnasium.

Genera:
  1. Comparativa de tasa de éxito con intervalos de confianza de Wilson.
  2. Distribución de error cartesiano en régimen permanente (e_ss) y velocidad residual (v_ss).
  3. Diagrama de dispersión / barras de tiempo de establecimiento (t_s) y esfuerzo de control (J_u).
  4. Comparativa de costo cuadrático canónico LQR (J_lqr).
"""
import argparse
import glob
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent


def apply_plot_style() -> None:
    """Configura los parámetros estéticos de las figuras."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#eeeeee"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


def load_evaluation_data(data_dir: Path, files: Optional[List[str]] = None) -> pd.DataFrame:
    """Carga y concatena los archivos CSV de episodios evaluados."""
    if files:
        csv_paths = [Path(f) for f in files]
    else:
        csv_paths = sorted(list(data_dir.glob("*_episodes.csv")))

    if not csv_paths:
        raise FileNotFoundError(f"No se encontraron archivos *_episodes.csv en {data_dir}")

    dfs = []
    for p in csv_paths:
        if p.exists():
            df = pd.read_csv(p)
            if "agent" not in df.columns:
                df["agent"] = p.stem.replace("_episodes", "")
            dfs.append(df)

    if not dfs:
        raise ValueError("No se pudieron cargar datos válidos de evaluación.")

    return pd.concat(dfs, ignore_index=True)


def plot_success_rates(df: pd.DataFrame, out_path: Path) -> None:
    """Genera gráfico de barras con tasas de éxito e intervalos de confianza Wilson."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    agents = df["agent"].unique()
    x = np.arange(len(agents))
    rates = []
    err_low = []
    err_high = []

    for a in agents:
        sub = df[df["agent"] == a]
        succ = int(sub["success"].sum())
        total = len(sub)
        p_hat = succ / total if total > 0 else 0.0

        # Intervalo Wilson 95%
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
    ax.set_title("Comparativa de Desempeño: Tasa de Estabilización Asintótica")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y")

    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{r:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {out_path}")


def plot_steady_state_errors(df: pd.DataFrame, out_path: Path) -> None:
    """Genera boxplots comparativos de error cartesiano e_ss y velocidad residual v_ss."""
    apply_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    agents = df["agent"].unique()
    data_e = [df[df["agent"] == a]["e_ss"].values for a in agents]
    data_v = [df[df["agent"] == a]["v_ss"].values for a in agents]

    bp1 = ax1.boxplot(data_e, patch_artist=True, showfliers=False)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#4a90e2")
        patch.set_alpha(0.7)
    ax1.set_xticks(range(1, len(agents) + 1))
    ax1.set_xticklabels(agents, fontweight="bold")
    ax1.axhline(0.06, color="red", linestyle="--", linewidth=1.2, label="Tolerancia e_ss (0.06 m)")
    ax1.set_ylabel("Error Estacionario e_ss (m)")
    ax1.set_title("Precisión Cartesiana en Régimen Permanente")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    bp2 = ax2.boxplot(data_v, patch_artist=True, showfliers=False)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#50e3c2")
        patch.set_alpha(0.7)
    ax2.set_xticks(range(1, len(agents) + 1))
    ax2.set_xticklabels(agents, fontweight="bold")
    ax2.axhline(0.10, color="red", linestyle="--", linewidth=1.2, label="Tolerancia v_ss (0.10 rad/s)")
    ax2.set_ylabel("Velocidad Residual v_ss (rad/s)")
    ax2.set_title("Disipación Cinética en Régimen Permanente")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {out_path}")


def plot_settling_and_effort(df: pd.DataFrame, out_path: Path) -> None:
    """Genera gráfico comparativo de tiempo de establecimiento t_s y esfuerzo de control J_u."""
    apply_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    agents = df["agent"].unique()
    data_ts = [df[df["agent"] == a]["t_s"].values for a in agents]
    data_ju = [df[df["agent"] == a]["j_u"].values for a in agents]

    bp1 = ax1.boxplot(data_ts, patch_artist=True, showfliers=False)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#f5a623")
        patch.set_alpha(0.7)
    ax1.set_xticks(range(1, len(agents) + 1))
    ax1.set_xticklabels(agents, fontweight="bold")
    ax1.set_ylabel("Tiempo de Establecimiento t_s (s)")
    ax1.set_title("Rapidez de Respuesta Temporal")
    ax1.grid(True)

    bp2 = ax2.boxplot(data_ju, patch_artist=True, showfliers=False)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#bd10e0")
        patch.set_alpha(0.7)
    ax2.set_xticks(range(1, len(agents) + 1))
    ax2.set_xticklabels(agents, fontweight="bold")
    ax2.set_ylabel("Esfuerzo de Control J_u")
    ax2.set_title("Costo Energético Acumulado")
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="plot.py",
        description="Generación desacoplada de figuras de evaluación y análisis mecatrónico.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=str, default="reports/data/evaluation",
                        help="Directorio que contiene los archivos CSV generados por evaluate.py.")
    parser.add_argument("--files", type=str, default=None,
                        help="Lista de rutas CSV separadas por comas (opcional).")
    parser.add_argument("--out-dir", type=str, default="reports/figures",
                        help="Directorio destino para las figuras PNG generadas.")
    args = parser.parse_args()

    data_dir = REPO_ROOT / args.data_dir
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    file_list = [s.strip() for s in args.files.split(",")] if args.files else None
    df = load_evaluation_data(data_dir, file_list)

    print(f"\n{'='*75}")
    print(f" GENERACIÓN DE FIGURAS COMPARATIVAS")
    print(f" Datos: {data_dir} ({len(df)} registros totales de {len(df['agent'].unique())} agentes)")
    print(f" Salida: {out_dir}")
    print(f"{'='*75}\n")

    plot_success_rates(df, out_dir / "fig1_success_rate_comparison.png")
    plot_steady_state_errors(df, out_dir / "fig2_steady_state_precision.png")
    plot_settling_and_effort(df, out_dir / "fig3_settling_time_and_effort.png")

    print("\n>>> [PLOT OK] Todas las figuras fueron generadas satisfactoriamente.")


if __name__ == "__main__":
    main()
