"""
FASE CUSTOM — COMPARATIVAS LIBRES ENTRE AGENTES (paso D2)
=========================================================
Genera figuras de comparación entre corridas ARBITRARIAS (p. ej. el PPO y el
SAC propios de un miembro del equipo, fuera de las fases monográficas
a/b/c), leyendo únicamente los artefactos que ``train.py`` dejó en
``experiments/<fase>/<exp>/seed_<seed>/``. No exige reportes históricos ni
esquemas de fase concretos.

USO (vía despachador)
=====================
  # Comparar agentes (uno por config; las semillas van 1-a-1 con configs)
  python scripts/generate_figures.py --phase custom \
      --configs configs/phase_X/tomi/ppo_x3_v1.yaml,configs/phase_X/tomi/sac_smooth_v1.yaml \
      --seeds 42

  # Solo algunas figuras, o saltar/redirigir
  python scripts/generate_figures.py --phase custom --configs ... --figures 1,2,3
  python scripts/generate_figures.py --phase custom --configs ... --skip-checkpoint-eval
  python scripts/generate_figures.py --phase custom --configs ... --out-dir tmp/figs

BANDERAS DEL MODO CUSTOM
========================
  --configs               (OBLIGATORIO) YAML de train.py, uno por agente.
  --experiments           nombres/directorios de corrida 1-a-1 con configs
                          (por defecto: experiment.name del YAML; si el
                          directorio real se llama distinto, pásalo aquí).
  --seeds                 semillas 1-a-1 con configs; default 42 para todos.
  --exps-dir              raíz alternativa de corridas (por defecto deriva
                          la fase de la ruta del YAML con la misma regla de
                          train.py: configs/phase_X/tomi/x.yaml ->
                          experiments/phase_X_tomi/<exp>/seed_<seed>/).
  --figures               subconjunto '1,2,...' (ver tabla).
  --skip-checkpoint-eval  no ejecuta la evaluación formal de checkpoints.
  --refresh-checkpoint-eval re-evalúa aunque exista el CSV anterior.
  --out-dir               directorio alternativo de figuras.

FIGURAS GENERADAS (por defecto en reports/figures/phase_X/)
===========================================================
  1  figX_01_training_reward_curves.png  Curvas de entrenamiento
        rollout/ep_rew_mean del TensorBoard de cada agente (banda media±STD
        entre semillas; las curvas de cada semilla en gris tenue).
  2  figX_02_periodic_evaluation_curve.png  Evaluación periódica del
        callback (logs/evaluations.npz): recompensa media de evaluación vs
        timestep con banda ±STD, puntos individuales de episodio y ESTRELLA
        en el máximo. ESTA es la figura para responder "¿en qué punto del
        entrenamiento estuvo el mejor modelo?": el mejor modelo es justo
        ``models/best_model/best_model.zip`` y el valor marcado ;;
        alternativa: el checkpoint más cercano a ese timestep (cada
        ckpt_<paso>/ lleva su model.zip acoplado a su instantánea).
  3  figX_03_episode_rewards_scatter.png  Recompensas INDIVIDUALES por
        episodio (las del npz) en el mejor punto de evaluación y en el
        final de cada agente: caja + enjambre de puntos (variabilidad).
  4  comparativa.csv (reports/data/phase_X/)  Tabla resumen por agente:
        recompensa final/mejor de evaluación, timestep del mejor, rutas al
        best_model y checkpoint vecino. Se imprime por consola, además.
  5  figX_05_checkpoint_success_curve.png  Evaluación FORMAL (dataset
        congelado n=200, éxito estricto 'hold' + Wilson 95%) de todos los
        checkpoints de cada corrida, con la recompensa media como eje
        secundario; ESTRELLA en el mejor por ÉXITO. Coste: ~1-2 min por
        checkpoint. Omisible con --skip-checkpoint-eval y reusable si
        reports/data/phase_X/checkpoint_eval.csv ya existe (a menos que se
        pase --refresh-checkpoint-eval).

NOTAS
=====
  * Si el mejor por recompensa (fig. 2) no coincide con el mejor por
    éxito (fig. 5), el módulo lo advierte por consola: para la selección
    del modelo a interactuar, el criterio objetivo de la TFM es el éxito
    estricto de la fig. 5.
  * VecNormalize: se evalúa cada checkpoint con su
    vecnormalize_stats.pkl acoplado, si lo tiene (fig. 5). Figuras 1-3 no
    ejecutan el entorno, solo leen registros.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.visualization import common  # noqa: E402

try:
    from tensorboard.backend.event_processing import event_accumulator  # noqa: E402
    _HAS_TB = True
except Exception:
    event_accumulator = None
    _HAS_TB = False

# Columnas de costo LQR y recompensas
_REWARD_COMPONENT_COLS = ["lqr_j_pos", "lqr_j_vel", "lqr_j_ctrl"]


# ---------------------------------------------------------------------------
# Resolución de corridas (mismo SSOT que train.py: la fase es la ruta del YAML)
# ---------------------------------------------------------------------------
def _resolve_run(cfg_path: Path, seed: int, exp_name: str = None,
                 exps_dir: str = None) -> dict:
    """Deriva el directorio de corrida de un YAML con la regla de train.py."""
    import yaml as _yaml
    from train import resolve_phase  # misma SSOT de fase que el entrenamiento

    cfg = _yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    algo = "SAC" if "sac" in cfg else ("PPO" if "ppo" in cfg else None)
    if algo is None:
        raise ValueError(f"{cfg_path}: define sección 'ppo:' o 'sac:'")
    name = exp_name or str(cfg.get("experiment", {}).get("name") or Path(cfg_path).stem)
    root = Path(exps_dir) if exps_dir else REPO_ROOT / "experiments" / resolve_phase(Path(cfg_path).resolve())
    run_dir = root / name / f"seed_{seed}"
    return {"label": name, "exp_dir": run_dir, "cfg": cfg, "algo": algo, "seed": seed}


def _load_tb_rew_mean(run_dir: Path):
    """Lee 'rollout/ep_rew_mean' del TensorBoard de una corrida (lista (v.o.x))."""
    if not _HAS_TB:
        return None
    logs = run_dir / "logs"
    events = sorted(logs.glob("**/events.out.tfevents.*"))
    if not events:
        return None
    ea = event_accumulator.EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if "rollout/ep_rew_mean" not in tags:
        return None
    return [(e.step, e.value) for e in ea.Scalars("rollout/ep_rew_mean")]


def _load_evaluations(run_dir: Path):
    """Lee logs/evaluations.npz del callback periódico -> (timesteps, results, n_eps)."""
    npz_path = run_dir / "logs" / "evaluations.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    timesteps = np.asarray(data["timesteps"], dtype=float)
    results = np.asarray(data["results"], dtype=float)  # (n_evals, n_episodes)
    if results.ndim == 1:  # SB3 antiguo: resultados aplanados
        results = results.reshape(-1, 1)
    return {"timesteps": timesteps, "results": results}


# ---------------------------------------------------------------------------
# Figuras 1-3 (solo lectura de registros; no ejecuta el entorno)
# ---------------------------------------------------------------------------
def _fig_training_curves(agents: dict, colors: dict, out_dir: Path):
    """Fig 1: rollout/ep_rew_mean por agente (TensorBoard)."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    plotted = False
    for label, info in agents.items():
        curve = info.get("tb_rew")
        if not curve:
            continue
        steps = np.array([p[0] for p in curve], dtype=float)
        vals = np.array([p[1] for p in curve], dtype=float)
        ws = max(5, len(vals) // 50)  # ventana móvil de suavizado
        vals_roll = np.convolve(vals, np.ones(ws) / ws, mode="same")
        ax.plot(steps, vals, color=colors[label], alpha=0.25, linewidth=0.8)
        ax.plot(steps, vals_roll, color=colors[label], linewidth=2.0,
                label=f"{label} ({info['algo']})")
        plotted = True
    if not plotted:
        plt.close(fig)
        print("  [FIG 1 OMITIDA] No se encontró TensorBoard con rollout/ep_rew_mean.")
        return False
    ax.set_xlabel("Timesteps de entrenamiento")
    ax.set_ylabel("Recompensa media por episodio (ventana móvil)")
    ax.set_title("Curvas de entrenamiento (rollout/ep_rew_mean)")
    ax.legend()
    common.save_fig_with_metadata(
        fig, out_dir / "figX_01_training_reward_curves.png",
        description="Curvas de entrenamiento (recompensa media por episodio) "
                    f"de {list(agents)} leídas del TensorBoard.",
        phase="X")
    return True


def _fig_periodic_eval(agents: dict, colors: dict, out_dir: Path):
    """Fig 2: evaluación periódica (evaluations.npz) con el mejor punto."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    plotted = False
    summary = {}
    for label, info in agents.items():
        ev = info.get("evals")
        if ev is None:
            continue
        plotted = True
        t, res = ev["timesteps"], ev["results"]
        mean = res.mean(axis=1)
        std = res.std(axis=1)
        c = colors[label]
        ax.plot(t, mean, color=c, marker="o", markersize=5, linewidth=1.8,
                label=f"{label} ({info['algo']})")
        ax.fill_between(t, mean - std, mean + std, color=c, alpha=0.15)
        # puntos individuales por episodio en cada evaluación
        for ti, row in zip(t, res):
            ax.scatter(np.full(len(row), ti), row, color=c, s=8, alpha=0.35,
                       edgecolors="none")
        i_best = int(np.argmax(mean))
        summary[label] = {"timesteps": t[i_best], "best_mean": float(mean[i_best])}
        ax.scatter([t[i_best]], [mean[i_best]], color=c, marker="*", s=380,
                   zorder=5, edgecolor="black", linewidths=0.7,
                   label=f"mejor {label}: paso {int(t[i_best]):,}")
        ax.annotate(f"{float(mean[i_best]):.2f}", (t[i_best], mean[i_best]),
                    textcoords="offset points", xytext=(0, 12), ha="center",
                    fontsize=9, color=c)
    if not plotted:
        plt.close(fig)
        print("  [FIG 2 OMITIDA] No hay logs/evaluations.npz (¿eval_freq=0 en el YAML?)")
        return summary
    ax.set_xlabel("Timesteps de entrenamiento (punto de evaluación)")
    ax.set_ylabel("Recompensa media de evaluación (determinista)")
    ax.set_title("Evaluación periódica durante el entrenamiento\n"
                 "(la estrella marca el mejor modelo del entrenamiento)")
    ax.legend()
    common.save_fig_with_metadata(
        fig, out_dir / "figX_02_periodic_evaluation_curve.png",
        description="Recompensa de evaluación periódica por timestep; puntos "
                    "= episodios individuales; estrella = mejor evaluación.",
        phase="X")
    return summary


def _fig_episode_scatter(agents: dict, colors: dict, out_dir: Path):
    """Fig 3: recompensas individuales por episodio (mejor punto y final)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)
    plotted = False
    for ax, key, title in zip(axes, ("best", "final"),
                              ("Mejor punto de evaluación", "Última evaluación (final)")):
        handle = False
        for label, info in agents.items():
            idx = info.get(key + "_eval_idx")
            if idx is None or idx is False:
                continue
            rewards = info["evals"]["results"][idx]
            c = colors[label]
            ax.boxplot(rewards, positions=[list(agents).index(label)], widths=0.35,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=c, alpha=0.3, color=c),
                       medianprops=dict(color=c))
            ax.scatter(np.full(len(rewards), list(agents).index(label))
                       + np.random.default_rng(42).uniform(-0.14, 0.14, len(rewards)),
                       rewards, color=c, s=14, alpha=0.75, label=f"{label} ({info['algo']})")
            handle = True
        if handle:
            plotted = True
        ax.set_xticks(range(len(agents)), list(agents))
        ax.set_title(title)
        ax.set_xlabel("Agente")
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_ylabel("Recompensa del episodio (individual)")
    fig.suptitle("Recompensas individuales por episodio de evaluación")
    common.save_fig_with_metadata(
        fig, out_dir / "figX_03_episode_rewards_scatter.png",
        description="Distribución de recompensas de cada episodio en el mejor "
                    "punto del entrenamiento (izq.) y en la última evaluación (der.).",
        phase="X")


# ---------------------------------------------------------------------------
# Figura 5: evaluación formal de checkpoints (éxito estricto por checkpoint)
# ---------------------------------------------------------------------------
def _checkpoint_eval_csv_path(data_dir: Path) -> Path:
    return data_dir / "checkpoint_eval.csv"


def _evaluate_checkpoints(agents: dict, data_dir: Path, refresh: bool = False):
    """Fig 5 (insumo): evalúa todos los checkpoints de cada agente (formal n=200)."""
    csv_path = _checkpoint_eval_csv_path(data_dir)
    if csv_path.exists() and not refresh:
        print(f"  [FIG 5] Reusando evaluación previa: {csv_path} "
              "(--refresh-checkpoint-eval para reevaluar).")
        return pd.read_csv(csv_path)

    from src.evaluation.checkpoints import (
        evaluate_single_checkpoint, extract_step_from_dirname)
    from src.evaluation.datasets import save_and_get_datasets

    val_dataset, _ = save_and_get_datasets()
    rows = []
    for label, info in agents.items():
        ckpts = sorted((info["exp_dir"] / "checkpoints").glob("ckpt_*"),
                       key=lambda p: extract_step_from_dirname(p.name))
        if not ckpts:
            print(f"  [FIG 5] {label}: sin checkpoints; se omite.")
            continue
        print(f">>> [EVAL CHECKPOINTS] {label}: {len(ckpts)} checkpoints "
              "@ 200 episodios (dataset congelado 'validation')...")
        for c_dir in ckpts:
            step = extract_step_from_dirname(c_dir.name)
            summ, episodes_df = evaluate_single_checkpoint(
                c_dir, val_dataset, info["cfg"], dataset_name="validation",
                algo_hint=info["algo"], schema="d")
            j_lqr = float(summ.get("mean_j_lqr", np.nan))
            rows.append({
                "experiment": label, "algorithm": info["algo"], "seed": info["seed"],
                "step": step, "checkpoint_dir": str(Path.cwd() / c_dir),
                "n_episodes": int(summ["total_episodes"]),
                "success_rate": summ["success_rate"],
                "ci_low": summ["ci_low"], "ci_high": summ["ci_high"],
                "mean_dist_last_20": summ.get("mean_dist_last_20", np.nan),
                "mean_vel_last_20": summ.get("mean_vel_last_20", np.nan),
                "mean_j_lqr": j_lqr,
                "has_vecnormalize": bool(summ["has_vecnormalize"]),
            })
            print(f"    paso {step:>7,} | éxito Op2={summ['success_rate']*100:5.1f}% "
                  f"| J_LQR={j_lqr:8.2f}")
    df = pd.DataFrame(rows)
    if not df.empty:
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"  [OK] CSV de checkpoints guardado: {csv_path}")
    return df


def _fig_checkpoints(ckpt_df: pd.DataFrame, agents: dict, colors: dict, out_dir: Path):
    """Fig 5: éxito Opción 2 y costo LQR canónico Opción 3 por checkpoint."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    best_by_success = {}
    for label in agents:
        g = ckpt_df[ckpt_df["experiment"] == label].sort_values("step")
        if g.empty:
            continue
        c = colors[label]
        y = g["success_rate"]
        err_low, err_high = (y - g["ci_low"]).abs(), (g["ci_high"] - y).abs()
        ax.errorbar(g["step"], y, yerr=[err_low, err_high], marker="o", markersize=5,
                    color=c, linewidth=1.6, capsize=3, label=f"{label} ({agents[label]['algo']})")
        x_best = g.loc[g["success_rate"].idxmax()]
        best_by_success[label] = {
            "step": int(x_best["step"]),
            "success_rate": float(x_best["success_rate"]),
            "checkpoint_dir": str(x_best["checkpoint_dir"]),
        }
        ax.scatter([x_best["step"]], [x_best["success_rate"]], color=c,
                   marker="*", s=380, zorder=5, edgecolor="black", linewidths=0.7,
                   label=f"mejor {label} por éxito: ckpt_{int(x_best['step']):,}")
    # eje secundario: costo integral J_LQR por checkpoint (línea tenue)
    ax2 = ax.twinx()
    for label in agents:
        g = ckpt_df[ckpt_df["experiment"] == label].sort_values("step")
        if not g.empty and "mean_j_lqr" in g:
            ax2.plot(g["step"], g["mean_j_lqr"], linestyle="--", color=colors[label],
                     alpha=0.45, linewidth=1.2, marker=".", markersize=4)
    ax2.set_ylabel("Costo acumulado canónico J_LQR (Opción 3)", color="#777777")
    ax2.tick_params(axis="y", colors="#777777")
    ax.set_xlabel("Timestep del checkpoint")
    ax.set_ylabel("Tasa de éxito Opción 2 (Estabilización k=20) ± IC Wilson 95%")
    ax.set_title("Selección de checkpoint: Estabilización k=20 (Opción 2) y Costo LQR (Opción 3)")
    ax.legend(fontsize=8, loc="lower right")
    common.save_fig_with_metadata(
        fig, out_dir / "figX_05_checkpoint_success_curve.png",
        description="Éxito de estabilización (Opción 2) y costo integral J_LQR (Opción 3) "
                    "por checkpoint (líneas punteadas = J_LQR en eje secundario; estrella = mejor por éxito).",
        phase="X")
    return best_by_success



# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
def run(out_dir=None, configs=None, experiments=None, seeds=None, figures=None,
        exps_dir=None, skip_checkpoint_eval=False, refresh_checkpoint_eval=False):
    """Genera las figuras de comparación custom entre agentes.

    configs    : lista de YAML de entrenamiento (obligatorio; 1 por agente).
    experiments: override 1-a-1 del nombre de corrida (directorio bajo exps_dir).
    seeds      : semillas 1-a-1 (default 42).
    figures    : subconjunto de IDs [1..5] (None = todas).
    exps_dir   : raíz alternativa de corridas (por defecto SSOT de fase).
    """
    if not configs:
        raise ValueError("--phase custom requiere --configs <yaml1,yaml2,...>")
    main_figure_dir = Path(out_dir) if out_dir else REPO_ROOT / "reports" / "figures" / "phase_X"
    data_dir = REPO_ROOT / "reports" / "data" / "phase_X"
    common.apply_style()

    cfg_paths = [Path(p) for p in configs]
    names = ([s.strip() for s in experiments.split(",")] if experiments else None)
    if names is not None and len(names) != len(cfg_paths):
        raise ValueError("--experiments debe tener tantas entradas como --configs")
    seed_list = seeds if seeds else [42]
    if len(seed_list) != len(cfg_paths):
        if len(seed_list) != 1:
            raise ValueError("--seeds debe tener 1 o tantas entradas como --configs")
        seed_list = seed_list * len(cfg_paths)

    agents = {}
    for i, c_path in enumerate(cfg_paths):
        info = _resolve_run(c_path, seed_list[i], names[i] if names else None, exps_dir)
        if not info["exp_dir"].exists():
            raise FileNotFoundError(f"No existe la corrida: {info['exp_dir']} "
                                    "(¿ajusta --experiments/--seeds/--exps-dir?)")
        info["cfg"]["experiment"] = info["cfg"].get("experiment", {})
        info["tb_rew"] = _load_tb_rew_mean(info["exp_dir"])
        info["evals"] = _load_evaluations(info["exp_dir"])
        if info["evals"] is not None:
            res = info["evals"]["results"]
            i_best = int(np.argmax(res.mean(axis=1)))
            info["best_eval_idx"], info["final_eval_idx"] = i_best, len(res) - 1
        agents[info["label"]] = info

    colors_obj = common.color_resolver({})
    colors = {k: colors_obj(k) for k in agents}
    wanted = set(figures) if figures else {1, 2, 3, 4, 5}

    print(f"\n{'=' * 78}\n FASE CUSTOM (paso D2) — agentes: {list(agents)}\n"
          f" salidas: figuras -> {main_figure_dir} | datos -> {data_dir}\n{'=' * 78}")

    generated = []
    if 1 in wanted and _fig_training_curves(agents, colors, main_figure_dir):
        generated.append(1)
    best_summary = None
    if 2 in wanted:
        best_summary = _fig_periodic_eval(agents, colors, main_figure_dir)
        if best_summary:
            generated.append(2)
    if 3 in wanted and any(a.get("evals") is not None for a in agents.values()):
        _fig_episode_scatter(agents, colors, main_figure_dir)
        generated.append(3)

    # Fig 4: CSV comparativa + guía de selección para la interacción
    if 2 in wanted and best_summary and 4 in wanted:
        rows = []
        for label, info in agents.items():
            ev = info["evals"]
            mean_r = ev["results"].mean(axis=1)
            best_i = int(np.argmax(mean_r))
            rows.append({
                "experiment": label,
                "algorithm": info["algo"],
                "seed": info["seed"],
                "run_dir": str(info["exp_dir"]),
                "n_evals": len(ev["timesteps"]),
                "final_timestep": int(ev["timesteps"][-1]),
                "final_mean_eval_reward": float(mean_r[-1]),
                "best_timestep": int(ev["timesteps"][best_i]),
                "best_mean_eval_reward": float(mean_r[best_i]),
                "best_model_dir": str(info["exp_dir"] / "models" / "best_model"),
                "note": "interacción: python interactive_render.py <best_model_dir>/best_model.zip",
            })
        comp = pd.DataFrame(rows)
        data_dir.mkdir(parents=True, exist_ok=True)
        comp.to_csv(data_dir / "comparativa.csv", index=False)
        print("\n  === COMPARATIVA (recompensa de evaluación) ===")
        print(comp[["experiment", "algorithm", "final_mean_eval_reward",
                    "best_mean_eval_reward", "best_timestep"]].to_string(index=False))
        generated.append(4)

    # Fig 5: selección de checkpoint por éxito formal
    best_success = None
    if 5 in wanted and not skip_checkpoint_eval:
        ckpt_df = _evaluate_checkpoints(agents, data_dir, refresh=refresh_checkpoint_eval)
        if not ckpt_df.empty:
            best_success = _fig_checkpoints(ckpt_df, agents, colors, main_figure_dir)
            generated.append(5)
            if best_summary and best_success:
                for label in agents:
                    t_rew = best_summary[label]["timesteps"]
                    t_suc = best_success[label]["step"]
                    if abs(t_suc - t_rew) > 0.25 * max(t_rew, t_suc):
                        print(f"  [AVISO] {label}: el mejor por recompensa (paso "
                              f"{int(t_rew):,}) NO coincide con el mejor por éxito "
                              f"formal (ckpt_{t_suc:,}); usa la figX_05 para elegir.")
        if best_success:
            print("\n  === SELECCIÓN PARA LA INTERACCIÓN ===")
            for label, b in best_success.items():
                print(f"  {label}: ckpt_{b['step']:,} (éxito {b['success_rate']*100:.1f}%) "
                      f"-> {b['checkpoint_dir']}")
            print("  python interactive_render.py <ruta arriba>/model.zip")

    print(f"\n>>> [OK] FASE CUSTOM: figuras {generated} en {main_figure_dir}")
