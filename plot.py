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
     - reports/figures_<dataset>/<M1>_<M2>_.../ (carpeta combinada bajo la partición correspondiente)
     - comparative_success_rate.png: Comparativa de tasa de estabilización terminal (IC Wilson 95%).
     - comparative_steady_state_error.png: Comparativa de distribución de error permanente (e_ss).
     - comparative_residual_velocity.png: Comparativa de velocidad residual terminal (v_ss).
     - comparative_settling_time.png: Comparativa de tiempo de establecimiento (t_s).
     - comparative_control_effort.png: Comparativa de esfuerzo energético de torque acumulado (J_u).
     - comparative_reward_components.png: Desglose comparativo de costos LQR (J_pos, J_vel, J_ctrl).
     - comparative_pareto_tradeoff.png: Frontera y dispersión bi-objetivo (e_ss vs J_u).
     - comparative_spatial_error.png: Mapeo espacial comparativo del error cartesiano.
     - comparative_temporal_trajectories.png: Superposición de trayectorias temporales (episodio crítico).
     - comparative_explained_variance.png: Varianza explicada de la función de valor (TensorBoard).
     - comparative_approx_kl.png: Divergencia KL aproximada entre políticas consecutivas (TensorBoard).
     - comparative_value_loss.png: Pérdida cuadrática del crítico (TensorBoard).
     - comparative_policy_loss.png: Pérdida del gradiente de la política (TensorBoard).
     - comparative_entropy.png: Evolución de entropía de la política (TensorBoard).
     - comparative_loss.png: Pérdida escalar total del optimizador (TensorBoard).
     - comparative_optimizer_summary.png: Panel consolidado 2x3 de telemetría de optimización.
     - comparative_learning_return.png: Retorno episódico acumulado durante entrenamiento.
     - comparative_learning_success_rate.png: Tasa de estabilización durante entrenamiento.
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


def parse_model_identifier(model_id: str, default_section: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Parsea un identificador de modelo compuesto ('seccion/modelo') o simple ('modelo').
    Retorna una tupla (nombre_modelo, seccion).
    """
    norm = model_id.replace("\\", "/").strip().strip("/")
    if "/" in norm:
        parts = norm.split("/")
        sec = parts[0].strip()
        name = parts[-1].strip()
        return name, sec
    return norm, default_section


def get_model_section(model_name: str, data_dir: Optional[Path] = None, explicit_section: Optional[str] = None) -> str:
    """Infiere o retorna la sección (subcarpeta) del modelo conforme a SSOT buscando en configs/, evaluation o experiments/."""
    m_name, sec_from_id = parse_model_identifier(model_name)
    if explicit_section:
        return explicit_section
    if sec_from_id:
        return sec_from_id

    base_name = m_name.split("_seed")[0] if "_seed" in m_name else m_name

    cfg_matches = list((REPO_ROOT / "configs").glob(f"**/{base_name}.yaml"))
    if cfg_matches:
        return cfg_matches[0].parent.name

    if data_dir and data_dir.exists():
        ep_matches = list(data_dir.glob(f"**/{m_name}/episodes.csv"))
        for ep in ep_matches:
            if not ep.parent.parent.name.startswith("eval_") and ep.parent.parent.name != data_dir.name:
                return ep.parent.parent.name

    exp_matches = list((REPO_ROOT / "experiments").glob(f"**/{base_name}/seed_*"))
    if exp_matches:
        return exp_matches[0].parent.parent.name

    return ""


def discover_evaluation_datasets(data_dir: Path, files: Optional[List[str]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Descubre y particiona las evaluaciones según el tamaño del conjunto de datos (eval_200, eval_300, etc.).
    Retorna un diccionario {dataset_tag: {nombre_modelo: df}}.
    """
    ref_coords = {}
    for ds_name in ["final_eval_set_300.json", "validation_set_200.json"]:
        p_ds = REPO_ROOT / "datasets" / ds_name
        if not p_ds.exists():
            p_ds = REPO_ROOT / "data/datasets" / ds_name
        if p_ds.exists():
            try:
                with open(p_ds, "r", encoding="utf-8") as f:
                    ds = json.load(f)
                    for item in ds:
                        ref_coords[int(item["episode_id"])] = (float(item["target_x"]), float(item["target_y"]))
            except Exception:
                pass

    datasets_map: Dict[str, Dict[str, pd.DataFrame]] = {}

    if files:
        csv_paths = [(Path(f), None, None) for f in files]
    else:
        csv_paths = []
        # 1. Explorar subdirectorios particionados (eval_200, eval_300, etc.) recursivamente
        for part_dir in sorted(data_dir.glob("eval_*")):
            if part_dir.is_dir():
                for sub in sorted(part_dir.glob("**/episodes.csv")):
                    sec = sub.parent.parent.name if sub.parent.parent != part_dir and not sub.parent.parent.name.startswith("eval_") else None
                    csv_paths.append((sub, part_dir.name, sec))

        # 2. Si data_dir apunta directamente a una partición o raíz
        if not csv_paths:
            for sub in sorted(data_dir.glob("**/episodes.csv")):
                sec = sub.parent.parent.name if not sub.parent.parent.name.startswith("eval_") and sub.parent.parent != data_dir else None
                csv_paths.append((sub, data_dir.name if data_dir.name.startswith("eval_") else None, sec))

    for p, known_tag, known_sec in csv_paths:
        if not p.exists() or p.is_dir():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
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

        # Determinar sección
        sec = known_sec
        if not sec and p.parent.parent.name != data_dir.name and not p.parent.parent.name.startswith("eval_"):
            sec = p.parent.parent.name

        # Asegurar coordenadas si no existieran
        if "target_x" not in df.columns and ref_coords:
            df["target_x"] = df["episode_id"].map(lambda ep: ref_coords.get(int(ep), (0.0, 0.0))[0])
            df["target_y"] = df["episode_id"].map(lambda ep: ref_coords.get(int(ep), (0.0, 0.0))[1])

        # Determinar partición de dataset
        if known_tag:
            d_tag = known_tag
        else:
            ep_count = len(df)
            d_tag = f"eval_{ep_count}"

        if d_tag not in datasets_map:
            datasets_map[d_tag] = {}

        # 1. Indexar con clave calificada si existe sección
        if sec:
            qualified_key = f"{sec}/{model_name}"
            datasets_map[d_tag][qualified_key] = df

        # 2. Indexar con clave simple (alias directo)
        if model_name not in datasets_map[d_tag]:
            datasets_map[d_tag][model_name] = df

    return datasets_map


def load_evaluation_data(data_dir: Path, files: Optional[List[str]] = None,
                         dataset_filter: Optional[str] = None) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Carga los resultados tabulares buscando en las particiones o en subdirectorios directos.
    Si se especifica dataset_filter ('200', '300', 'eval_200', 'eval_300'), filtra por dicha partición.
    """
    datasets_map = discover_evaluation_datasets(data_dir, files)
    if not datasets_map:
        raise FileNotFoundError(f"No se encontraron evaluaciones válidas en {data_dir}")

    if dataset_filter:
        norm_filter = dataset_filter.strip().lower()
        if norm_filter in ["200", "eval_200"]:
            target_key = "eval_200"
        elif norm_filter in ["300", "eval_300"]:
            target_key = "eval_300"
        else:
            target_key = norm_filter

        if target_key not in datasets_map:
            raise ValueError(f"No existe la partición {target_key} en {data_dir}. Particiones disponibles: {list(datasets_map.keys())}")
        dfs_by_model = datasets_map[target_key]
    else:
        # Fusionar todas las particiones (si un modelo está en 300 tiene prioridad sobre 200 para la vista consolidada)
        dfs_by_model = {}
        for d_tag in sorted(datasets_map.keys(), reverse=True):
            for m_name, df in datasets_map[d_tag].items():
                if m_name not in dfs_by_model:
                    dfs_by_model[m_name] = df

    if not dfs_by_model:
        raise ValueError("No se pudieron cargar datos válidos de evaluación.")

    all_concat = pd.concat(list(dfs_by_model.values()), ignore_index=True)
    return dfs_by_model, all_concat


def resolve_model_in_partition(
    part_models_dict: Dict[str, pd.DataFrame],
    model_spec: str,
    default_section: Optional[str] = None,
    data_dir: Optional[Path] = None
) -> Tuple[str, str, pd.DataFrame]:
    """
    Resuelve un identificador de modelo (simple 'PPO_7' o calificado 'PPO_eval_500k/PPO_7')
    dentro del diccionario de modelos de una partición evaluada.
    Retorna (nombre_modelo, seccion, dataframe).
    Lanza ValueError ante ambigüedad entre homónimos o KeyError si no existe.
    """
    m_name, sec = parse_model_identifier(model_spec, default_section)

    # 1. Búsqueda con sección calificada o explícita
    if sec:
        qual_key = f"{sec}/{m_name}"
        if qual_key in part_models_dict:
            return m_name, sec, part_models_dict[qual_key]
        if m_name in part_models_dict:
            cand_sec = get_model_section(m_name, data_dir, explicit_section=sec)
            if cand_sec == sec:
                return m_name, sec, part_models_dict[m_name]
        raise KeyError(f"No se encontró el modelo calificado '{qual_key}' en la partición evaluada.")

    # 2. Búsqueda sin sección: detectar coincidencias calificadas exactas
    qual_matches = [k for k in part_models_dict.keys() if k.endswith(f"/{m_name}")]

    if len(qual_matches) > 1:
        secs_found = [k.split("/")[0] for k in qual_matches]
        raise ValueError(
            f"Ambigüedad detectada para el modelo '{m_name}'. Existen múltiples instancias en las secciones: {secs_found}.\n"
            f"Especifique la ruta calificada (ej. '--model {secs_found[0]}/{m_name}') o utilice el flag '--section {secs_found[0]}'."
        )

    if len(qual_matches) == 1:
        matched_sec = qual_matches[0].split("/")[0]
        return m_name, matched_sec, part_models_dict[qual_matches[0]]

    if m_name in part_models_dict:
        inferred_sec = get_model_section(m_name, data_dir)
        return m_name, inferred_sec, part_models_dict[m_name]

    raise KeyError(f"El modelo '{m_name}' no existe en la partición evaluada. Disponibles: {list(part_models_dict.keys())}")


def load_sample_trajectories(data_dir: Path, model_name: str, dataset_tag: Optional[str] = None, section: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Carga sample_trajectories.csv buscando en la partición jerárquica o en la raíz."""
    m_name, sec_from_id = parse_model_identifier(model_name, section)
    candidates = []
    if sec_from_id:
        if dataset_tag:
            candidates.append(data_dir / dataset_tag / sec_from_id / m_name / "sample_trajectories.csv")
        candidates.extend(list(data_dir.glob(f"eval_*/{sec_from_id}/{m_name}/sample_trajectories.csv")))
        candidates.append(data_dir / sec_from_id / m_name / "sample_trajectories.csv")

    if dataset_tag:
        candidates.extend(list((data_dir / dataset_tag).glob(f"**/{m_name}/sample_trajectories.csv")))
    candidates.extend(list(data_dir.glob(f"eval_*/**/{m_name}/sample_trajectories.csv")))
    candidates.extend(list(data_dir.glob(f"**/{m_name}/sample_trajectories.csv")))
    for p in candidates:
        if p.exists() and p.is_file():
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None


def resolve_seed_event_files(ld: Path) -> List[Path]:
    """
    Filtra y retorna exclusivamente los archivos de eventos de TensorBoard correspondientes
    a la sesión canónica de la semilla. Si existen múltiples subcarpetas de ejecución
    (ej. corridas incompletas, abortadas o reiniciadas vs corrida íntegra), selecciona la
    sesión válida que corresponde al modelo evaluado y con mayor progreso temporal,
    descartando sesiones duplicadas o fallidas.
    """
    all_events = sorted(list(ld.glob("**/events.out.tfevents*")), key=lambda p: p.stat().st_mtime)
    if not all_events:
        return []

    by_folder: Dict[Path, List[Path]] = {}
    for ef in all_events:
        by_folder.setdefault(ef.parent, []).append(ef)

    if len(by_folder) == 1:
        return all_events

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return all_events

    eval_npz = ld / "evaluations.npz"
    eval_mtime = eval_npz.stat().st_mtime if eval_npz.exists() else None
    eval_max_ts = None
    if eval_npz.exists():
        try:
            d = np.load(eval_npz)
            ts = d["timesteps"]
            if len(ts) > 0:
                eval_max_ts = int(ts[-1])
        except Exception:
            pass

    sessions = []
    for folder, efs in by_folder.items():
        latest_ef = efs[-1]
        mtime = latest_ef.stat().st_mtime
        min_step, max_step = 0, 0
        try:
            ea = EventAccumulator(str(latest_ef))
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            for cand in ["train/loss", "train/policy_gradient_loss", "train/actor_loss", "rollout/ep_rew_mean"]:
                if cand in tags:
                    sc = ea.Scalars(cand)
                    if sc:
                        min_step = int(sc[0].step)
                        max_step = int(sc[-1].step)
                        break
        except Exception:
            pass

        sessions.append({
            "folder": folder,
            "files": efs,
            "mtime": mtime,
            "min_step": min_step,
            "max_step": max_step
        })

    if eval_mtime is not None:
        valid_candidates = sessions
        if eval_max_ts:
            filtered = [s for s in sessions if s["max_step"] >= eval_max_ts * 0.5]
            if filtered:
                valid_candidates = filtered
        best_sess = min(valid_candidates, key=lambda s: abs(s["mtime"] - eval_mtime))
    else:
        best_sess = max(sessions, key=lambda s: (s["max_step"], s["mtime"]))

    selected_files = list(best_sess["files"])

    if best_sess["min_step"] > 20000:
        predecessors = [s for s in sessions if s != best_sess and s["max_step"] <= best_sess["min_step"]]
        if predecessors:
            best_pred = max(predecessors, key=lambda s: s["max_step"])
            selected_files = list(best_pred["files"]) + selected_files

    return selected_files


def extract_tensorboard_scalars(log_dirs: List[Path], tags: List[str]) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Extrae series temporales de escalares (steps, values) desde archivos de eventos de TensorBoard.
    Concatena y ordena cronológicamente los archivos de eventos de una misma semilla (sesiones reanudadas).
    Retorna un diccionario {tag: [(steps_seed1, values_seed1), (steps_seed2, values_seed2), ...]}.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    results: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {t: [] for t in tags}

    for ld in log_dirs:
        event_files = resolve_seed_event_files(ld)
        if not event_files:
            continue

        seed_series: Dict[str, Dict[int, float]] = {t: {} for t in tags}

        for ef in event_files:
            try:
                ea = EventAccumulator(str(ef))
                ea.Reload()
                available_scalars = ea.Tags().get("scalars", [])
                for t in tags:
                    if t in available_scalars:
                        for e in ea.Scalars(t):
                            seed_series[t][int(e.step)] = float(e.value)
            except Exception:
                continue

        for t in tags:
            if seed_series[t]:
                steps_sorted = np.array(sorted(seed_series[t].keys()))
                vals_sorted = np.array([seed_series[t][s] for s in steps_sorted])
                results[t].append((steps_sorted, vals_sorted))

    return {k: v for k, v in results.items() if len(v) > 0}


# ==============================================================================
# REGISTRO MODULAR Y PARAMÉTRICO DE MÉTRICAS (EVALUACIÓN Y ENTRENAMIENTO)
# ==============================================================================

CANONICAL_METRICS: Dict[str, Dict[str, Any]] = {
    # 1. Escalares del Optimizador (TensorBoard)
    "explained_variance": {
        "category": "optimizer",
        "tag": "train/explained_variance",
        "tags": ["train/explained_variance", "train/final_distance_mean"],
        "filename": "comparative_explained_variance.png",
        "model_filename": "optimizer_explained_variance.png",
        "title": "Comparativa: Varianza Explicada (PPO) / Distancia Final (SAC)",
        "ylabel": "Varianza Explicada / Distancia (m)",
        "ref_line": 1.0,
        "ref_label": "Óptimo ($1.0$)",
        "zero_line": True,
        "y_limits": None,
    },
    "approx_kl": {
        "category": "optimizer",
        "tag": "train/approx_kl",
        "tags": ["train/approx_kl", "train/ent_coef_loss"],
        "filename": "comparative_approx_kl.png",
        "model_filename": "optimizer_approx_kl.png",
        "title": "Comparativa: Divergencia KL (PPO) / Pérdida Temp. α (SAC)",
        "ylabel": "Divergencia KL ($D_{KL}$) / Pérdida $\\alpha$",
        "ref_line": 0.0,
        "ref_label": "Referencia ($0.0$)",
        "zero_line": True,
        "y_limits": None,
    },
    "value_loss": {
        "category": "optimizer",
        "tag": "train/value_loss",
        "tags": ["train/value_loss", "train/critic_loss"],
        "filename": "comparative_value_loss.png",
        "model_filename": "optimizer_value_loss.png",
        "title": "Comparativa: Pérdida Cuadrática de la Función de Valor (Crítico)",
        "ylabel": "Pérdida de Valor ($L_V$ / Crítico)",
        "ref_line": None,
        "ref_label": None,
        "zero_line": False,
        "y_limits": None,
    },
    "policy_loss": {
        "category": "optimizer",
        "tag": "train/policy_gradient_loss",
        "tags": ["train/policy_gradient_loss", "train/actor_loss"],
        "filename": "comparative_policy_loss.png",
        "model_filename": "optimizer_policy_loss.png",
        "title": "Comparativa: Pérdida de la Política (PPO / SAC)",
        "ylabel": "Pérdida de Política ($L_{clip}$ / Actor)",
        "ref_line": 0.0,
        "ref_label": None,
        "zero_line": True,
        "y_limits": None,
    },
    "entropy": {
        "category": "optimizer",
        "tag": "train/entropy_loss",
        "tags": ["train/entropy_loss", "train/ent_coef", "train/ent_coef_loss"],
        "filename": "comparative_entropy.png",
        "model_filename": "optimizer_entropy.png",
        "title": "Comparativa: Evolución de Entropía / Temp. α",
        "ylabel": "Entropía ($H$) / Coef. $\\alpha$",
        "ref_line": None,
        "ref_label": None,
        "zero_line": False,
        "y_limits": None,
    },
    "loss": {
        "category": "optimizer",
        "tag": "train/loss",
        "tags": ["train/loss", "train/critic_loss"],
        "filename": "comparative_loss.png",
        "model_filename": "optimizer_total_loss.png",
        "title": "Comparativa: Pérdida Total / Crítico del Optimizador",
        "ylabel": "Pérdida Total ($L_{total}$ / Crítico)",
        "ref_line": None,
        "ref_label": None,
        "zero_line": False,
        "y_limits": None,
    },
    "optimizer_summary": {
        "category": "optimizer_panel",
        "filename": "comparative_optimizer_summary.png",
        "model_filename": "optimizer_convergence_summary.png",
        "title": "Panel Resumen Comparativo del Optimizador",
    },

    # 2. Curvas de Aprendizaje (Evaluations.npz)
    "learning_return": {
        "category": "learning_curve",
        "filename": "comparative_learning_return.png",
        "model_filename": "learning_curve_return.png",
        "title": "Comparativa: Curva de Aprendizaje de Retorno Episódico",
        "ylabel": "Retorno Acumulado LQR",
    },
    "learning_success_rate": {
        "category": "learning_curve",
        "filename": "comparative_learning_success_rate.png",
        "model_filename": "learning_curve_success_rate.png",
        "title": "Comparativa: Curva de Aprendizaje de Tasa de Estabilización",
        "ylabel": "Tasa de Estabilización (%)",
    },

    # 3. Métricas de Evaluación de Benchmark (episodes.csv)
    "success_rate": {
        "category": "evaluation",
        "filename": "comparative_success_rate.png",
        "model_filename": "success_rate.png",
    },
    "steady_state_error": {
        "category": "evaluation",
        "filename": "comparative_steady_state_error.png",
        "model_filename": "steady_state_error.png",
    },
    "residual_velocity": {
        "category": "evaluation",
        "filename": "comparative_residual_velocity.png",
        "model_filename": "residual_velocity.png",
    },
    "settling_time": {
        "category": "evaluation",
        "filename": "comparative_settling_time.png",
        "model_filename": "settling_time.png",
    },
    "control_effort": {
        "category": "evaluation",
        "filename": "comparative_control_effort.png",
        "model_filename": "control_effort.png",
    },
    "reward_components": {
        "category": "evaluation",
        "filename": "comparative_reward_components.png",
        "model_filename": "reward_components.png",
    },
    "pareto_tradeoff": {
        "category": "evaluation",
        "filename": "comparative_pareto_tradeoff.png",
        "model_filename": "pareto_tradeoff.png",
    },
    "spatial_error": {
        "category": "evaluation",
        "filename": "comparative_spatial_error.png",
        "model_filename": "workspace_spatial_error.png",
    },
    "temporal_trajectories": {
        "category": "evaluation",
        "filename": "comparative_temporal_trajectories.png",
        "model_filename": "temporal_trajectory_episode.png",
    },
}

METRIC_ALIASES: Dict[str, str] = {
    # Escalares del Optimizador
    "explained_variance": "explained_variance",
    "var_exp": "explained_variance",
    "ev": "explained_variance",
    "train/explained_variance": "explained_variance",
    "approx_kl": "approx_kl",
    "kl": "approx_kl",
    "divergence": "approx_kl",
    "train/approx_kl": "approx_kl",
    "value_loss": "value_loss",
    "critic_loss": "value_loss",
    "vl": "value_loss",
    "train/value_loss": "value_loss",
    "policy_loss": "policy_loss",
    "policy_gradient_loss": "policy_loss",
    "pl": "policy_loss",
    "train/policy_gradient_loss": "policy_loss",
    "entropy": "entropy",
    "entropy_loss": "entropy",
    "ent": "entropy",
    "train/entropy_loss": "entropy",
    "loss": "loss",
    "total_loss": "loss",
    "train/loss": "loss",
    "optimizer_summary": "optimizer_summary",
    "optimizer_convergence_summary": "optimizer_summary",
    "actor_loss": "policy_loss",
    "train/actor_loss": "policy_loss",
    "train/critic_loss": "value_loss",
    "ent_coef": "entropy",
    "train/ent_coef": "entropy",
    "ent_coef_loss": "approx_kl",
    "train/ent_coef_loss": "approx_kl",
    "final_distance_mean": "explained_variance",
    "train/final_distance_mean": "explained_variance",
    "train_final_distance": "explained_variance",
    "final_joint_speed_mean": "explained_variance",
    "train/final_joint_speed_mean": "explained_variance",

    # Curvas de Aprendizaje
    "learning_return": "learning_return",
    "return": "learning_return",
    "return_curve": "learning_return",
    "train_return": "learning_return",
    "ret": "learning_return",
    "learning_curve_return": "learning_return",
    "learning_success_rate": "learning_success_rate",
    "train_success": "learning_success_rate",
    "learning_sr": "learning_success_rate",
    "learning_curve_success_rate": "learning_success_rate",
    "learning_success": "learning_success_rate",

    # Métricas de Evaluación
    "success_rate": "success_rate",
    "success": "success_rate",
    "sr": "success_rate",
    "tasa_exito": "success_rate",
    "steady_state_error": "steady_state_error",
    "e_ss": "steady_state_error",
    "error": "steady_state_error",
    "error_estacionario": "steady_state_error",
    "residual_velocity": "residual_velocity",
    "v_ss": "residual_velocity",
    "velocity": "residual_velocity",
    "velocidad_residual": "residual_velocity",
    "settling_time": "settling_time",
    "t_s": "settling_time",
    "time": "settling_time",
    "tiempo_establecimiento": "settling_time",
    "control_effort": "control_effort",
    "j_u": "control_effort",
    "effort": "control_effort",
    "torque": "control_effort",
    "esfuerzo_control": "control_effort",
    "reward_components": "reward_components",
    "costs": "reward_components",
    "cost": "reward_components",
    "componentes_costo": "reward_components",
    "pareto_tradeoff": "pareto_tradeoff",
    "pareto": "pareto_tradeoff",
    "tradeoff": "pareto_tradeoff",
    "frontera_pareto": "pareto_tradeoff",
    "spatial_error": "spatial_error",
    "spatial": "spatial_error",
    "workspace": "spatial_error",
    "mapa_espacial": "spatial_error",
    "temporal_trajectories": "temporal_trajectories",
    "temporal": "temporal_trajectories",
    "trajectory": "temporal_trajectories",
    "trajectories": "temporal_trajectories",
    "trayectorias": "temporal_trajectories",
}

METRIC_MACROS: Dict[str, List[str]] = {
    "eval": [
        "success_rate", "steady_state_error", "residual_velocity",
        "settling_time", "control_effort", "reward_components",
        "pareto_tradeoff", "spatial_error", "temporal_trajectories"
    ],
    "evaluation": [
        "success_rate", "steady_state_error", "residual_velocity",
        "settling_time", "control_effort", "reward_components",
        "pareto_tradeoff", "spatial_error", "temporal_trajectories"
    ],
    "train": [
        "loss", "policy_loss", "value_loss", "entropy",
        "approx_kl", "explained_variance", "optimizer_summary",
        "learning_return", "learning_success_rate"
    ],
    "training": [
        "loss", "policy_loss", "value_loss", "entropy",
        "approx_kl", "explained_variance", "optimizer_summary",
        "learning_return", "learning_success_rate"
    ],
    "optimizer": [
        "loss", "policy_loss", "value_loss", "entropy",
        "approx_kl", "explained_variance", "optimizer_summary"
    ],
    "learning_curves": [
        "learning_return", "learning_success_rate"
    ],
    "all": [
        "success_rate", "steady_state_error", "residual_velocity",
        "settling_time", "control_effort", "reward_components",
        "pareto_tradeoff", "spatial_error", "temporal_trajectories",
        "loss", "policy_loss", "value_loss", "entropy",
        "approx_kl", "explained_variance", "optimizer_summary",
        "learning_return", "learning_success_rate"
    ]
}


def resolve_requested_metrics(metric_arg: Optional[str]) -> List[str]:
    """
    Parsea y expande el argumento --metric / --metrics a una lista de claves canónicas.
    Si metric_arg es None, retorna las 9 métricas de evaluación por defecto (retrocompatibilidad total).
    """
    if not metric_arg or metric_arg.strip().lower() in ["none", "default"]:
        return METRIC_MACROS["eval"]

    tokens = [t.strip().lower() for t in metric_arg.split(",") if t.strip()]
    resolved: List[str] = []

    for tok in tokens:
        if tok in METRIC_MACROS:
            for m in METRIC_MACROS[tok]:
                if m not in resolved:
                    resolved.append(m)
        elif tok in METRIC_ALIASES:
            canonical = METRIC_ALIASES[tok]
            if canonical not in resolved:
                resolved.append(canonical)
        else:
            avail_aliases = sorted(list(set(list(METRIC_ALIASES.keys()) + list(METRIC_MACROS.keys()))))
            raise ValueError(
                f"Métrica desconocida: '{tok}'.\n"
                f"Métricas y macros disponibles: {', '.join(avail_aliases)}"
            )

    return resolved


def load_model_training_telemetry(model_name: str, eval_log: Optional[Path] = None, section: Optional[str] = None) -> Tuple[List[Path], List[Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray]]]]:
    """Recupera los directorios de logs de TensorBoard y las curvas de evaluación (evaluations.npz) para un modelo."""
    m_name, sec = parse_model_identifier(model_name, section)
    target_seed = None
    if "_seed" in m_name:
        base_name, seed_part = m_name.split("_seed", 1)
        seed_part = seed_part.lstrip("_")
        target_seed = f"seed_{seed_part}"
    else:
        base_name = m_name

    if sec:
        if target_seed:
            s_cand = REPO_ROOT / "experiments" / sec / base_name / target_seed
            all_seed_dirs = [s_cand] if s_cand.exists() else sorted(list((REPO_ROOT / "experiments").glob(f"**/{sec}/{base_name}/{target_seed}")))
        else:
            all_seed_dirs = sorted(list((REPO_ROOT / "experiments" / sec / base_name).glob("seed_*")))
            if not all_seed_dirs:
                all_seed_dirs = sorted(list((REPO_ROOT / "experiments").glob(f"**/{sec}/{base_name}/seed_*")))
    else:
        if target_seed:
            all_seed_dirs = sorted(list((REPO_ROOT / "experiments").glob(f"**/{base_name}/{target_seed}")))
        else:
            all_seed_dirs = sorted(list((REPO_ROOT / "experiments").glob(f"**/{base_name}/seed_*")))
    deduped_seeds: Dict[int, Path] = {}
    for s_idx, s_dir in enumerate(all_seed_dirs):
        try:
            seed_num = int(s_dir.name.replace("seed_", ""))
        except ValueError:
            seed_num = s_idx + 1
        if seed_num not in deduped_seeds:
            deduped_seeds[seed_num] = s_dir
        else:
            if s_dir.stat().st_mtime > deduped_seeds[seed_num].stat().st_mtime:
                deduped_seeds[seed_num] = s_dir
    seed_dirs = [deduped_seeds[k] for k in sorted(deduped_seeds.keys())]
    log_dirs: List[Path] = []
    eval_data_list: List[Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []

    if eval_log and eval_log.exists():
        try:
            data = np.load(eval_log)
            eval_data_list.append((0, data["timesteps"], data["results"], data["successes"] if "successes" in data else None))
            log_dirs.append(eval_log.parent)
        except Exception:
            pass

    for s_idx, s_dir in enumerate(seed_dirs):
        try:
            seed_num = int(s_dir.name.replace("seed_", ""))
        except ValueError:
            seed_num = s_idx + 1
        npz_candidates = sorted(list(s_dir.glob("**/evaluations.npz")), key=lambda p: p.stat().st_mtime, reverse=True)
        if npz_candidates and not eval_log:
            try:
                d = np.load(npz_candidates[0])
                ts = d["timesteps"]
                res = d["results"]
                succ = d["successes"] if "successes" in d else None

                # Si el archivo npz fue truncado por resume (comienza en >10.000 pasos),
                # recuperar evaluaciones iniciales registradas en TensorBoard
                if len(ts) > 0 and ts[0] > 10000:
                    l_dir = s_dir / "logs"
                    if l_dir.exists():
                        try:
                            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                            tb_r, tb_s = {}, {}
                            for ef in resolve_seed_event_files(l_dir):
                                ea = EventAccumulator(str(ef))
                                ea.Reload()
                                avail = ea.Tags().get("scalars", [])
                                if "eval/mean_reward" in avail:
                                    for e in ea.Scalars("eval/mean_reward"):
                                        if e.step < ts[0]:
                                            tb_r[int(e.step)] = float(e.value)
                                if "eval/success_rate" in avail:
                                    for e in ea.Scalars("eval/success_rate"):
                                        if e.step < ts[0]:
                                            tb_s[int(e.step)] = float(e.value)
                            if tb_r:
                                extra_ts = np.array(sorted(tb_r.keys()))
                                extra_res = np.array([[tb_r[st]] * res.shape[1] for st in extra_ts])
                                ts = np.concatenate([extra_ts, ts])
                                res = np.concatenate([extra_res, res])
                                if succ is not None:
                                    extra_succ = np.array([[bool(tb_s.get(st, 0.0) >= 0.5)] * succ.shape[1] for st in extra_ts])
                                    succ = np.concatenate([extra_succ, succ])
                        except Exception:
                            pass

                eval_data_list.append((seed_num, ts, res, succ))
            except Exception:
                pass
        l_dir = s_dir / "logs"
        if l_dir.exists():
            log_dirs.append(l_dir)

    return log_dirs, eval_data_list


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

    # Trazar curvas con suavizado de media móvil y dispersión sutil
    window = min(15, max(3, len(df) // 10))
    pos_smooth = pd.Series(j_pos).rolling(window=window, min_periods=1).mean().values
    vel_smooth = pd.Series(j_vel).rolling(window=window, min_periods=1).mean().values
    ctrl_smooth = pd.Series(j_ctrl).rolling(window=window, min_periods=1).mean().values

    # Curva 1: Posición
    ax.plot(episodes, pos_smooth, color="#2980b9", linewidth=2.0,
            label=f"Posición $J_{{pos}}$ (Media: {np.mean(j_pos):.2f})")
    ax.fill_between(episodes, pos_smooth, color="#2980b9", alpha=0.15)

    # Curva 2: Velocidad
    ax.plot(episodes, vel_smooth, color="#27ae60", linewidth=2.0,
            label=f"Velocidad $J_{{vel}}$ (Media: {np.mean(j_vel):.2f})")
    ax.fill_between(episodes, vel_smooth, color="#27ae60", alpha=0.15)

    # Curva 3: Control
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
    ax1.plot(t, dist_mm, color="#2980b9", linewidth=2.0, label=r"Distancia cartesiana $\|e(t)\|$")
    ax1.axhline(20.0, color="#c0392b", linestyle="--", linewidth=1.4, label="Tolerancia $r_{target} = 20$ mm")
    ax1.axhspan(0, 20.0, color="#2ecc71", alpha=0.12, label="Banda de estabilización")
    ax1.set_ylabel("Error Cartesiano [mm]")
    ax1.set_title(f"Respuesta Temporal en Episodio {episode_id} — {model_name}", pad=8)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Panel 2: Variables Articulares
    ax2.plot(t, th1, color="#8e44ad", linewidth=1.8, label="$q_1(t)$ [rad]")
    ax2.plot(t, th2, color="#9b59b6", linestyle="--", linewidth=1.8, label="$q_2(t)$ [rad]")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, dth1, color="#16a085", linewidth=1.2, alpha=0.7, label=r"$\dot{q}_1(t)$ [rad/s]")
    ax2_twin.plot(t, dth2, color="#1abc9c", linestyle=":", linewidth=1.2, alpha=0.7, label=r"$\dot{q}_2(t)$ [rad/s]")
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
                               model_name: str, model_fig_dir: Path,
                               requested_metrics: Optional[List[str]] = None) -> None:
    """
    Genera gráficas de aprendizaje (retorno y tasa de éxito) ajustadas automáticamente a una o varias semillas.
    Cada elemento en eval_data_list es (seed, timesteps, results, successes).
    """
    apply_plot_style()
    if not eval_data_list:
        return

    # Pre-cómputo de retorno
    all_means = []
    common_t = eval_data_list[0][1]
    for seed, t, res, _ in eval_data_list:
        m = np.mean(res, axis=1)
        all_means.append(m)

    all_means_arr = np.array(all_means)
    grand_mean = np.mean(all_means_arr, axis=0)
    grand_std = np.std(all_means_arr, axis=0) if len(eval_data_list) > 1 else np.std(eval_data_list[0][2], axis=1)

    # Pre-cómputo de éxito
    all_succ = []
    has_successes = any(s[3] is not None for s in eval_data_list)
    if has_successes:
        for seed, t, _, succ in eval_data_list:
            if succ is not None:
                sr = np.mean(succ, axis=1) * 100.0
                all_succ.append(sr)

    if all_succ:
        all_succ_arr = np.array(all_succ)
        mean_succ = np.mean(all_succ_arr, axis=0)
        std_succ = np.std(all_succ_arr, axis=0)
    else:
        mean_succ, std_succ = None, None

    # 1. Gráfica de Retorno Episódico
    if requested_metrics is None or "learning_return" in requested_metrics or "learning_curves" in requested_metrics:
        fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=300)
        for seed, t, res, _ in eval_data_list:
            m = np.mean(res, axis=1)
            if len(eval_data_list) > 1:
                ax.plot(t, m, alpha=0.45, linewidth=1.2, label=f"Semilla {seed}")

        label_mean = rf"Retorno Medio ($\mu$) ({len(eval_data_list)} semillas)" if len(eval_data_list) > 1 else "Retorno Medio"
        ax.plot(common_t, grand_mean, color="#27ae60", linewidth=2.2, label=label_mean)
        ax.fill_between(common_t, grand_mean - grand_std, grand_mean + grand_std, color="#27ae60", alpha=0.22,
                        label=r"Dispersión $\pm 1\sigma$")

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
    if (requested_metrics is None or "learning_success_rate" in requested_metrics or "learning_curves" in requested_metrics) and has_successes and all_succ:
        fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=300)
        for seed, t, _, succ in eval_data_list:
            if succ is not None:
                sr = np.mean(succ, axis=1) * 100.0
                if len(eval_data_list) > 1:
                    ax.plot(t, sr, alpha=0.45, linewidth=1.2, label=f"Semilla {seed}")

        label_s = rf"Tasa de Estabilización ($\mu$) ({len(eval_data_list)} semillas)" if len(eval_data_list) > 1 else "Tasa de Estabilización"
        ax.plot(common_t, mean_succ, color="#2b5c8f", marker="o", markersize=3.5, linewidth=2.0, label=label_s)
        if len(eval_data_list) > 1:
            ax.fill_between(common_t, np.maximum(0, mean_succ - std_succ), np.minimum(100, mean_succ + std_succ),
                            color="#2b5c8f", alpha=0.20, label=r"Dispersión $\pm 1\sigma$")

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
    if (requested_metrics is None or "learning_curves" in requested_metrics or ("learning_return" in requested_metrics and "learning_success_rate" in requested_metrics)) and len(eval_data_list) > 1 and has_successes and all_succ:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.5), dpi=300, sharex=True)
        ax1.plot(common_t, grand_mean, color="#27ae60", linewidth=2.0, label=rf"Media $\mu$ ({len(eval_data_list)} semillas)")
        ax1.fill_between(common_t, grand_mean - grand_std, grand_mean + grand_std, color="#27ae60", alpha=0.22)
        ax1.set_ylabel("Retorno Acumulado LQR")
        ax1.set_title(f"Convergencia Multisemilla con Bandas Estadísticas — {model_name}", pad=8)
        ax1.legend(loc="lower right")

        ax2.plot(common_t, mean_succ, color="#2b5c8f", linewidth=2.0, label=r"Tasa de Éxito $\mu$")
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
                                model_name: str, model_fig_dir: Path,
                                requested_metrics: Optional[List[str]] = None) -> None:
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

    is_sac = ("train/actor_loss" in tb_data and "train/approx_kl" not in tb_data) or "SAC" in model_name.upper()

    if is_sac:
        opt_config = [
            ("loss", "optimizer_total_loss.png", "Pérdida del Crítico Q (SAC)", "Pérdida del Crítico ($L_Q$)", "#c0392b", ["train/critic_loss", "train/loss"]),
            ("policy_loss", "optimizer_policy_loss.png", "Pérdida del Actor (SAC)", "Pérdida del Actor ($L_\\pi$)", "#e67e22", ["train/actor_loss", "train/policy_gradient_loss"]),
            ("value_loss", "optimizer_value_loss.png", "Pérdida de Valor Q / Crítico (SAC)", "Pérdida de Valor ($L_Q$ / Crítico)", "#2980b9", ["train/critic_loss", "train/value_loss"]),
            ("entropy", "optimizer_entropy.png", "Temperatura de Entropía α (SAC)", "Temperatura ($\\alpha$)", "#8e44ad", ["train/ent_coef", "train/entropy_loss"]),
            ("approx_kl", "optimizer_approx_kl.png", "Pérdida Temp. Dual α ($L_\\alpha$)", "Pérdida Temp. ($L_\\alpha$)", "#16a085", ["train/ent_coef_loss", "train/approx_kl"]),
            ("explained_variance", "optimizer_explained_variance.png", "Distancia Terminal en Entrenamiento", "Distancia Final Media (m)", "#27ae60", ["train/final_distance_mean", "train/explained_variance"]),
        ]
        alt_filenames = {
            "approx_kl": "optimizer_ent_coef_loss.png",
            "explained_variance": "optimizer_final_distance.png",
        }
    else:
        opt_config = [
            ("loss", "optimizer_total_loss.png", "Pérdida Total del Optimizador", "Pérdida Total ($L_{total}$)", "#c0392b", ["train/loss", "train/critic_loss"]),
            ("policy_loss", "optimizer_policy_loss.png", "Pérdida de la Política", "Pérdida de Política ($L_{clip}$ / Actor)", "#e67e22", ["train/policy_gradient_loss", "train/actor_loss"]),
            ("value_loss", "optimizer_value_loss.png", "Pérdida de la Función de Valor (Crítico)", "Pérdida de Valor ($L_V$ / Crítico)", "#2980b9", ["train/value_loss", "train/critic_loss"]),
            ("entropy", "optimizer_entropy.png", "Evolución de Entropía", "Entropía ($H$)", "#8e44ad", ["train/entropy_loss", "train/ent_coef", "train/ent_coef_loss"]),
            ("approx_kl", "optimizer_approx_kl.png", "Divergencia KL Aproximada", "Divergencia KL ($D_{KL}$)", "#16a085", ["train/approx_kl"]),
            ("explained_variance", "optimizer_explained_variance.png", "Varianza Explicada de la Función de Valor", "Varianza Explicada", "#27ae60", ["train/explained_variance"]),
        ]
        alt_filenames = {}

    # 1. Gráficas independientes por escalar
    for canon_key, fname, title, ylabel, color, cand_tags in opt_config:
        if requested_metrics is not None and canon_key not in requested_metrics:
            continue
        series_list = []
        for ct in cand_tags:
            if ct in tb_data and tb_data[ct]:
                series_list = tb_data[ct]
                break
        if not series_list:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)

        for idx, (steps, vals) in enumerate(series_list):
            lbl = f"Semilla {idx + 1}" if len(series_list) > 1 else None
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
        if canon_key in alt_filenames:
            alt_f = model_fig_dir / alt_filenames[canon_key]
            fig.savefig(alt_f)
            print(f">>> [FIGURA] Guardada: {alt_f.name}")
        plt.close(fig)
        print(f">>> [FIGURA] Guardada: {out_f.name}")

    # 2. Panel Resumen 2x3
    if requested_metrics is None or "optimizer_summary" in requested_metrics:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
        axes_flat = axes.flatten()

        for idx, (canon_key, _, title, ylabel, color, cand_tags) in enumerate(opt_config):
            ax = axes_flat[idx]
            series_list = []
            for ct in cand_tags:
                if ct in tb_data and tb_data[ct]:
                    series_list = tb_data[ct]
                    break
            if series_list:
                for s_idx, (steps, vals) in enumerate(series_list):
                    w = min(15, max(3, len(vals) // 8))
                    smooth_vals = pd.Series(vals).rolling(window=w, min_periods=1).mean().values
                    ax.plot(steps, smooth_vals, color=color, linewidth=1.6, alpha=0.85)
                ax.set_title(title, fontsize=10, pad=6)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_xlabel("Timesteps", fontsize=9)
            else:
                na_msg = f"Sin datos: {cand_tags[0]}"
                ax.text(0.5, 0.5, na_msg, ha="center", va="center", fontsize=9, color="#7f8c8d")

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

    # Paleta con indicador explícito de saturación para e_ss > 250 mm
    cmap = plt.cm.viridis.copy()
    cmap.set_over("#e74c3c")

    for idx, a in enumerate(agents):
        ax = axes[idx]
        df = dfs_by_model[a]
        tgt_x = df["target_x"].values if "target_x" in df else np.zeros(len(df))
        tgt_y = df["target_y"].values if "target_y" in df else np.zeros(len(df))
        e_ss = df["e_ss"].values * 1000.0

        ax.plot(r_max * np.cos(theta), r_max * np.sin(theta), color="#34495e", linestyle="--", linewidth=1.2)
        ax.plot(r_min * np.cos(theta), r_min * np.sin(theta), color="#7f8c8d", linestyle=":", linewidth=0.9)

        sc = ax.scatter(tgt_x, tgt_y, c=e_ss, cmap=cmap, vmin=0, vmax=250, s=28, alpha=0.85)
        ax.set_title(a, fontweight="bold", pad=8)
        ax.set_xlabel("X [m]")
        if idx == 0:
            ax.set_ylabel("Y [m]")
        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.25, 0.25)
        ax.set_aspect("equal")

    cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.03, extend="max")
    cbar.set_label("Error Estacionario $e_{ss}$ [mm] (Rojo: > 250 mm)")
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
    ax2.set_ylabel(r"Magnitud Torque $\|\tau\|$ [N·m]")
    ax2.set_xlabel("Tiempo $t$ [s]")
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [COMPARATIVA] Guardada: {out_path.name}")


def plot_comparative_optimizer_metric(
    models_tb_data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    metric_key: str,
    out_path: Path
) -> None:
    """Genera gráfica comparativa de un escalar del optimizador entre múltiples modelos."""
    apply_plot_style()
    meta = CANONICAL_METRICS[metric_key]
    tag = meta["tag"]
    title = meta["title"]
    ylabel = meta["ylabel"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    palette = ["#2b5c8f", "#27ae60", "#e67e22", "#8e44ad", "#16a085", "#c0392b", "#d35400", "#2c3e50"]
    linestyles = ["-", "--", "-.", ":"]

    has_data = False
    for idx, (m_name, series_list) in enumerate(models_tb_data.items()):
        if not series_list:
            continue
        has_data = True
        color = palette[idx % len(palette)]
        ls = linestyles[(idx // len(palette)) % len(linestyles)]

        if len(series_list) == 1:
            s, v = series_list[0]
            w = min(15, max(3, len(v) // 8))
            sm_v = pd.Series(v).rolling(window=w, min_periods=1).mean().values
            ax.plot(s, sm_v, color=color, linestyle=ls, linewidth=2.0, label=m_name)
            ax.plot(s, v, color=color, alpha=0.15, linewidth=0.8)
        else:
            min_s = max(s[0].min() for s in series_list)
            max_s = min(s[0].max() for s in series_list)
            common_s = np.linspace(min_s, max_s, 100)
            interp = []
            for s, v in series_list:
                w = min(15, max(3, len(v) // 8))
                sm_v = pd.Series(v).rolling(window=w, min_periods=1).mean().values
                interp.append(np.interp(common_s, s, sm_v))
            mean_c = np.mean(interp, axis=0)
            std_c = np.std(interp, axis=0)
            lbl = f"{m_name} (μ, {len(series_list)} semillas)"
            ax.plot(common_s, mean_c, color=color, linestyle=ls, linewidth=2.2, label=lbl)
            ax.fill_between(common_s, mean_c - std_c, mean_c + std_c, color=color, alpha=0.18)

    if not has_data:
        plt.close(fig)
        print(f">>> [OMITIDO] No se encontraron datos para la métrica '{metric_key}' ({tag})")
        return

    # Líneas de referencia
    if meta.get("ref_line") is not None:
        ax.axhline(meta["ref_line"], color="#c0392b", linestyle="--", linewidth=1.2,
                   label=meta.get("ref_label"))
    if meta.get("zero_line"):
        ax.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.0)

    if meta.get("y_limits") is not None:
        ax.set_ylim(meta["y_limits"])

    ax.set_xlabel("Pasos de Entrenamiento (Timesteps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [COMPARATIVA] Guardada: {out_path.name}")


def plot_comparative_optimizer_summary(
    models_tb_dict: Dict[str, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]],
    out_path: Path
) -> None:
    """Genera panel 2x3 con las 6 métricas del optimizador comparando todos los modelos."""
    apply_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=300)
    axes_flat = axes.flatten()

    opt_keys = ["loss", "policy_loss", "value_loss", "entropy", "approx_kl", "explained_variance"]
    palette = ["#2b5c8f", "#27ae60", "#e67e22", "#8e44ad", "#16a085", "#c0392b", "#d35400", "#2c3e50"]

    for idx, key in enumerate(opt_keys):
        ax = axes_flat[idx]
        meta = CANONICAL_METRICS[key]
        tag = meta["tag"]
        ylabel = meta["ylabel"]

        has_data = False
        cand_tags = meta.get("tags", [meta.get("tag")])
        for m_idx, (m_name, tb_data) in enumerate(models_tb_dict.items()):
            series_list = []
            for ct in cand_tags:
                if ct in tb_data and tb_data[ct]:
                    series_list = tb_data[ct]
                    break
            if not series_list:
                continue
            has_data = True
            color = palette[m_idx % len(palette)]
            if len(series_list) == 1:
                s, v = series_list[0]
                w = min(15, max(3, len(v) // 8))
                sm_v = pd.Series(v).rolling(window=w, min_periods=1).mean().values
                ax.plot(s, sm_v, color=color, linewidth=1.6, label=m_name)
            else:
                min_s = max(s[0].min() for s in series_list)
                max_s = min(s[0].max() for s in series_list)
                common_s = np.linspace(min_s, max_s, 100)
                interp = []
                for s, v in series_list:
                    w = min(15, max(3, len(v) // 8))
                    sm_v = pd.Series(v).rolling(window=w, min_periods=1).mean().values
                    interp.append(np.interp(common_s, s, sm_v))
                mean_c = np.mean(interp, axis=0)
                std_c = np.std(interp, axis=0)
                ax.plot(common_s, mean_c, color=color, linewidth=1.8, label=f"{m_name} (μ)")
                ax.fill_between(common_s, mean_c - std_c, mean_c + std_c, color=color, alpha=0.15)

        if not has_data:
            na_msg = "N/A para SAC (Off-Policy)" if key in ["approx_kl", "explained_variance"] else f"Sin datos: {cand_tags[0]}"
            ax.text(0.5, 0.5, na_msg, ha="center", va="center", fontsize=9, color="#7f8c8d")
        else:
            if meta.get("ref_line") is not None:
                ax.axhline(meta["ref_line"], color="#c0392b", linestyle="--", linewidth=1.0)
            if meta.get("zero_line"):
                ax.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=0.8)
            if meta.get("y_limits") is not None:
                ax.set_ylim(meta["y_limits"])
            if idx == 0:
                ax.legend(loc="best", fontsize=8)

        ax.set_title(meta["title"].replace("Comparativa: ", ""), fontsize=10, pad=6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Timesteps", fontsize=9)

    fig.suptitle("Panel Consolidado Comparativo del Optimizador", fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [COMPARATIVA] Guardada: {out_path.name}")


def plot_comparative_learning_curves_metric(
    models_eval_data: Dict[str, List[Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray]]]],
    metric_key: str,
    out_path: Path
) -> None:
    """Genera gráfica comparativa de curvas de aprendizaje (retorno o tasa de éxito) entre modelos."""
    apply_plot_style()
    meta = CANONICAL_METRICS[metric_key]
    title = meta["title"]
    ylabel = meta["ylabel"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    palette = ["#2b5c8f", "#27ae60", "#e67e22", "#8e44ad", "#16a085", "#c0392b", "#d35400", "#2c3e50"]

    has_data = False
    for idx, (m_name, eval_list) in enumerate(models_eval_data.items()):
        if not eval_list:
            continue
        color = palette[idx % len(palette)]
        if metric_key == "learning_return":
            all_means = [np.mean(res, axis=1) for _, _, res, _ in eval_list]
            t = eval_list[0][1]
            if len(all_means) == 1:
                ax.plot(t, all_means[0], color=color, linewidth=2.0, label=m_name)
                std_eval = np.std(eval_list[0][2], axis=1)
                ax.fill_between(t, all_means[0] - std_eval, all_means[0] + std_eval, color=color, alpha=0.15)
            else:
                arr = np.array(all_means)
                grand_m = np.mean(arr, axis=0)
                grand_s = np.std(arr, axis=0)
                ax.plot(t, grand_m, color=color, linewidth=2.2, label=f"{m_name} (μ, {len(all_means)} sem)")
                ax.fill_between(t, grand_m - grand_s, grand_m + grand_s, color=color, alpha=0.18)
            has_data = True
        elif metric_key == "learning_success_rate":
            valid_succ = [(t, succ) for _, t, _, succ in eval_list if succ is not None]
            if not valid_succ:
                continue
            all_sr = [np.mean(succ, axis=1) * 100.0 for _, succ in valid_succ]
            t = valid_succ[0][0]
            if len(all_sr) == 1:
                ax.plot(t, all_sr[0], color=color, marker="o", markersize=3.5, linewidth=2.0, label=m_name)
            else:
                arr = np.array(all_sr)
                grand_m = np.mean(arr, axis=0)
                grand_s = np.std(arr, axis=0)
                ax.plot(t, grand_m, color=color, marker="o", markersize=3.5, linewidth=2.2, label=f"{m_name} (μ, {len(all_sr)} sem)")
                ax.fill_between(t, np.maximum(0, grand_m - grand_s), np.minimum(100, grand_m + grand_s),
                                color=color, alpha=0.18)
            has_data = True

    if not has_data:
        plt.close(fig)
        print(f">>> [OMITIDO] No se encontraron datos para {metric_key}")
        return

    if metric_key == "learning_success_rate":
        ax.set_ylim(-2, 105)

    ax.set_xlabel("Pasos de Interacción (Timesteps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f">>> [COMPARATIVA] Guardada: {out_path.name}")


def _render_model_suite(df_m: pd.DataFrame, model_name: str, model_fig_dir: Path,
                        trajs_by_model: Dict[str, Optional[pd.DataFrame]],
                        eval_log: Optional[Path] = None,
                        metrics: Optional[List[str]] = None,
                        section: Optional[str] = None) -> None:
    """Genera el conjunto de figuras individuales para un modelo específico (filtrable por métrica)."""
    m_name, sec = parse_model_identifier(model_name, section)
    model_fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Generando figuras individuales para: {m_name} (sección: {sec or 'N/A'}) en {model_fig_dir} ---")

    def want(key: str) -> bool:
        return metrics is None or key in metrics

    if want("success_rate"):
        plot_model_success_rate(df_m, m_name, model_fig_dir / "success_rate.png")
    if want("steady_state_error"):
        plot_model_steady_state_error(df_m, m_name, model_fig_dir / "steady_state_error.png")
    if want("residual_velocity"):
        plot_model_residual_velocity(df_m, m_name, model_fig_dir / "residual_velocity.png")
    if want("settling_time"):
        plot_model_settling_time(df_m, m_name, model_fig_dir / "settling_time.png")
    if want("control_effort"):
        plot_model_control_effort(df_m, m_name, model_fig_dir / "control_effort.png")
    if want("reward_components"):
        plot_model_reward_components(df_m, m_name, model_fig_dir / "reward_components.png")
    if want("pareto_tradeoff"):
        plot_model_pareto_tradeoff(df_m, m_name, model_fig_dir / "pareto_tradeoff.png")
    if want("spatial_error"):
        plot_model_workspace_spatial_error(df_m, m_name, model_fig_dir / "workspace_spatial_error.png")

    if want("temporal_trajectories") and trajs_by_model.get(model_name) is not None:
        plot_model_temporal_trajectory(trajs_by_model[model_name], m_name,
                                      model_fig_dir / "temporal_trajectory_episode.png", episode_id=0)

    # Curvas de entrenamiento
    log_dirs, eval_data_list = load_model_training_telemetry(m_name, eval_log, section=sec)

    needs_learning = want("learning_return") or want("learning_success_rate")
    if eval_data_list and (metrics is None or needs_learning):
        print(f">>> Incorporando curvas de evaluación ({len(eval_data_list)} fuentes) para {model_name}")
        plot_model_learning_curves(eval_data_list, model_name, model_fig_dir, requested_metrics=metrics)

    needs_optimizer = any(want(k) for k in ["loss", "policy_loss", "value_loss", "entropy", "approx_kl", "explained_variance", "optimizer_summary"])
    if log_dirs and (metrics is None or needs_optimizer):
        tags_to_extract = [
            "train/loss", "train/policy_gradient_loss", "train/actor_loss",
            "train/value_loss", "train/critic_loss", "train/entropy_loss",
            "train/ent_coef", "train/ent_coef_loss", "train/approx_kl", "train/explained_variance",
            "train/final_distance_mean", "train/final_joint_speed_mean"
        ]
        tb_scalars = extract_tensorboard_scalars(log_dirs, tags_to_extract)
        if tb_scalars:
            print(f">>> Incorporando telemetría del optimizador desde TensorBoard para {model_name}")
            plot_model_optimizer_curves(tb_scalars, model_name, model_fig_dir, requested_metrics=metrics)


def resolve_figures_dir(out_dir: Path, partition_tag: str) -> Path:
    """
    Mapea particiones (eval_200, eval_300, eval_mixed) a carpetas semánticas dedicadas:
      - eval_200   -> figures_valid_200
      - eval_300   -> figures_final_300
      - eval_mixed -> figures_mixed
    Si out_dir apunta directamente a una de estas carpetas, se respeta.
    Si out_dir es 'reports' o 'reports/figures', se rutea a reports/<target_name>.
    """
    norm_tag = partition_tag.strip().lower()
    if norm_tag in ["200", "eval_200", "valid_200", "figures_valid_200"]:
        target_name = "figures_valid_200"
    elif norm_tag in ["300", "eval_300", "final_300", "figures_final_300"]:
        target_name = "figures_final_300"
    elif norm_tag in ["mixed", "eval_mixed", "figures_mixed"]:
        target_name = "figures_mixed"
    else:
        n_str = norm_tag.replace("eval_", "")
        target_name = f"figures_{n_str}" if n_str.isdigit() else f"figures_{norm_tag}"

    if out_dir.name in ["figures_valid_200", "figures_final_300", "figures_mixed"]:
        return out_dir
    if out_dir.name in ["figures", "figures_200", "figures_300"]:
        return out_dir.parent / target_name
    if out_dir.name == "reports":
        return out_dir / target_name
    return out_dir / target_name


def generate_all_plots(data_dir: Path, out_dir: Path,
                       specific_model: Optional[str] = None,
                       compare_models: Optional[List[str]] = None,
                       eval_log: Optional[Path] = None,
                       files: Optional[List[str]] = None,
                       dataset_filter: Optional[str] = None,
                       strict: bool = False,
                       metrics: Optional[str] = None,
                       section: Optional[str] = None) -> None:
    """Ejecuta el protocolo completo de generación modular de gráficas independientes particionadas por dataset."""
    datasets_map = discover_evaluation_datasets(data_dir, files)
    if not datasets_map and not compare_models:
        raise FileNotFoundError(f"No se encontraron evaluaciones válidas en {data_dir}")

    # Normalizar filtro de dataset
    target_partition: Optional[str] = None
    if dataset_filter and dataset_filter.strip().lower() != "all":
        df_norm = dataset_filter.strip().lower()
        if df_norm in ["200", "eval_200"]:
            target_partition = "eval_200"
        elif df_norm in ["300", "eval_300"]:
            target_partition = "eval_300"
        else:
            target_partition = df_norm

    requested_metrics = resolve_requested_metrics(metrics)
    has_eval_metrics = any(CANONICAL_METRICS[m]["category"] == "evaluation" for m in requested_metrics)
    has_opt_metrics = any(CANONICAL_METRICS[m]["category"] in ["optimizer", "optimizer_panel"] for m in requested_metrics)
    has_learn_metrics = any(CANONICAL_METRICS[m]["category"] == "learning_curve" for m in requested_metrics)

    # -------------------------------------------------------------
    # CASO 1: COMPARATIVA EXPLÍCITA (--compare M1,M2,...)
    # -------------------------------------------------------------
    if compare_models:
        is_mixed = False
        chosen_partition = target_partition or "eval_300"
        chosen_dfs = {}
        model_sec_map = {}
        model_disp_names = {}

        # 1.1 Si se especificó explícitamente el dataset
        if target_partition:
            chosen_partition = target_partition
            if target_partition in datasets_map:
                avail_dict = datasets_map[target_partition]
                for m_spec in compare_models:
                    try:
                        m_name, m_sec, df_found = resolve_model_in_partition(avail_dict, m_spec, default_section=section, data_dir=data_dir)
                        chosen_dfs[m_spec] = df_found.copy()
                        model_sec_map[m_spec] = m_sec
                        model_disp_names[m_spec] = m_name
                    except (KeyError, ValueError) as err:
                        if has_eval_metrics:
                            raise ValueError(f"[ERROR] No se pudo resolver '{m_spec}' en {target_partition}: {err}")
            elif has_eval_metrics:
                raise ValueError(f"No existe la partición {target_partition} en {data_dir}. Disponibles: {list(datasets_map.keys())}")
        else:
            # 1.2 Detectar automáticamente partición coincidente
            matching_partitions = []
            for d_tag, m_dict in datasets_map.items():
                all_found = True
                for m_spec in compare_models:
                    try:
                        resolve_model_in_partition(m_dict, m_spec, default_section=section, data_dir=data_dir)
                    except Exception:
                        all_found = False
                        break
                if all_found:
                    matching_partitions.append(d_tag)

            if matching_partitions:
                chosen_partition = "eval_300" if "eval_300" in matching_partitions else matching_partitions[0]
                avail_dict = datasets_map[chosen_partition]
                for m_spec in compare_models:
                    m_name, m_sec, df_found = resolve_model_in_partition(avail_dict, m_spec, default_section=section, data_dir=data_dir)
                    chosen_dfs[m_spec] = df_found.copy()
                    model_sec_map[m_spec] = m_sec
                    model_disp_names[m_spec] = m_name
            else:
                where_models = {}
                for m_spec in compare_models:
                    parts = []
                    for d_tag, m_dict in datasets_map.items():
                        try:
                            resolve_model_in_partition(m_dict, m_spec, default_section=section, data_dir=data_dir)
                            parts.append(d_tag)
                        except Exception:
                            pass
                    where_models[m_spec] = parts

                missing_models = [m for m, parts in where_models.items() if not parts]
                if missing_models and has_eval_metrics:
                    raise FileNotFoundError(f"[ERROR] No se encontraron evaluaciones para los modelos: {missing_models}")

                if has_eval_metrics:
                    if strict:
                        err_lines = [
                            "\n" + "=" * 78,
                            " [ERROR METODOLÓGICO] DISCREPANCIA EN EL TAMAÑO DEL DATASET DE EVALUACIÓN (--strict)",
                            "=" * 78,
                            " No es metodológicamente válido comparar modelos evaluados sobre conjuntos",
                            " de prueba de distinto tamaño (e.g. 200 episodios vs 300 episodios).",
                            "",
                            " Detalle de datasets por modelo solicitado:"
                        ]
                        for m, parts in where_models.items():
                            p_str = ", ".join(parts) if parts else "Sin datos"
                            err_lines.append(f"   • {m:10s} -> {p_str}")
                        err_lines.extend([
                            "",
                            " Solución requerida:",
                            "   1. Para comparar sobre 200 episodios: evalúe los modelos con validation_set_200.",
                            "   2. Para comparar sobre 300 episodios: evalúe los modelos con final_eval_set_300.",
                            "   3. Si existen ambas evaluaciones, especifique el flag: --dataset 200 o --dataset 300.",
                            "=" * 78
                        ])
                        raise ValueError("\n".join(err_lines))

                    # Modo no bloqueante: emitir advertencia metodológica
                    is_mixed = True
                    chosen_partition = "eval_mixed"
                    for m_spec in compare_models:
                        parts = where_models[m_spec]
                        pref_part = "eval_300" if "eval_300" in parts else ("eval_200" if "eval_200" in parts else parts[0])
                        m_name, m_sec, df_found = resolve_model_in_partition(datasets_map[pref_part], m_spec, default_section=section, data_dir=data_dir)
                        chosen_dfs[m_spec] = df_found.copy()
                        model_sec_map[m_spec] = m_sec
                        model_disp_names[m_spec] = m_name

                    warn_lines = [
                        "\n" + "=" * 78,
                        " [ADVERTENCIA METODOLÓGICA] DISCREPANCIA EN EL TAMAÑO DEL DATASET DE EVALUACIÓN",
                        "=" * 78,
                        " Comparación exploratoria entre modelos evaluados sobre conjuntos de distinto tamaño.",
                        " Se genera la comparativa informativa anotando explícitamente el tamaño muestral N",
                        " en las etiquetas y leyendas de cada gráfica.",
                        "",
                        " Detalle de particiones asignadas por modelo:"
                    ]
                    for m_spec, df in chosen_dfs.items():
                        warn_lines.append(f"   • {m_spec:15s} -> N = {len(df)} episodios")
                    warn_lines.extend([
                        "",
                        " Nota: Para forzar consistencia homogénea estricta en el informe final use --strict.",
                        "=" * 78 + "\n"
                    ])
                    print("\n".join(warn_lines))
                else:
                    chosen_partition = "eval_300"

        part_fig_dir = resolve_figures_dir(out_dir, chosen_partition)
        all_secs = [model_sec_map.get(m, get_model_section(m, data_dir, section)) for m in compare_models]
        valid_secs = set(filter(None, all_secs))
        comp_slug = "_".join(m.replace("/", "_") for m in compare_models)

        if len(valid_secs) == 1:
            target_comparative_dir = part_fig_dir / list(valid_secs)[0] / comp_slug
        else:
            target_comparative_dir = part_fig_dir / "comparisons" / comp_slug
        target_comparative_dir.mkdir(parents=True, exist_ok=True)

        row_counts = {m: len(df) for m, df in chosen_dfs.items()} if chosen_dfs else {}
        print(f"\n{'='*75}")
        print(f" GENERADOR GRÁFICO COMPARATIVO — Partición: {chosen_partition}")
        print(f" Modelos: {compare_models} {f'({row_counts})' if row_counts else ''}")
        print(f" Métricas solicitadas: {requested_metrics}")
        print(f" Directorio de salida: {target_comparative_dir}")
        print(f"{'='*75}\n")

        trajs_by_model = {}
        if metrics is None or "temporal_trajectories" in requested_metrics:
            for m_spec in compare_models:
                m_name, m_sec = parse_model_identifier(m_spec, section)
                trajs_by_model[m_spec] = load_sample_trajectories(data_dir, m_name, None, section=m_sec)

        # Generar figuras individuales si no se filtraron métricas específicas
        if metrics is None:
            for m_spec in compare_models:
                if m_spec in chosen_dfs:
                    m_name, m_sec = parse_model_identifier(m_spec, section)
                    m_sec = m_sec or model_sec_map.get(m_spec, "")
                    m_dir = part_fig_dir / m_sec / m_name if m_sec else part_fig_dir / m_name
                    _render_model_suite(chosen_dfs[m_spec], m_name, m_dir, trajs_by_model, eval_log, section=m_sec)

        # Preparar DataFrames para las figuras comparativas de evaluación
        plot_dfs = {}
        sub_all_df = pd.DataFrame()
        # Verificar si hay homónimos o secciones cruzadas para desambiguar etiquetas en gráficos
        disp_names_list = [model_disp_names.get(m, parse_model_identifier(m)[0]) for m in compare_models]
        has_duplicate_names = len(disp_names_list) != len(set(disp_names_list))

        if has_eval_metrics and chosen_dfs:
            for m_spec, df in chosen_dfs.items():
                df_plot = df.copy()
                m_name = model_disp_names.get(m_spec, parse_model_identifier(m_spec)[0])
                m_sec = model_sec_map.get(m_spec, "")
                label_base = f"{m_sec}/{m_name}" if (has_duplicate_names or len(valid_secs) > 1) and m_sec else m_name
                label_with_n = f"{label_base} (N={len(df_plot)})" if is_mixed else label_base
                df_plot["agent"] = label_with_n
                plot_dfs[label_with_n] = df_plot
            sub_all_df = pd.concat(list(plot_dfs.values()), ignore_index=True)

        print(f"\n--- Generando figuras comparativas consolidadas en: {target_comparative_dir} ---")

        # 1. Métricas de Evaluación
        if "success_rate" in requested_metrics and not sub_all_df.empty:
            plot_comparative_success_rates(sub_all_df, target_comparative_dir / "comparative_success_rate.png")
        if "steady_state_error" in requested_metrics and not sub_all_df.empty:
            plot_comparative_steady_state_error(sub_all_df, target_comparative_dir / "comparative_steady_state_error.png")
        if "residual_velocity" in requested_metrics and not sub_all_df.empty:
            plot_comparative_residual_velocity(sub_all_df, target_comparative_dir / "comparative_residual_velocity.png")
        if "settling_time" in requested_metrics and not sub_all_df.empty:
            plot_comparative_settling_time(sub_all_df, target_comparative_dir / "comparative_settling_time.png")
        if "control_effort" in requested_metrics and not sub_all_df.empty:
            plot_comparative_control_effort(sub_all_df, target_comparative_dir / "comparative_control_effort.png")
        if "reward_components" in requested_metrics and not sub_all_df.empty:
            plot_comparative_reward_components(sub_all_df, target_comparative_dir / "comparative_reward_components.png")
        if "pareto_tradeoff" in requested_metrics and not sub_all_df.empty:
            plot_comparative_pareto_tradeoff(sub_all_df, target_comparative_dir / "comparative_pareto_tradeoff.png")
        if "spatial_error" in requested_metrics and plot_dfs:
            plot_comparative_spatial_error(plot_dfs, target_comparative_dir / "comparative_spatial_error.png")
        if "temporal_trajectories" in requested_metrics:
            valid_traj_comp = {m: trajs_by_model[m] for m in compare_models if trajs_by_model.get(m) is not None}
            if len(valid_traj_comp) > 1:
                plot_comparative_temporal_trajectories(valid_traj_comp, target_comparative_dir / "comparative_temporal_trajectories.png", episode_id=0)

        # 2. Métricas del Optimizador (TensorBoard)
        if has_opt_metrics:
            needed_tags = []
            for m in requested_metrics:
                if CANONICAL_METRICS[m]["category"] in ["optimizer", "optimizer_panel"]:
                    cand = CANONICAL_METRICS[m].get("tags", [CANONICAL_METRICS[m].get("tag")])
                    for t in cand:
                        if t and t not in needed_tags:
                            needed_tags.append(t)
            if "optimizer_summary" in requested_metrics:
                for k in ["loss", "policy_loss", "value_loss", "entropy", "approx_kl", "explained_variance"]:
                    for t in CANONICAL_METRICS[k].get("tags", [CANONICAL_METRICS[k].get("tag")]):
                        if t and t not in needed_tags:
                            needed_tags.append(t)

            models_tb_dict = {}
            for m_spec in compare_models:
                m_name, m_sec = parse_model_identifier(m_spec, section)
                m_sec = m_sec or model_sec_map.get(m_spec)
                label_base = f"{m_sec}/{m_name}" if (has_duplicate_names or len(valid_secs) > 1) and m_sec else m_name
                l_dirs, _ = load_model_training_telemetry(m_name, eval_log, section=m_sec)
                models_tb_dict[label_base] = extract_tensorboard_scalars(l_dirs, needed_tags)

            comp_labels = list(models_tb_dict.keys())
            for m_key in requested_metrics:
                if CANONICAL_METRICS[m_key]["category"] == "optimizer":
                    cand_tags = CANONICAL_METRICS[m_key].get("tags", [CANONICAL_METRICS[m_key].get("tag")])
                    scalar_data = {}
                    for lbl in comp_labels:
                        found_series = []
                        for ct in cand_tags:
                            if ct in models_tb_dict[lbl] and models_tb_dict[lbl][ct]:
                                found_series = models_tb_dict[lbl][ct]
                                break
                        scalar_data[lbl] = found_series
                    out_f = target_comparative_dir / CANONICAL_METRICS[m_key]["filename"]
                    plot_comparative_optimizer_metric(scalar_data, m_key, out_f)

            if "optimizer_summary" in requested_metrics:
                plot_comparative_optimizer_summary(models_tb_dict, target_comparative_dir / "comparative_optimizer_summary.png")

        # 3. Métricas de Curvas de Aprendizaje (Evaluations.npz)
        if has_learn_metrics:
            models_eval_data = {}
            for m_spec in compare_models:
                m_name, m_sec = parse_model_identifier(m_spec, section)
                m_sec = m_sec or model_sec_map.get(m_spec)
                label_base = f"{m_sec}/{m_name}" if (has_duplicate_names or len(valid_secs) > 1) and m_sec else m_name
                _, e_list = load_model_training_telemetry(m_name, eval_log, section=m_sec)
                models_eval_data[label_base] = e_list

            for m_key in requested_metrics:
                if CANONICAL_METRICS[m_key]["category"] == "learning_curve":
                    out_f = target_comparative_dir / CANONICAL_METRICS[m_key]["filename"]
                    plot_comparative_learning_curves_metric(models_eval_data, m_key, out_f)

        print(f"\n>>> [PLOT OK] Comparativa completada exitosamente en {target_comparative_dir}.")
        return

    # -------------------------------------------------------------
    # CASO 2: MODELO ESPECÍFICO O LOTE COMPLETO
    # -------------------------------------------------------------
    active_partitions = [target_partition] if target_partition else sorted(datasets_map.keys())

    for part in active_partitions:
        part_models_dict = datasets_map[part]
        part_fig_dir = resolve_figures_dir(out_dir, part)
        part_fig_dir.mkdir(parents=True, exist_ok=True)

        if specific_model:
            try:
                m_name, sec_resolved, df_m = resolve_model_in_partition(part_models_dict, specific_model, default_section=section, data_dir=data_dir)
            except (KeyError, ValueError) as e:
                if target_partition:
                    raise ValueError(f"Error resolviendo modelo {specific_model} en {part}: {e}")
                continue

            models_to_run = [(m_name, sec_resolved, df_m)]
        else:
            # En lote completo, evitar duplicados: recorrer únicamente claves canónicas
            # Si una clave no contiene '/' pero tiene su versión calificada, se omite la simple
            canonical_keys = [k for k in part_models_dict.keys() if "/" in k]
            if not canonical_keys:
                canonical_keys = list(part_models_dict.keys())
            
            models_to_run = []
            for k in sorted(canonical_keys):
                m_name, sec_k = parse_model_identifier(k, section)
                sec_k = sec_k or get_model_section(m_name, data_dir)
                models_to_run.append((m_name, sec_k, part_models_dict[k]))

        if not models_to_run:
            continue

        print(f"\n{'='*75}")
        print(f" PROCESANDO PARTICIÓN: {part} ({len(models_to_run)} modelos)")
        print(f" Directorio de salida: {part_fig_dir}")
        print(f"{'='*75}")

        trajs_by_model = {m: load_sample_trajectories(data_dir, m, part, section=s) for m, s, _ in models_to_run}

        for m_name, sec, df_m in models_to_run:
            m_dir = part_fig_dir / sec / m_name if sec else part_fig_dir / m_name
            _render_model_suite(df_m, m_name, m_dir, trajs_by_model, eval_log, metrics=requested_metrics if metrics else None, section=sec)

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
                        help="Nombre o ruta calificada del modelo para graficar únicamente sus figuras (ej. PPO_2 o PPO_eval_500k/PPO_7).")
    parser.add_argument("--section", type=str, default=None,
                        help="Sección o fase a la que pertenece el modelo (ej. PPO_eval_500k, entorno, SAC_eval_300k).")
    parser.add_argument("--compare", type=str, default=None,
                        help="Lista de modelos (simples o calificados) separados por comas para generar comparativa combinada exclusiva (ej. entorno/PPO_7,PPO_eval_500k/PPO_7).")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filtrar por tamaño de dataset de evaluación: 200, 300, eval_200, eval_300 o all.")
    parser.add_argument("--metric", "--metrics", dest="metrics", type=str, default=None,
                        help="Métrica o lista de métricas separadas por comas (ej. explained_variance, approx_kl, steady_state_error, optimizer, eval, all).")
    parser.add_argument("--files", type=str, default=None,
                        help="Lista opcional de rutas CSV separadas por comas.")
    parser.add_argument("--eval-log", type=str, default=None,
                        help="Ruta al archivo evaluations.npz del log de entrenamiento.")
    parser.add_argument("--out-dir", type=str, default="reports",
                        help="Directorio base para la salida gráfica (se rutea automáticamente a reports/figures_valid_200, reports/figures_final_300 y reports/figures_mixed).")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Exigir consistencia estricta en el tamaño de dataset en comparativas (lanza excepción en lugar de advertencia).")
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
        dataset_filter=args.dataset,
        strict=args.strict,
        metrics=args.metrics,
        section=args.section,
    )


if __name__ == "__main__":
    main()
