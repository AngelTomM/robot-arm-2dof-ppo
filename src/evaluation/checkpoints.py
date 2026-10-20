"""
Evaluación de Checkpoints (núcleo unificado de Fases B, B/ext y C).

Sustituye a los antiguos scripts phase_b_eval_checkpoints.py,
phase_b_ext_eval_checkpoints.py y phase_c_eval_checkpoints.py, cuyos
cuerpos se conservan verbatim en los runners de este módulo.

FIGURAS... EVALUACIONES
=======================
Evaluación sistemática de checkpoints guardados (carpetas
``ckpt_<paso>/`` con ``model.zip`` + ``vecnormalize_stats.pkl``) sobre los
conjuntos de validación congelados, en modo determinista riguroso
(``evaluate_on_dataset`` / ``src.evaluation.evaluator.py``):
- Fase B:      ``validation`` (200 episodios).
- B/extensión: ``validation_b2`` (500 episodios) y final ``final_b2``.
- Fase C:      ``validation_c`` (500 episodios), modos determinista y
               estocástico diagnóstico.

Salida por checkpoint: tasa de éxito estricta con intervalo de Wilson al
95 %, medianas de pasos al éxito, distancias medias, suavidad
(``mean_action_delta``) y tasas de truncamiento/balística/oscilatoria.

INSUMOS REQUERIDOS
==================
- Checkpoints ``experiments/<fase>/<exp>/seed_<seed>/checkpoints/ckpt_<paso>/``.
- Datasets en ``data/datasets/`` (validation_set_200.json,
  phase_b_ext/validation_b2_500.json, phase_c/validation_c_500.json).
- ``configs/phase_a/thresholds.yaml`` (umbrales de éxito estricto).

USO
===
Punto único de entrada (fases B y C; B/ext se orquesta con sus scripts
batch existentes):

    python scripts/evaluate_checkpoints.py --phase b --exp_dir <directorio> \
        --config <yaml>
    python scripts/evaluate_checkpoints.py --phase c --dataset <json> \
        --output-dir <dir> --mode deterministic|stochastic

Invocación como biblioteca:

    from src.evaluation.checkpoints import (
        evaluate_single_checkpoint, evaluate_c0_controller,
        evaluate_experiment_checkpoints, run_phase_c_evaluation,
    )

PARÁMETROS CONFIGURABLES
========================
evaluate_single_checkpoint(ckpt_dir, val_dataset, env_cfg,
                           dataset_name="validation_b2",
                           eval_mode="deterministic",
                           thresholds_path="configs/phase_a/thresholds.yaml",
                           algo_hint=None, schema="b")
    ckpt_dir      : ``Path`` a la carpeta ``ckpt_<paso>`` (model.zip con
                    respaldo a cualquier ``*.zip`` del directorio).
    val_dataset   : episodios del conjunto de validación (``list``/``json``).
    env_cfg       : configuración del entorno dict (YAML de la fase o dict
                    inline como en la Fase C).
    dataset_name  : etiqueta del conjunto para el evaluador. Valores
                    posibles: ``"validation"`` (Fase B), ``"validation_b2"``
                    (extensión; por defecto, compatibilidad histórica con
                    las llamadas posicionales), ``"validation_c"`` (Fase C).
    eval_mode     : ``"deterministic"`` (por defecto, protocolo del
                    proyecto) o ``"stochastic"`` (diagnóstico).
    thresholds_path: YAML de umbrales (por defecto thresholds.yaml).
    algo_hint     : ``None`` (autodetección SAC↔PPO), ``"SAC"`` o ``"PPO"``.
    schema        : nomenclatura de claves del resumen. Valores posibles:
                    ``"b"`` (Fase B/ext: ``total_episodes``,
                    ``strict_success_rate``, ``steps_to_success_median``,
                    ``has_vecnormalize``) o ``"c"`` (Fase C:
                    ``n_episodes``, ``success_rate``,
                    ``median_steps_to_success``, ``frac_entered_radius``).
    Devuelve ``(summary, episodes_df)``.

evaluate_c0_controller(val_dataset, env_cfg, max_speed=1.5, label=None,
                       dataset_name="validation_b2")
    Evaluación del controlador analítico C0 (JacobianDampedController,
    kp=3.0, lambda=0.01); ``label`` asigna ``experiment="C0_<label>"``.
    Devuelve ``(summary, episodes_df)`` en esquema "b".

evaluate_experiment_checkpoints(exp_dir, cfg_path, output_csv=None)
    Runner de Fase B: evalúa todos los ``ckpt_*`` de una corrida ordenados
    por pasos (dataset ``validation``), añade ``step``, ``experiment`` y
    ``seed``, y opcionalmente exporta ``output_csv``.

run_phase_c_evaluation(dataset_path=..., output_dir=..., max_step=655360,
                       eval_mode="deterministic")
    Runner de Fase C: S1 (SAC, experiments/phase_c/S1) y X3 (PPO,
    experiments/phase_b/X3) sobre ``validation_c``, más la referencia C0
    (``C0_v{max_speed}``, ``seed="analytic"``); exporta
    ``checkpoints_eval_validation_details[_<modo>].csv`` y su versión
    agregada (media ± std ddof=1 inter-semilla).

extract_step_from_dirname(dir_name)
    Extrae el paso de nombres tipo ``ckpt_40960``; devuelve -1 si no hay
    dígitos.

load_agent_model(model_path, algo_hint=None)
    Carga SB3 probando SAC primero si ``algo_hint``/la ruta indican S1;
    si no, PPO primero.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO, SAC

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import evaluate_on_dataset
from src.evaluation.metrics import wilson_score_interval
from src.controllers.jacobian_controller import JacobianDampedController


def extract_step_from_dirname(dir_name: str) -> int:
    """Extrae el número de paso de nombres tipo ckpt_40960 o ppo_X5_near_40960_steps."""
    m = re.search(r"(\d+)", str(dir_name))
    return int(m.group(1)) if m else -1


def load_agent_model(model_path: Path, algo_hint: str = None):
    """Carga modelo SB3 detectando automáticamente si es SAC o PPO."""
    if algo_hint == "SAC" or "S1" in str(model_path):
        try:
            return SAC.load(str(model_path))
        except Exception:
            return PPO.load(str(model_path))
    else:
        try:
            return PPO.load(str(model_path))
        except Exception:
            return SAC.load(str(model_path))


def _resolve_model_path(ckpt_dir: Path) -> Path:
    """model.zip con respaldo a cualquier *.zip del directorio del checkpoint."""
    model_path = ckpt_dir / "model.zip"
    if not model_path.exists():
        zips = list(ckpt_dir.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"No se encontró archivo de modelo .zip en {ckpt_dir}")
        model_path = zips[0]
    return model_path


def _operational_metrics(episodes_df: pd.DataFrame) -> dict:
    """Métricas operativas canónicas de un bloque de episodios (nombres internos)."""
    n_total = len(episodes_df)
    n_succ = int(episodes_df["success"].sum())
    p_hat, ci_low, ci_high = wilson_score_interval(n_succ, n_total)

    succ_df = episodes_df[episodes_df["success"]]
    med_steps = float(succ_df["steps"].median()) if len(succ_df) > 0 else np.nan
    mean_dist = float(episodes_df["final_distance"].mean())
    mean_delta_a = (float(episodes_df["mean_action_delta"].dropna().mean())
                    if "mean_action_delta" in episodes_df else np.nan)

    trunc_count = int(episodes_df["truncated"].sum())
    ball_count = int(episodes_df["ballistic_flag"].sum())
    osc_count = int(episodes_df["oscillatory_flag"].sum())
    entered_radius = int(episodes_df["steps_to_first_target_entry"].notna().sum())

    return {
        "n_total": n_total,
        "n_succ": n_succ,
        "p_hat": p_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "med_steps": med_steps,
        "mean_dist": mean_dist,
        "mean_delta_a": mean_delta_a,
        "trunc_rate": trunc_count / n_total if n_total > 0 else np.nan,
        "ball_rate": ball_count / n_total if n_total > 0 else np.nan,
        "osc_rate": osc_count / n_total if n_total > 0 else np.nan,
        "frac_entered": entered_radius / n_total if n_total > 0 else 0.0,
    }


def _format_row(m: dict, schema: str, **identity) -> dict:
    """Formatea las métricas canónicas según el esquema de claves histórico."""
    if schema == "b":
        row = {
            "checkpoint_dir": identity.get("checkpoint_dir"),
            "total_episodes": m["n_total"],
            "successes": m["n_succ"],
            "strict_success_rate": m["p_hat"],
            "ci_low": m["ci_low"],
            "ci_high": m["ci_high"],
            "truncation_rate": m["trunc_rate"],
            "ballistic_rate": m["ball_rate"],
            "oscillatory_rate": m["osc_rate"],
            "steps_to_success_median": m["med_steps"],
            "mean_final_distance": m["mean_dist"],
            "mean_action_delta": m["mean_delta_a"],
            "has_vecnormalize": identity.get("has_vecnormalize"),
        }
    elif schema == "c":
        row = {
            "n_episodes": m["n_total"],
            "success_rate": m["p_hat"],
            "ci_low": m["ci_low"],
            "ci_high": m["ci_high"],
            "median_steps_to_success": m["med_steps"],
            "mean_final_distance": m["mean_dist"],
            "mean_action_delta": m["mean_delta_a"],
            "truncation_rate": m["trunc_rate"],
            "ballistic_rate": m["ball_rate"],
            "oscillatory_rate": m["osc_rate"],
            "frac_entered_radius": m["frac_entered"],
        }
    else:
        raise ValueError(f"Esquema de claves desconocido: {schema!r} (valores: 'b', 'c')")
    return row


def evaluate_single_checkpoint(ckpt_dir: Path, val_dataset: list, env_cfg: dict,
                               dataset_name: str = "validation_b2",
                               eval_mode: str = "deterministic",
                               thresholds_path: str = "configs/phase_a/thresholds.yaml",
                               algo_hint: str = None,
                               schema: str = "b") -> tuple:
    """
    Carga un checkpoint (model.zip + vecnormalize_stats.pkl) y lo evalúa sobre
    el dataset de validación indicado.

    schema : "b" (Fase B/extensión) o "c" (Fase C); ver docstring del módulo.
    Devuelve ``(summary, episodes_df)``.
    """
    ckpt_dir = Path(ckpt_dir)
    model_path = _resolve_model_path(ckpt_dir)

    norm_path = ckpt_dir / "vecnormalize_stats.pkl"
    norm_path_str = str(norm_path) if norm_path.exists() else None

    model = load_agent_model(model_path, algo_hint)

    episodes_df, agg_metrics = evaluate_on_dataset(
        agent=model,
        dataset=val_dataset,
        dataset_name=dataset_name,
        eval_mode=eval_mode,
        thresholds_path=thresholds_path,
        env_config=env_cfg,
        vec_normalize_path=norm_path_str
    )

    summary = _format_row(_operational_metrics(episodes_df), schema=schema,
                          checkpoint_dir=str(ckpt_dir),
                          has_vecnormalize=norm_path.exists())
    return summary, episodes_df


def evaluate_c0_controller(val_dataset: list, env_cfg: dict, max_speed: float, label: str,
                           dataset_name: str = "validation_b2") -> tuple:
    """
    Evalúa el controlador C0 analítico sobre el dataset (compatibilidad
    extensión de Fase B: devuelve ``(summary, episodes_df)`` en esquema "b").
    """
    c0 = JacobianDampedController(kp=3.0, lambda_damping=0.01, max_speed=max_speed)
    episodes_df, _ = evaluate_on_dataset(
        agent=c0,
        dataset=val_dataset,
        dataset_name=dataset_name,
        eval_mode="deterministic",
        env_config=env_cfg
    )

    m = _operational_metrics(episodes_df)
    summary = _format_row(m, schema="b", checkpoint_dir=f"C0_{label}_analytic",
                          has_vecnormalize=False)
    summary["experiment"] = f"C0_{label}"
    summary["step"] = -1
    summary["seed"] = 0
    return summary, episodes_df


def evaluate_experiment_checkpoints(exp_dir: Path, cfg_path: Path, output_csv: Path = None) -> pd.DataFrame:
    """
    Evalúa todos los checkpoints ordenados por timesteps para una corrida (Fase B).
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f)

    from src.evaluation.datasets import save_and_get_datasets
    val_dataset, _ = save_and_get_datasets()

    ckpts_base = exp_dir / "checkpoints"
    if not ckpts_base.exists():
        raise FileNotFoundError(f"Directorio de checkpoints no existe: {ckpts_base}")

    # Listar carpetas ckpt_*
    ckpt_dirs = [p for p in ckpts_base.iterdir() if p.is_dir()]
    ckpt_dirs = sorted(ckpt_dirs, key=lambda p: extract_step_from_dirname(p.name))

    results = []
    print(f"\n>>> [EVAL CHECKPOINTS] Evaluando {len(ckpt_dirs)} checkpoints en {exp_dir}...")
    for c_dir in ckpt_dirs:
        step = extract_step_from_dirname(c_dir.name)
        summ, _ = evaluate_single_checkpoint(c_dir, val_dataset, env_cfg,
                                             dataset_name="validation", schema="b")
        summ["step"] = step
        summ["experiment"] = env_cfg.get("experiment", {}).get("name", exp_dir.parent.name)
        # extraer semilla de nombre de carpeta (ej. seed_42)
        m_seed = re.search(r"seed_(\d+)", str(exp_dir))
        summ["seed"] = int(m_seed.group(1)) if m_seed else 42
        results.append(summ)
        print(f"  Step {step:>7,}: Exito Estricto = {summ['strict_success_rate']*100:5.1f}% [{summ['ci_low']*100:5.1f}%, {summ['ci_high']*100:5.1f}%] | Pasos Mediana = {summ['steps_to_success_median']}")

    df = pd.DataFrame(results)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f">>> [OK] Resultados guardados en: {output_csv}")
    return df


# Agentes de la Fase C (constante sustituible para humos: S1 SAC y X3 PPO)
PHASE_C_TARGETS = [
    {"agent": "S1", "algo": "SAC", "base_dir": REPO_ROOT / "experiments" / "phase_c" / "S1"},
    {"agent": "X3", "algo": "PPO", "base_dir": REPO_ROOT / "experiments" / "phase_b" / "X3"},
]


def run_phase_c_evaluation(dataset_path: str = "data/datasets/phase_c/validation_c_500.json",
                           output_dir: str = "reports/data/phase_c",
                           max_step: int = 655360,
                           eval_mode: str = "deterministic"):
    val_file = REPO_ROOT / dataset_path
    if not val_file.exists():
        raise FileNotFoundError(f"Dataset de validación no encontrado: {val_file}")

    with open(val_file, "r", encoding="utf-8") as f:
        val_dataset = json.load(f)

    out_path = REPO_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    env_cfg = {
        "env": {
            "max_speed": 1.5,
            "success_mode": "hold",
            "hold_steps": 20,
            "velocity_tolerance": 0.10,
            "target_radius": 0.06
        }
    }

    targets = PHASE_C_TARGETS
    all_rows = []
    print(f"\n{'='*75}")
    print(f" EVALUACIÓN DE CHECKPOINTS EN VALIDACIÓN C ({eval_mode.upper()})")
    print(f" Dataset: {dataset_path} ({len(val_dataset)} episodios)")
    print(f"{'='*75}\n")

    for target in targets:
        agent_id = target["agent"]
        algo_hint = target["algo"]
        base_dir = target["base_dir"]

        if not base_dir.exists():
            print(f">>> [AVISO] Directorio no encontrado para {agent_id}: {base_dir}. Omitiendo.")
            continue

        seed_dirs = sorted([d for d in base_dir.glob("seed_*") if d.is_dir()])
        for sdir in seed_dirs:
            seed_str = sdir.name.replace("seed_", "")
            ckpt_dirs = sorted([d for d in (sdir / "checkpoints").glob("ckpt_*") if d.is_dir()],
                               key=lambda p: extract_step_from_dirname(p.name))

            for cdir in ckpt_dirs:
                step = extract_step_from_dirname(cdir.name)
                if step > max_step:
                    continue

                print(f" Evaluando: {agent_id} | Semilla: {seed_str} | Paso: {step:,}...", end="", flush=True)
                row, _ = evaluate_single_checkpoint(cdir, val_dataset, env_cfg,
                                                    dataset_name="validation_c",
                                                    eval_mode=eval_mode,
                                                    algo_hint=algo_hint,
                                                    schema="c")
                row["experiment"] = agent_id
                row["seed"] = int(seed_str) if seed_str.isdigit() else seed_str
                row["step"] = step
                row["eval_mode"] = eval_mode
                all_rows.append(row)
                print(f" -> Éxito: {row['success_rate']*100:.2f}% | Pasos: {row['median_steps_to_success']}")

    # Evaluar también C0 como referencia
    print("\n Evaluando referencia analítica C0 (v_max = 1.5)...", end="", flush=True)
    c0_env_cfg = {"env": {"max_speed": 1.5}}
    c0 = JacobianDampedController(kp=3.0, lambda_damping=0.01, max_speed=1.5)
    c0_episodes_df, _ = evaluate_on_dataset(
        agent=c0,
        dataset=val_dataset,
        dataset_name="validation_c",
        eval_mode="deterministic",
        thresholds_path="configs/phase_a/thresholds.yaml",
        env_config=c0_env_cfg,
        vec_normalize_path=None
    )
    c0_row = _format_row(_operational_metrics(c0_episodes_df), schema="c")
    c0_row["experiment"] = "C0_v1.5"
    c0_row["step"] = 655360
    c0_row["seed"] = "analytic"
    c0_row["eval_mode"] = eval_mode
    all_rows.append(c0_row)
    print(f" -> Éxito: {c0_row['success_rate']*100:.2f}%")

    # Guardar CSV detallado
    details_df = pd.DataFrame(all_rows)
    suffix = f"_{eval_mode}" if eval_mode != "deterministic" else ""
    details_csv_path = out_path / f"checkpoints_eval_validation_details{suffix}.csv"
    details_df.to_csv(details_csv_path, index=False)
    print(f"\n>>> [CSV] Resultados detallados guardados en: {details_csv_path}")

    # Guardar CSV agregado (media +- std muestral ddof=1)
    rl_df = details_df[details_df["seed"] != "analytic"]
    if len(rl_df) > 0:
        agg_records = []
        for (exp_name, step_val), group in rl_df.groupby(["experiment", "step"]):
            n_seeds = len(group)
            m_succ = float(group["success_rate"].mean())
            s_succ = float(group["success_rate"].std(ddof=1)) if n_seeds > 1 else np.nan
            m_steps = float(group["median_steps_to_success"].mean())
            s_steps = float(group["median_steps_to_success"].std(ddof=1)) if n_seeds > 1 else np.nan
            m_dist = float(group["mean_final_distance"].mean())
            m_delta_a = float(group["mean_action_delta"].mean())
            s_delta_a = float(group["mean_action_delta"].std(ddof=1)) if n_seeds > 1 else np.nan
            m_trunc = float(group["truncation_rate"].mean())
            m_ball = float(group["ballistic_rate"].mean())
            m_entered = float(group["frac_entered_radius"].mean())

            agg_records.append({
                "experiment": exp_name,
                "step": step_val,
                "n_seeds": n_seeds,
                "mean_success_rate": m_succ,
                "std_success_rate": s_succ,
                "mean_steps_median": m_steps,
                "std_steps_median": s_steps,
                "mean_final_distance": m_dist,
                "mean_action_delta": m_delta_a,
                "std_action_delta": s_delta_a,
                "mean_truncation_rate": m_trunc,
                "mean_ballistic_rate": m_ball,
                "mean_frac_entered_radius": m_entered,
                "eval_mode": eval_mode
            })

        agg_df = pd.DataFrame(agg_records)
        agg_csv_path = out_path / f"checkpoints_eval_validation_aggregated{suffix}.csv"
        agg_df.to_csv(agg_csv_path, index=False)
        print(f">>> [CSV] Resultados agregados guardados en: {agg_csv_path}")

    return details_df
