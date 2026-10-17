"""
Script de Ejecución del Piloto Combinado S3_combined (Semilla 42):
- Configuración: configs/phase_c/S3_combined.yaml
- Presupuesto: 245.760 timesteps (checkpoints cada 40.960 timesteps).
- Evaluación determinista estricta sobre validation_c_500.json (6 checkpoints).
- Extracción de telemetría de TensorBoard (ent_coef / alpha).
- Registro de tiempos de entrenamiento y evaluación.
- Generación de reportes en reports/phase_c_s3_analysis/.
"""
import os
import sys
import time
import json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_c_train_sac import train_phase_c_sac_agent
from src.evaluation.evaluator import evaluate_on_dataset
from src.evaluation.metrics import wilson_score_interval

CONFIG_PATH = REPO_ROOT / "configs" / "phase_c" / "S3_combined.yaml"
RULE_PATH = REPO_ROOT / "configs" / "phase_c" / "selection_rule_s3_combined.yaml"
DATASET_PATH = REPO_ROOT / "data" / "datasets" / "phase_c" / "validation_c_500.json"
THRESHOLDS_PATH = REPO_ROOT / "configs" / "phase_a" / "thresholds.yaml"
OUTPUT_DIR = REPO_ROOT / "reports" / "phase_c_s3_analysis"
EXPERIMENT_BASE = REPO_ROOT / "experiments" / "phase_c"

SEED = 42
TOTAL_TIMESTEPS = 245760
CHECKPOINT_FREQ = 40960
CHECKPOINT_STEPS = [40960, 81920, 122880, 163840, 204800, 245760]

def extract_alpha_trajectory(logs_dir: Path):
    """Extrae la trayectoria de ent_coef (alpha) desde los eventos de TensorBoard."""
    event_files = list(logs_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return {}
    
    # Tomar el archivo de eventos más reciente
    event_files.sort(key=lambda p: p.stat().st_mtime)
    ea = EventAccumulator(str(event_files[-1]))
    ea.Reload()
    
    alpha_by_step = {}
    if "train/ent_coef" in ea.Tags().get("scalars", []):
        events = ea.Scalars("train/ent_coef")
        # events tiene elementos con (wall_time, step, value)
        for ev in events:
            alpha_by_step[ev.step] = ev.value
            
    return alpha_by_step

def get_closest_alpha(alpha_by_step: dict, target_step: int):
    """Encuentra el valor de alpha en el paso más próximo a target_step."""
    if not alpha_by_step:
        return np.nan
    steps = np.array(list(alpha_by_step.keys()))
    idx = np.argmin(np.abs(steps - target_step))
    closest_step = steps[idx]
    return float(alpha_by_step[closest_step])

def main():
    print("=" * 85)
    print(" INICIO DE EJECUCIÓN: PILOTO COMBINADO S3_combined (Semilla 42)")
    print(f" Presupuesto: {TOTAL_TIMESTEPS:,} pasos | Checkpoints cada: {CHECKPOINT_FREQ:,} pasos")
    print("=" * 85)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_global_start = time.perf_counter()

    # ------------------------------------------------------------------
    # FASE 1: ENTRENAMIENTO DE LA SEMILLA 42
    # ------------------------------------------------------------------
    print(f"\n>>> Entrenando S3_combined | Semilla {SEED}...")
    t_train_start = time.perf_counter()
    train_metrics = train_phase_c_sac_agent(
        config_path_or_dict=CONFIG_PATH,
        seed=SEED,
        total_timesteps_override=TOTAL_TIMESTEPS,
        checkpoint_freq_override=CHECKPOINT_FREQ,
        output_base_dir=str(EXPERIMENT_BASE)
    )
    t_train_elapsed = time.perf_counter() - t_train_start
    train_fps = TOTAL_TIMESTEPS / t_train_elapsed if t_train_elapsed > 0 else 0.0
    print(f">>> Entrenamiento completado en {t_train_elapsed:.2f} s ({t_train_elapsed/60:.2f} min) | FPS: {train_fps:.1f}")

    # ------------------------------------------------------------------
    # FASE 2: EVALUACIÓN DETERMINISTA ESTRICTA (6 CHECKPOINTS)
    # ------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" EVALUACIÓN DETERMINISTA ESTRICTA SOBRE validation_c_500.json (6 CHECKPOINTS)")
    print("=" * 85)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        val_dataset = json.load(f)

    env_cfg = {
        "env": {
            "max_speed": 1.5,
            "success_mode": "hold",
            "hold_steps": 20,
            "velocity_tolerance": 0.10,
            "target_radius": 0.06
        }
    }

    logs_dir = EXPERIMENT_BASE / "S3_combined" / f"seed_{SEED}" / "logs"
    alpha_history = extract_alpha_trajectory(logs_dir)

    t_eval_start = time.perf_counter()
    detailed_rows = []
    eval_times_per_ckpt = {}

    seed_ckpt_dir = EXPERIMENT_BASE / "S3_combined" / f"seed_{SEED}" / "checkpoints"
    for step in CHECKPOINT_STEPS:
        ckpt_dir = seed_ckpt_dir / f"ckpt_{step}"
        model_zip = ckpt_dir / "model.zip"
        norm_pkl = ckpt_dir / "vecnormalize_stats.pkl"

        if not model_zip.exists():
            raise FileNotFoundError(f"No existe {model_zip}")

        print(f"  Evaluando: S3_combined | Semilla {SEED} | Paso {step:,}...", end="", flush=True)
        t_ckpt_start = time.perf_counter()

        agent = SAC.load(str(model_zip))
        norm_str = str(norm_pkl) if norm_pkl.exists() else None

        episodes_df, _ = evaluate_on_dataset(
            agent=agent,
            dataset=val_dataset,
            dataset_name="validation_c",
            eval_mode="deterministic",
            thresholds_path=str(THRESHOLDS_PATH),
            env_config=env_cfg,
            vec_normalize_path=norm_str
        )
        t_ckpt_eval = time.perf_counter() - t_ckpt_start
        eval_times_per_ckpt[step] = t_ckpt_eval

        n_total = len(episodes_df)
        n_succ = int(episodes_df["success"].sum())
        succ_rate = n_succ / n_total
        p_hat, ci_low, ci_high = wilson_score_interval(n_succ, n_total)

        entered_series = episodes_df["steps_to_first_target_entry"]
        never_entered_rate = float((entered_series.isna()).mean())
        entered_no_hold_rate = float((entered_series.notna() & ~episodes_df["success"]).mean())

        mean_delta_a = float(episodes_df["mean_action_delta"].dropna().mean()) if "mean_action_delta" in episodes_df else np.nan
        succ_df = episodes_df[episodes_df["success"]]
        med_steps = float(succ_df["steps"].median()) if len(succ_df) > 0 else np.nan
        mean_dist = float(episodes_df["final_distance"].mean())
        n_exits_mean = float(episodes_df["n_exits_after_first_entry"].mean()) if "n_exits_after_first_entry" in episodes_df else np.nan
        
        # Alpha en este paso
        alpha_val = get_closest_alpha(alpha_history, step)

        detailed_rows.append({
            "model": "SAC S3_combined",
            "seed": SEED,
            "step": step,
            "n_episodes": n_total,
            "success_rate": succ_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "never_entered_rate": never_entered_rate,
            "entered_no_hold_rate": entered_no_hold_rate,
            "action_delta_mean": mean_delta_a,
            "median_steps_to_success": med_steps,
            "final_distance_mean": mean_dist,
            "n_exits_mean": n_exits_mean,
            "alpha_ent_coef": alpha_val,
            "truncation_rate": float(episodes_df["truncated"].mean()),
            "eval_time_sec": t_ckpt_eval
        })

        print(f" -> Exito: {succ_rate*100:.2f}% | NE: {never_entered_rate*100:.2f}% | ENH: {entered_no_hold_rate*100:.2f}% | ||delta_a||: {mean_delta_a:.4f} | alpha: {alpha_val:.4f} ({t_ckpt_eval:.1f} s)")

    t_eval_total = time.perf_counter() - t_eval_start
    t_global_total = time.perf_counter() - t_global_start

    # Guardar CSV de detalle de checkpoints de S3_combined
    df_s3 = pd.DataFrame(detailed_rows)
    s3_csv_path = OUTPUT_DIR / "metricas_checkpoints_s3_combined_seed42.csv"
    df_s3.to_csv(s3_csv_path, index=False)
    print(f"\n>>> [CSV] Metricas guardadas en: {s3_csv_path}")

    # Guardar Timing JSON
    timing_data = {
        "model": "SAC S3_combined",
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "training": {
            "elapsed_seconds": t_train_elapsed,
            "elapsed_minutes": t_train_elapsed / 60.0,
            "fps": train_fps
        },
        "evaluation": {
            "elapsed_seconds": t_eval_total,
            "elapsed_minutes": t_eval_total / 60.0,
            "per_checkpoint_seconds": eval_times_per_ckpt
        },
        "total_workflow": {
            "elapsed_seconds": t_global_total,
            "elapsed_minutes": t_global_total / 60.0
        }
    }
    timing_path = OUTPUT_DIR / "timing_s3_combined.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_data, f, indent=4)
    print(f">>> [JSON] Tiempos guardados en: {timing_path}")

    # ------------------------------------------------------------------
    # FASE 3: TABLA COMPARATIVA CONSOLIDADA (SEMILLA 42)
    # ------------------------------------------------------------------
    master_csv_path = REPO_ROOT / "reports" / "phase_c_s2_analysis" / "tabla_comparativa_245k_final.csv"
    df_master = pd.read_csv(master_csv_path)
    s42_master = df_master[df_master["seed"].astype(str) == "42"].copy()
    
    # Preparar df_s3 con las mismas columnas
    cols_common = ["model", "seed", "step", "success_rate", "never_entered_rate", "entered_no_hold_rate", "action_delta_mean", "n_exits_mean", "median_steps_to_success", "final_distance_mean"]
    df_s3_sub = df_s3[cols_common].copy()
    
    df_comp_seed42 = pd.concat([s42_master[cols_common], df_s3_sub], ignore_index=True)
    comp_csv_path = OUTPUT_DIR / "tabla_comparativa_seed42.csv"
    df_comp_seed42.to_csv(comp_csv_path, index=False)
    print(f">>> [CSV] Tabla comparativa semilla 42 guardada en: {comp_csv_path}")

    # ------------------------------------------------------------------
    # FASE 4: APLICACIÓN FORMAL DE LA REGLA PREREGISTRADA
    # ------------------------------------------------------------------
    s3_final_row = df_s3[df_s3["step"] == 245760].iloc[0]
    final_success = float(s3_final_row["success_rate"])

    with open(RULE_PATH, "r", encoding="utf-8") as f:
        rule_cfg = yaml.safe_load(f)

    th_A = float(rule_cfg["decision_rule_thresholds"]["threshold_A_replicate"])
    th_C = float(rule_cfg["decision_rule_thresholds"]["threshold_C_close"])

    print("\n" + "=" * 85)
    print(" EVALUACIÓN DE LA REGLA PREREGISTRADA (PASO 245,760, SEMILLA 42)")
    print("=" * 85)
    print(f" Exito final S3_combined (seed 42): {final_success*100:.2f}% ({int(final_success*500)}/500)")
    print(f" Umbral A (Replicar): >= {th_A*100:.2f}%")
    print(f" Umbral C (Cerrar):   <  {th_C*100:.2f}%")
    print(f" Zona B (Ambiguo):    [{th_C*100:.2f}%, {th_A*100:.2f}%)")

    if final_success >= th_A:
        decision = "A (Replicar a semillas 100 y 2026)"
        desc = "El agente S3_combined supera el umbral A de no inferioridad estadística frente a PPO X3."
    elif final_success < th_C:
        decision = "C (Cierre definitivo de exploracion SAC con estas palancas)"
        desc = "El agente S3_combined no supera a la mejor variante previa (S2_freq) por encima del ruido muestral."
    else:
        decision = "B (Ambiguo / Zona Gris)"
        desc = "El resultado se ubica en el intervalo intermedio; requiere decision explícita del usuario."

    print(f"\n >>> VEREDICTO DE LA REGLA: RESULTADO {decision}")
    print(f" >>> Detalle: {desc}")
    print("=" * 85)

if __name__ == "__main__":
    main()
