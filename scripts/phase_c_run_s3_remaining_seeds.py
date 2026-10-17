"""
Script de Replicación de S3_combined en Semillas 100 y 2026:
- Entrena semillas 100 y 2026 bajo configs/phase_c/S3_combined.yaml (245.760 timesteps).
- Evalúa deterministamente los 6 checkpoints por semilla sobre validation_c_500.json.
- Extrae telemetría de TensorBoard (ent_coef / alpha).
- Consolida las 3 semillas (42, 100, 2026) en CSVs detallados y agregados.
- Genera reports/phase_c_s3_analysis/tabla_comparativa_3seeds_245k.csv.
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
DATASET_PATH = REPO_ROOT / "data" / "datasets" / "phase_c" / "validation_c_500.json"
THRESHOLDS_PATH = REPO_ROOT / "configs" / "phase_a" / "thresholds.yaml"
OUTPUT_DIR = REPO_ROOT / "reports" / "phase_c_s3_analysis"
EXPERIMENT_BASE = REPO_ROOT / "experiments" / "phase_c"

SEEDS_REMAINING = [100, 2026]
TOTAL_TIMESTEPS = 245760
CHECKPOINT_FREQ = 40960
CHECKPOINT_STEPS = [40960, 81920, 122880, 163840, 204800, 245760]

def extract_alpha_trajectory(logs_dir: Path):
    """Extrae la trayectoria de ent_coef (alpha) desde los eventos de TensorBoard."""
    event_files = list(logs_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return {}
    event_files.sort(key=lambda p: p.stat().st_mtime)
    ea = EventAccumulator(str(event_files[-1]))
    ea.Reload()
    alpha_by_step = {}
    if "train/ent_coef" in ea.Tags().get("scalars", []):
        events = ea.Scalars("train/ent_coef")
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
    print(" INICIO DE REPLICACIÓN: S3_combined EN SEMILLAS 100 Y 2026")
    print(f" Presupuesto: {TOTAL_TIMESTEPS:,} pasos por semilla | Checkpoints cada: {CHECKPOINT_FREQ:,}")
    print("=" * 85)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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

    # Cargar timing previo de semilla 42
    timing_file = OUTPUT_DIR / "timing_s3_combined.json"
    if timing_file.exists():
        with open(timing_file, "r", encoding="utf-8") as f:
            timing_master = json.load(f)
    else:
        timing_master = {}

    seeds_timing = timing_master.get("seeds", {})
    if "seed" in timing_master and timing_master.get("seed") == 42:
        # Migrar estructura de semilla única a multi-semilla
        seeds_timing["42"] = {
            "training": timing_master.get("training", {}),
            "evaluation": timing_master.get("evaluation", {}),
            "total_workflow": timing_master.get("total_workflow", {})
        }

    new_rows = []

    for seed in SEEDS_REMAINING:
        print(f"\n{'='*85}")
        print(f" PROCESANDO SEMILLA {seed} DE S3_combined")
        print(f"{'='*85}")

        # 1. Entrenamiento
        t_tr_start = time.perf_counter()
        train_metrics = train_phase_c_sac_agent(
            config_path_or_dict=CONFIG_PATH,
            seed=seed,
            total_timesteps_override=TOTAL_TIMESTEPS,
            checkpoint_freq_override=CHECKPOINT_FREQ,
            output_base_dir=str(EXPERIMENT_BASE)
        )
        t_tr_elapsed = time.perf_counter() - t_tr_start
        fps_tr = TOTAL_TIMESTEPS / t_tr_elapsed if t_tr_elapsed > 0 else 0.0
        print(f">>> Semilla {seed}: Entrenamiento finalizado en {t_tr_elapsed:.2f} s ({t_tr_elapsed/60:.2f} min) | FPS: {fps_tr:.1f}")

        # 2. Evaluación determinista estricta (6 checkpoints)
        print(f"\n>>> Evaluando 6 checkpoints de Semilla {seed} sobre validation_c_500.json...")
        logs_dir = EXPERIMENT_BASE / "S3_combined" / f"seed_{seed}" / "logs"
        alpha_history = extract_alpha_trajectory(logs_dir)

        t_ev_start = time.perf_counter()
        ckpt_eval_times = {}

        seed_ckpt_dir = EXPERIMENT_BASE / "S3_combined" / f"seed_{seed}" / "checkpoints"
        for step in CHECKPOINT_STEPS:
            ckpt_dir = seed_ckpt_dir / f"ckpt_{step}"
            model_zip = ckpt_dir / "model.zip"
            norm_pkl = ckpt_dir / "vecnormalize_stats.pkl"

            if not model_zip.exists():
                raise FileNotFoundError(f"No existe {model_zip}")

            print(f"  Evaluando: S3_combined | Semilla {seed} | Paso {step:,}...", end="", flush=True)
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
            ckpt_eval_times[step] = t_ckpt_eval

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
            alpha_val = get_closest_alpha(alpha_history, step)

            new_rows.append({
                "model": "SAC S3_combined",
                "seed": seed,
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

        t_ev_elapsed = time.perf_counter() - t_ev_start
        t_wf_elapsed = t_tr_elapsed + t_ev_elapsed

        seeds_timing[str(seed)] = {
            "training": {
                "elapsed_seconds": t_tr_elapsed,
                "elapsed_minutes": t_tr_elapsed / 60.0,
                "fps": fps_tr
            },
            "evaluation": {
                "elapsed_seconds": t_ev_elapsed,
                "elapsed_minutes": t_ev_elapsed / 60.0,
                "per_checkpoint_seconds": ckpt_eval_times
            },
            "total_workflow": {
                "elapsed_seconds": t_wf_elapsed,
                "elapsed_minutes": t_wf_elapsed / 60.0
            }
        }

    # ------------------------------------------------------------------
    # FASE 3: CONSOLIDACIÓN DE LAS 3 SEMILLAS
    # ------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" CONSOLIDANDO MÉTRICAS DE LAS 3 SEMILLAS (42, 100, 2026)")
    print("=" * 85)

    s42_csv_path = OUTPUT_DIR / "metricas_checkpoints_s3_combined_seed42.csv"
    df_s42 = pd.read_csv(s42_csv_path)
    df_new = pd.DataFrame(new_rows)
    df_all_seeds = pd.concat([df_s42, df_new], ignore_index=True)

    # Ordenar por semilla y paso
    df_all_seeds.sort_values(by=["seed", "step"], inplace=True)
    all_seeds_csv_path = OUTPUT_DIR / "metricas_checkpoints_s3_combined_all_seeds.csv"
    df_all_seeds.to_csv(all_seeds_csv_path, index=False)
    print(f">>> [CSV] Todas las semillas guardadas en: {all_seeds_csv_path}")

    # Tabla agregada (media +- std con ddof=1)
    agg_rows = []
    for step in CHECKPOINT_STEPS:
        sub = df_all_seeds[df_all_seeds["step"] == step]
        n_seeds = len(sub)
        assert n_seeds == 3, f"Se esperaban 3 semillas para el paso {step}, se obtuvieron {n_seeds}"

        succ_m, succ_s = sub["success_rate"].mean(), sub["success_rate"].std(ddof=1)
        ne_m, ne_s = sub["never_entered_rate"].mean(), sub["never_entered_rate"].std(ddof=1)
        enh_m, enh_s = sub["entered_no_hold_rate"].mean(), sub["entered_no_hold_rate"].std(ddof=1)
        da_m, da_s = sub["action_delta_mean"].mean(), sub["action_delta_mean"].std(ddof=1)
        ex_m, ex_s = sub["n_exits_mean"].mean(), sub["n_exits_mean"].std(ddof=1)
        med_m = sub["median_steps_to_success"].mean()
        dist_m = sub["final_distance_mean"].mean()
        alpha_m = sub["alpha_ent_coef"].mean()

        agg_rows.append({
            "model": "SAC S3_combined",
            "step": step,
            "n_seeds": n_seeds,
            "success_rate_mean": succ_m,
            "success_rate_std": succ_s,
            "success_rate_min": sub["success_rate"].min(),
            "success_rate_max": sub["success_rate"].max(),
            "never_entered_rate_mean": ne_m,
            "never_entered_rate_std": ne_s,
            "entered_no_hold_rate_mean": enh_m,
            "entered_no_hold_rate_std": enh_s,
            "action_delta_mean_mean": da_m,
            "action_delta_mean_std": da_s,
            "n_exits_mean_mean": ex_m,
            "n_exits_mean_std": ex_s,
            "median_steps_to_success_mean": med_m,
            "final_distance_mean_mean": dist_m,
            "alpha_ent_coef_mean": alpha_m
        })

    df_agg = pd.DataFrame(agg_rows)
    agg_csv_path = OUTPUT_DIR / "metricas_checkpoints_s3_combined_agg.csv"
    df_agg.to_csv(agg_csv_path, index=False)
    print(f">>> [CSV] Metricas agregadas guardadas en: {agg_csv_path}")

    # Actualizar timing JSON
    tot_tr_sec = sum(s["training"]["elapsed_seconds"] for s in seeds_timing.values())
    tot_ev_sec = sum(s["evaluation"]["elapsed_seconds"] for s in seeds_timing.values())
    tot_wf_sec = tot_tr_sec + tot_ev_sec
    mean_fps_all = (TOTAL_TIMESTEPS * 3) / tot_tr_sec if tot_tr_sec > 0 else 0.0

    updated_timing = {
        "model": "SAC S3_combined",
        "total_timesteps_per_seed": TOTAL_TIMESTEPS,
        "seeds": seeds_timing,
        "aggregate_timing": {
            "total_training_seconds": tot_tr_sec,
            "total_training_minutes": tot_tr_sec / 60.0,
            "mean_fps": mean_fps_all,
            "total_evaluation_seconds": tot_ev_sec,
            "total_evaluation_minutes": tot_ev_sec / 60.0,
            "total_workflow_seconds": tot_wf_sec,
            "total_workflow_minutes": tot_wf_sec / 60.0
        }
    }
    with open(timing_file, "w", encoding="utf-8") as f:
        json.dump(updated_timing, f, indent=4)
    print(f">>> [JSON] Timing consolidado guardado en: {timing_file}")

    # ------------------------------------------------------------------
    # FASE 4: TABLA COMPARATIVA A 245 760 PASOS (3 SEMILLAS)
    # ------------------------------------------------------------------
    master_csv_path = REPO_ROOT / "reports" / "phase_c_s2_analysis" / "tabla_comparativa_245k_final.csv"
    df_master = pd.read_csv(master_csv_path)

    # Filtrar paso 245760 y calcular medias agregadas para los modelos existentes
    comp_models = ["PPO X3", "SAC S1", "SAC S2_freq", "SAC S2_wa"]
    comp_summary_rows = []

    for m in comp_models:
        sub = df_master[(df_master["model"] == m) & (df_master["step"] == 245760) & (df_master["seed"].astype(str).isin(["42", "100", "2026"]))]
        comp_summary_rows.append({
            "model": m,
            "success_rate_mean": sub["success_rate"].mean(),
            "success_rate_std": sub["success_rate"].std(ddof=1),
            "never_entered_mean": sub["never_entered_rate"].mean(),
            "never_entered_std": sub["never_entered_rate"].std(ddof=1),
            "entered_no_hold_mean": sub["entered_no_hold_rate"].mean(),
            "entered_no_hold_std": sub["entered_no_hold_rate"].std(ddof=1),
            "action_delta_mean": sub["action_delta_mean"].mean(),
            "action_delta_std": sub["action_delta_mean"].std(ddof=1),
            "n_exits_mean": sub["n_exits_mean"].mean(),
            "n_exits_std": sub["n_exits_mean"].std(ddof=1)
        })

    # Añadir S3_combined en paso 245760
    s3_245k = df_agg[df_agg["step"] == 245760].iloc[0]
    comp_summary_rows.append({
        "model": "SAC S3_combined",
        "success_rate_mean": s3_245k["success_rate_mean"],
        "success_rate_std": s3_245k["success_rate_std"],
        "never_entered_mean": s3_245k["never_entered_rate_mean"],
        "never_entered_std": s3_245k["never_entered_rate_std"],
        "entered_no_hold_mean": s3_245k["entered_no_hold_rate_mean"],
        "entered_no_hold_std": s3_245k["entered_no_hold_rate_std"],
        "action_delta_mean": s3_245k["action_delta_mean_mean"],
        "action_delta_std": s3_245k["action_delta_mean_std"],
        "n_exits_mean": s3_245k["n_exits_mean_mean"],
        "n_exits_std": s3_245k["n_exits_mean_std"]
    })

    df_comp_final = pd.DataFrame(comp_summary_rows)
    comp_3seeds_path = OUTPUT_DIR / "tabla_comparativa_3seeds_245k.csv"
    df_comp_final.to_csv(comp_3seeds_path, index=False)
    print(f">>> [CSV] Comparativa 3 semillas guardada en: {comp_3seeds_path}")

    print("\n" + "=" * 85)
    print(" RESUMEN AGREGADO S3_combined (PASO 245 760, 3 SEMILLAS):")
    print(f" Éxito estricto: {s3_245k['success_rate_mean']*100:.2f}% +- {s3_245k['success_rate_std']*100:.2f}%")
    print(f" Never entered:  {s3_245k['never_entered_rate_mean']*100:.2f}% +- {s3_245k['never_entered_rate_std']*100:.2f}%")
    print(f" Entered no hold:{s3_245k['entered_no_hold_rate_mean']*100:.2f}% +- {s3_245k['entered_no_hold_rate_std']*100:.2f}%")
    print(f" ||delta_a||:    {s3_245k['action_delta_mean_mean']:.4f} +- {s3_245k['action_delta_mean_std']:.4f}")
    print(f" Salidas meta:   {s3_245k['n_exits_mean_mean']:.3f} +- {s3_245k['n_exits_mean_std']:.3f}")
    print("=" * 85)

if __name__ == "__main__":
    main()
