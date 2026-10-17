"""
Ejecución de la Tanda 1 (S2_freq):
- Variante unifactorial: train_freq = 4 (único cambio respecto de S1).
- 3 Semillas: 42, 100, 2026.
- Presupuesto: 245.760 timesteps por semilla (checkpoints cada 40.960 timesteps).
- Evaluación determinista estricta sobre validation_c_500.json de los 18 checkpoints.
- Exporta métricas detalladas y agregadas a reports/phase_c_s2_analysis/.
"""
import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_c_train_sac import train_phase_c_sac_agent
from src.evaluation.evaluator import evaluate_on_dataset
from src.evaluation.metrics import wilson_score_interval

CONFIG_PATH = REPO_ROOT / "configs" / "phase_c" / "S2_freq.yaml"
DATASET_PATH = REPO_ROOT / "data" / "datasets" / "phase_c" / "validation_c_500.json"
THRESHOLDS_PATH = REPO_ROOT / "configs" / "phase_a" / "thresholds.yaml"
OUTPUT_DIR = REPO_ROOT / "reports" / "phase_c_s2_analysis"
EXPERIMENT_BASE = REPO_ROOT / "experiments" / "phase_c"

SEEDS = [42, 100, 2026]
TOTAL_TIMESTEPS = 245760
CHECKPOINT_FREQ = 40960
CHECKPOINT_STEPS = [40960, 81920, 122880, 163840, 204800, 245760]

def main():
    print("=" * 85)
    print(" INICIO DE EJECUCIÓN: ETAPA C-3 — TANDA 1 (S2_freq, train_freq=4)")
    print(f" Presupuesto: {TOTAL_TIMESTEPS:,} pasos por semilla | 3 semillas: {SEEDS}")
    print("=" * 85)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_global_start = time.perf_counter()
    training_times = {}

    # ------------------------------------------------------------------
    # FASE 1: ENTRENAMIENTO DE LAS 3 SEMILLAS
    # ------------------------------------------------------------------
    for seed in SEEDS:
        print(f"\n>>> Entrenando S2_freq | Semilla {seed}...")
        t_s = time.perf_counter()
        metrics = train_phase_c_sac_agent(
            config_path_or_dict=CONFIG_PATH,
            seed=seed,
            total_timesteps_override=TOTAL_TIMESTEPS,
            checkpoint_freq_override=CHECKPOINT_FREQ,
            output_base_dir=str(EXPERIMENT_BASE)
        )
        elapsed_s = time.perf_counter() - t_s
        fps_s = TOTAL_TIMESTEPS / elapsed_s if elapsed_s > 0 else 0
        training_times[seed] = {
            "elapsed_sec": elapsed_s,
            "elapsed_min": elapsed_s / 60.0,
            "fps": fps_s
        }
        print(f">>> Semilla {seed} completada en {elapsed_s:.1f} s ({elapsed_s/60:.2f} min) | FPS: {fps_s:.1f}")

    t_train_total = time.perf_counter() - t_global_start
    print(f"\n[OK] Entrenamiento de las 3 semillas finalizado en {t_train_total/60:.2f} min")

    # ------------------------------------------------------------------
    # FASE 2: EVALUACIÓN DETERMINISTA ESTRICTA (18 CHECKPOINTS)
    # ------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" EVALUACIÓN DETERMINISTA ESTRICTA SOBRE validation_c_500.json (18 CHECKPOINTS)")
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

    t_eval_start = time.perf_counter()
    detailed_rows = []

    for seed in SEEDS:
        seed_ckpt_dir = EXPERIMENT_BASE / "S2_freq" / f"seed_{seed}" / "checkpoints"
        for step in CHECKPOINT_STEPS:
            ckpt_dir = seed_ckpt_dir / f"ckpt_{step}"
            model_zip = ckpt_dir / "model.zip"
            norm_pkl = ckpt_dir / "vecnormalize_stats.pkl"

            if not model_zip.exists():
                raise FileNotFoundError(f"No existe {model_zip}")

            print(f"  Evaluando: S2_freq | Semilla {seed} | Paso {step:,}...", end="", flush=True)
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

            detailed_rows.append({
                "experiment": "S2_freq",
                "seed": seed,
                "step": step,
                "n_episodes": n_total,
                "success_rate": succ_rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "never_entered_rate": never_entered_rate,
                "entered_no_hold_rate": entered_no_hold_rate,
                "mean_action_delta": mean_delta_a,
                "median_steps_to_success": med_steps,
                "mean_final_distance": mean_dist,
                "n_exits_mean": n_exits_mean,
                "truncation_rate": float(episodes_df["truncated"].mean()),
                "eval_time_sec": t_ckpt_eval
            })

            print(f" -> Éxito: {succ_rate*100:.2f}% | NE: {never_entered_rate*100:.2f}% | ENH: {entered_no_hold_rate*100:.2f}% | ||Δa||: {mean_delta_a:.4f} ({t_ckpt_eval:.1f} s)")

    t_eval_total = time.perf_counter() - t_eval_start
    t_global_total = time.perf_counter() - t_global_start

    # Guardar CSV detallado
    df_details = pd.DataFrame(detailed_rows)
    details_csv_path = OUTPUT_DIR / "metricas_checkpoints_s2_freq.csv"
    df_details.to_csv(details_csv_path, index=False)
    print(f"\n>>> [CSV] Detalle guardado en: {details_csv_path}")

    # Calcular y guardar CSV agregado (media +- std muestral ddof=1)
    agg_rows = []
    for step in CHECKPOINT_STEPS:
        sub = df_details[df_details["step"] == step]
        n_seeds = len(sub)
        succ_m, succ_s = sub["success_rate"].mean(), sub["success_rate"].std(ddof=1)
        ne_m, ne_s = sub["never_entered_rate"].mean(), sub["never_entered_rate"].std(ddof=1)
        enh_m, enh_s = sub["entered_no_hold_rate"].mean(), sub["entered_no_hold_rate"].std(ddof=1)
        act_m, act_s = sub["mean_action_delta"].mean(), sub["mean_action_delta"].std(ddof=1)
        step_m = sub["median_steps_to_success"].mean()
        dist_m = sub["mean_final_distance"].mean()
        exits_m = sub["n_exits_mean"].mean()

        agg_rows.append({
            "experiment": "S2_freq",
            "step": step,
            "n_seeds": n_seeds,
            "mean_success_rate": succ_m,
            "std_success_rate": succ_s,
            "mean_never_entered_rate": ne_m,
            "std_never_entered_rate": ne_s,
            "mean_entered_no_hold_rate": enh_m,
            "std_entered_no_hold_rate": enh_s,
            "mean_action_delta": act_m,
            "std_action_delta": act_s,
            "mean_median_steps_to_success": step_m,
            "mean_final_distance": dist_m,
            "mean_n_exits": exits_m
        })

    df_agg = pd.DataFrame(agg_rows)
    agg_csv_path = OUTPUT_DIR / "metricas_checkpoints_s2_freq_agg.csv"
    df_agg.to_csv(agg_csv_path, index=False)
    print(f">>> [CSV] Agregado guardado en: {agg_csv_path}")

    # Guardar métricas de tiempo
    timing_meta = {
        "experiment": "S2_freq",
        "seeds": SEEDS,
        "total_timesteps_per_seed": TOTAL_TIMESTEPS,
        "training_times_per_seed": training_times,
        "training_total_sec": t_train_total,
        "training_total_min": t_train_total / 60.0,
        "eval_total_sec": t_eval_total,
        "eval_total_min": t_eval_total / 60.0,
        "wall_clock_total_sec": t_global_total,
        "wall_clock_total_min": t_global_total / 60.0
    }
    with open(OUTPUT_DIR / "timing_s2_freq.json", "w", encoding="utf-8") as f:
        json.dump(timing_meta, f, indent=4)

    print("\n" + "=" * 85)
    print(" RESUMEN DE TIEMPOS DE EJECUCIÓN (TANDA 1 - S2_freq)")
    print(f" Entrenamiento 3 semillas: {t_train_total/60:.2f} min ({t_train_total:.1f} s)")
    for s, t_info in training_times.items():
        print(f"   Semilla {s:>4}: {t_info['elapsed_min']:.2f} min ({t_info['fps']:.1f} FPS)")
    print(f" Evaluación 18 checkpoints: {t_eval_total/60:.2f} min ({t_eval_total:.1f} s)")
    print(f" TIEMPO TOTAL WALL-CLOCK: {t_global_total/60:.2f} min ({t_global_total:.1f} s)")
    print("=" * 85)

if __name__ == "__main__":
    main()
