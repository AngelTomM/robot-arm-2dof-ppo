"""
Evaluación de Checkpoints sobre el Conjunto de VALIDACIÓN (Fase B):
- Procesa los 150 checkpoints (25 checkpoints * 3 semillas * 2 experimentos: X5_near y X3).
- Evalúa estrictamente sobre data/datasets/validation_set_200.json en modo DETERMINISTA.
- NUNCA lee el conjunto final (final_eval_set_300).
- Guarda resultados individuales y agregados en reports/data/phase_b/.
"""
import sys
import time
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.checkpoints import evaluate_experiment_checkpoints

def run_batch_evaluation():
    exps = [
        ("X5_near", REPO_ROOT / "configs" / "phase_b" / "X5_near.yaml"),
        ("X3", REPO_ROOT / "configs" / "phase_b" / "X3.yaml")
    ]
    seeds = [42, 100, 2026]
    
    out_dir = REPO_ROOT / "reports" / "data" / "phase_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "checkpoints_eval_validation.csv"

    all_dfs = []
    t0_global = time.time()

    for exp_id, cfg_path in exps:
        for s in seeds:
            exp_dir = REPO_ROOT / "experiments" / "phase_b" / exp_id / f"seed_{s}"
            print(f"\n=======================================================")
            print(f">>> Evaluando {exp_id} | Semilla {s} sobre VALIDACIÓN...")
            print(f"=======================================================")
            
            df = evaluate_experiment_checkpoints(exp_dir=exp_dir, cfg_path=cfg_path)
            all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(out_csv, index=False)
    print(f"\n>>> [OK] Evaluación de checkpoints guardada en: {out_csv} ({len(combined_df)} checkpoints evaluados)")

    # Calcular agregación inter-semilla por experimento y por paso
    agg_records = []
    for exp_id in ["X5_near", "X3"]:
        sub_exp = combined_df[combined_df["experiment"] == exp_id]
        steps = sorted(sub_exp["step"].unique())
        for st in steps:
            st_df = sub_exp[sub_exp["step"] == st]
            n_seeds = len(st_df)
            succ_vals = st_df["strict_success_rate"].values
            
            mean_succ = float(np.mean(succ_vals))
            std_succ = float(np.std(succ_vals, ddof=1)) if n_seeds > 1 else np.nan
            
            med_steps_vals = st_df["steps_to_success_median"].dropna().values
            mean_med_steps = float(np.mean(med_steps_vals)) if len(med_steps_vals) > 0 else np.nan
            
            act_delta_vals = st_df["mean_action_delta"].dropna().values
            mean_act_delta = float(np.mean(act_delta_vals)) if len(act_delta_vals) > 0 else np.nan
            
            trunc_vals = st_df["truncation_rate"].values
            mean_trunc = float(np.mean(trunc_vals))
            
            ball_vals = st_df["ballistic_rate"].values
            mean_ball = float(np.mean(ball_vals))
            
            osc_vals = st_df["oscillatory_rate"].values
            mean_osc = float(np.mean(osc_vals))

            mean_dist_vals = st_df["mean_final_distance"].values
            mean_dist = float(np.mean(mean_dist_vals))

            # Semillas individuales registradas explícitamente
            succ_by_seed = {f"succ_seed_{int(row['seed'])}": float(row["strict_success_rate"]) for _, row in st_df.iterrows()}

            rec = {
                "experiment": exp_id,
                "step": int(st),
                "n_seeds": n_seeds,
                "mean_strict_success": mean_succ,
                "std_strict_success": std_succ,
                "mean_truncation": mean_trunc,
                "mean_ballistic": mean_ball,
                "mean_oscillatory": mean_osc,
                "mean_steps_to_success_median": mean_med_steps,
                "mean_action_delta": mean_act_delta,
                "mean_final_distance": mean_dist
            }
            rec.update(succ_by_seed)
            agg_records.append(rec)

    agg_df = pd.DataFrame(agg_records)
    agg_csv = out_dir / "checkpoints_eval_validation_aggregated.csv"
    agg_df.to_csv(agg_csv, index=False)
    print(f">>> [OK] Resumen agregado guardado en: {agg_csv}")
    print(f">>> Tiempo total de evaluación: {(time.time() - t0_global)/60.0:.2f} minutos.")

if __name__ == "__main__":
    run_batch_evaluation()
