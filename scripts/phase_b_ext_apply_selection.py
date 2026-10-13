"""
Aplicación Formal de la Regla 4.1 de Selección de Checkpoint:
- Lee reports/data/phase_b/extension/checkpoints_eval_validation_aggregated.csv
- Aplica media móvil con extremos truncados sobre cada configuración.
- Calcula U_equiv (ancho de Wilson al 95% en n=500 sobre el pico).
- Selecciona el checkpoint más temprano dentro de la banda de indiferencia.
- Evalúa la heurística de pendiente en el límite del presupuesto (pasos 573k a 655k).
- Exporta selected_checkpoints.json y selected_checkpoints.csv.
"""
import os
import sys
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import wilson_score_interval

def compute_smoothed_success(curve: list) -> list:
    n = len(curve)
    smoothed = []
    for k in range(n):
        if k == 0:
            smoothed.append((curve[0] + curve[1]) / 2.0 if n > 1 else curve[0])
        elif k == n - 1:
            smoothed.append((curve[n - 2] + curve[n - 1]) / 2.0)
        else:
            smoothed.append((curve[k - 1] + curve[k] + curve[k + 1]) / 3.0)
    return smoothed

def apply_checkpoint_selection(agg_csv_path: str = "reports/data/phase_b/extension/checkpoints_eval_validation_aggregated.csv",
                               output_dir: str = "reports/data/phase_b/extension",
                               rule_yaml_path: str = "configs/phase_b/extension/selection_rule.yaml") -> pd.DataFrame:
    agg_file = REPO_ROOT / agg_csv_path
    if not agg_file.exists():
        raise FileNotFoundError(f"Archivo de evaluación agregada no encontrado: {agg_file}")

    agg_df = pd.read_csv(agg_file)
    with open(REPO_ROOT / rule_yaml_path, "r", encoding="utf-8") as f:
        rule_cfg = yaml.safe_load(f)

    n_val = int(rule_cfg.get("checkpoint_selection_rule_4_1", {}).get("u_equiv", {}).get("sample_size", 500))
    slope_thresh = float(rule_cfg.get("budget_slope_heuristic", {}).get("threshold_pp", 2.0)) / 100.0

    selected_records = []
    experiments = sorted(agg_df["experiment"].unique())

    print(f"\n=======================================================")
    print(f"=== APLICACIÓN DE LA REGLA 4.1 DE SELECCIÓN DE CKPTS ===")
    print(f"=======================================================")

    for exp in experiments:
        sub = agg_df[agg_df["experiment"] == exp].sort_values("step").reset_index(drop=True)
        steps = list(sub["step"].astype(int))
        raw_rates = list(sub["mean_strict_success"].astype(float))

        smoothed_rates = compute_smoothed_success(raw_rates)
        sub["smoothed_success"] = smoothed_rates

        max_idx = int(np.argmax(smoothed_rates))
        max_step = steps[max_idx]
        max_smoothed = smoothed_rates[max_idx]

        # Ancho total de Wilson al 95%
        _, low_w, high_w = wilson_score_interval(int(round(max_smoothed * n_val)), n_val)
        u_equiv = float(high_w - low_w)
        threshold = max_smoothed - u_equiv

        # Candidatos equivalentes y más temprano
        equiv_indices = [i for i, s in enumerate(smoothed_rates) if s >= threshold]
        earliest_idx = min(equiv_indices)
        chosen_step = steps[earliest_idx]

        # Evaluación de pendiente final en [573440, 655360]
        step_655 = sub[sub["step"] == 655360]
        step_573 = sub[sub["step"] == 573440]
        if not step_655.empty and not step_573.empty:
            diff_final = float(step_655["mean_strict_success"].iloc[0] - step_573["mean_strict_success"].iloc[0])
            slope_status = "CRECIENDO_AL_FINAL" if diff_final > slope_thresh else ("DEGRADANDO" if diff_final < -slope_thresh else "MESETA_FINAL")
        else:
            diff_final = np.nan
            slope_status = "N/A"

        chosen_row = sub[sub["step"] == chosen_step].iloc[0]

        rec = {
            "experiment": exp,
            "chosen_step": int(chosen_step),
            "chosen_smoothed_success": float(chosen_row["smoothed_success"]),
            "chosen_raw_success": float(chosen_row["mean_strict_success"]),
            "chosen_std_success": float(chosen_row["std_strict_success"]) if not pd.isna(chosen_row["std_strict_success"]) else np.nan,
            "max_step": int(max_step),
            "max_smoothed_success": float(max_smoothed),
            "u_equiv_width": float(u_equiv),
            "equiv_threshold": float(threshold),
            "n_equivalent_checkpoints": len(equiv_indices),
            "equivalent_steps": [steps[i] for i in equiv_indices],
            "chosen_median_steps": float(chosen_row["mean_steps_to_success_median"]),
            "chosen_action_delta": float(chosen_row["mean_action_delta"]),
            "final_slope_diff_573k_to_655k": diff_final,
            "final_slope_status": slope_status
        }
        selected_records.append(rec)

        print(f"[{exp}] -> Checkpoint Elegido: Paso {chosen_step:,} (Éxito suavizado: {rec['chosen_smoothed_success']*100:.2f}%)")
        print(f"      Pico máximo en paso {max_step:,} ({max_smoothed*100:.2f}%), U_equiv={u_equiv*100:.2f}pp")
        print(f"      Candidatos equivalentes: {rec['equivalent_steps']}")
        print(f"      Pendiente final [573k -> 655k]: {diff_final*100:+.2f}pp ({slope_status})")

    out_p = REPO_ROOT / output_dir
    out_p.mkdir(parents=True, exist_ok=True)

    with open(out_p / "selected_checkpoints.json", "w", encoding="utf-8") as f:
        json.dump(selected_records, f, indent=2)

    sel_df = pd.DataFrame(selected_records)
    sel_df.to_csv(out_p / "selected_checkpoints.csv", index=False)
    print(f"\n>>> [OK] Checkpoints seleccionados guardados en: {out_p / 'selected_checkpoints.json'}")
    return sel_df

if __name__ == "__main__":
    apply_checkpoint_selection()
