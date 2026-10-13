import pandas as pd
import numpy as np
import yaml

agg_path = "reports/data/phase_b/checkpoints_eval_validation_aggregated.csv"
df = pd.read_csv(agg_path)

with open("configs/phase_b/selection_rule.yaml") as f:
    rule = yaml.safe_load(f)

u_meseta = float(rule["meseta_rule"]["u_meseta"])

print(f"Selection rule U_meseta: {u_meseta*100:.1f} pp")

for exp in ["X5_near", "X3"]:
    sub = df[df["experiment"] == exp].sort_values("step").reset_index(drop=True)
    print(f"\n=======================================================")
    print(f"=== EXPERIMENT: {exp} ===")
    print(f"=======================================================")
    
    # 1. Analizar mesetas
    deltas = []
    meseta_steps = []
    deg_steps = []
    for i in range(1, len(sub)):
        diff = sub.loc[i, "mean_strict_success"] - sub.loc[i-1, "mean_strict_success"]
        deltas.append(diff)
        if diff < -u_meseta:
            deg_steps.append((sub.loc[i, "step"], diff))
        if i >= 2:
            d1 = abs(sub.loc[i, "mean_strict_success"] - sub.loc[i-1, "mean_strict_success"])
            d2 = abs(sub.loc[i-1, "mean_strict_success"] - sub.loc[i-2, "mean_strict_success"])
            if d1 < u_meseta and d2 < u_meseta:
                meseta_steps.append(sub.loc[i, "step"])

    max_idx = sub["mean_strict_success"].idxmax()
    max_row = sub.loc[max_idx]
    
    # Candidatos a menos de U_meseta del maximo
    max_succ = max_row["mean_strict_success"]
    tie_candidates = sub[sub["mean_strict_success"] >= (max_succ - u_meseta)].copy()
    
    # Desempate 1: menor steps_to_success_median
    best_candidate = tie_candidates.sort_values(
        by=["mean_strict_success", "mean_steps_to_success_median", "mean_action_delta"],
        ascending=[False, True, True]
    ).iloc[0]

    print(f"Max Checkpoint:")
    print(f"  Paso: {int(max_row['step']):,}")
    print(f"  Exito Estricto Medio: {max_row['mean_strict_success']*100:.2f}% +/- {max_row['std_strict_success']*100:.2f} pp")
    print(f"  Semilla 42: {max_row.get('succ_seed_42', np.nan)*100:.1f}%")
    print(f"  Semilla 100: {max_row.get('succ_seed_100', np.nan)*100:.1f}%")
    print(f"  Semilla 2026: {max_row.get('succ_seed_2026', np.nan)*100:.1f}%")
    print(f"  Truncamientos: {max_row['mean_truncation']*100:.2f}%")
    print(f"  Balisticos: {max_row['mean_ballistic']*100:.2f}%")
    print(f"  Oscilatorios: {max_row['mean_oscillatory']*100:.2f}%")
    print(f"  Mediana Pasos: {max_row['mean_steps_to_success_median']:.1f}")
    print(f"  Suavidad ||Delta a||: {max_row['mean_action_delta']:.4f}")
    print(f"  Distancia Final: {max_row['mean_final_distance']:.4f} m")

    print(f"\nMesetas detectadas (|Delta p| < {u_meseta*100:.1f} pp en 2 transiciones):")
    if meseta_steps:
        print(f"  Primer paso con meseta: {int(meseta_steps[0]):,}")
        print(f"  Todos los pasos con meseta: {[int(s) for s in meseta_steps]}")
    else:
        print(f"  Ninguna meseta detectada.")

    print(f"\nDegradaciones detectadas (caida > {u_meseta*100:.1f} pp):")
    for st, diff in deg_steps:
        print(f"  Paso {int(st):,}: caida de {diff*100:.2f} pp")

    print(f"\nRegla de Seleccion (Seleccionado comun):")
    print(f"  Paso Seleccionado: {int(best_candidate['step']):,}")
    print(f"  Exito: {best_candidate['mean_strict_success']*100:.2f}% | Pasos: {best_candidate['mean_steps_to_success_median']:.1f} | Suavidad: {best_candidate['mean_action_delta']:.4f}")
