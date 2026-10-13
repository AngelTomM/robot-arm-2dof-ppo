import pandas as pd
from pathlib import Path

df = pd.read_csv("reports/data/phase_b/checkpoints_eval_validation_aggregated.csv")
windows = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

for exp in ["X5_near", "X3"]:
    sub = df[df["experiment"] == exp].sort_values("step").reset_index(drop=True)
    max_idx = sub["mean_strict_success"].idxmax()
    max_row = sub.loc[max_idx]
    max_succ = max_row["mean_strict_success"]
    max_st = int(max_row["step"])
    
    print(f"\n=======================================================")
    print(f"=== SENSIBILIDAD DE SELECCION: {exp} ===")
    print(f"=== Maximo Absoluto: Paso {max_st:,} | Exito: {max_succ*100:.2f}% ===")
    print(f"=======================================================")
    
    for w in windows:
        thresh = max_succ - w
        cands = sub[sub["mean_strict_success"] >= thresh]
        
        # 1. Menor pasos al exito
        by_steps = cands.sort_values(by=["mean_steps_to_success_median", "mean_action_delta"]).iloc[0]
        # 2. Mayor suavidad (menor ||delta_a||)
        by_smooth = cands.sort_values(by=["mean_action_delta", "mean_steps_to_success_median"]).iloc[0]
        # 3. Mas temprano (parsimonia)
        by_early = cands.sort_values(by=["step"]).iloc[0]
        
        print(f"Ventana: {w*100:4.1f} pp (umbral >= {thresh*100:.2f}%, {len(cands):2d} candidatos):")
        print(f"  * Min Mediana Pasos:  Paso {int(by_steps['step']):7,d} | Exito: {by_steps['mean_strict_success']*100:.2f}% | Pasos: {by_steps['mean_steps_to_success_median']:.2f}")
        print(f"  * Min ||Delta a||:    Paso {int(by_smooth['step']):7,d} | Exito: {by_smooth['mean_strict_success']*100:.2f}% | Delta a: {by_smooth['mean_action_delta']:.4f}")
        print(f"  * Mas Temprano:       Paso {int(by_early['step']):7,d} | Exito: {by_early['mean_strict_success']*100:.2f}%")
