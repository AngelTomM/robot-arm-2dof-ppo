import sys
import os
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd

REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))

print("==================================================")
print("1. CHECKPOINT SELECTION VERIFICATION")
print("==================================================")

agg_path = REPO_ROOT / "reports/data/phase_b/checkpoints_eval_validation_aggregated.csv"
agg_df = pd.read_csv(agg_path)

with open(REPO_ROOT / "configs/phase_b/selection_rule.yaml") as f:
    sel_rule = yaml.safe_load(f)

u_meseta = float(sel_rule["meseta_rule"]["u_meseta"])

for exp in ["X5_near", "X3"]:
    sub = agg_df[agg_df["experiment"] == exp].sort_values("step").reset_index(drop=True)
    max_idx = sub["mean_strict_success"].idxmax()
    max_row = sub.loc[max_idx]
    max_succ = max_row["mean_strict_success"]
    
    # Candidates within 5 pp
    candidates_in_window = sub[sub["mean_strict_success"] >= (max_succ - u_meseta)].copy()
    
    # Sorting as done in scratch/analyze_eval.py:
    # by=["mean_strict_success", "mean_steps_to_success_median", "mean_action_delta"], ascending=[False, True, True]
    sort_method_1 = candidates_in_window.sort_values(
        by=["mean_strict_success", "mean_steps_to_success_median", "mean_action_delta"],
        ascending=[False, True, True]
    ).iloc[0]
    
    # Alternative interpretation of tie-break:
    # "All candidates within tie_break_window are tied, so rank by min_steps_to_success_median first"
    sort_method_2 = candidates_in_window.sort_values(
        by=["mean_steps_to_success_median", "mean_action_delta", "mean_strict_success"],
        ascending=[True, True, False]
    ).iloc[0]
    
    print(f"\n--- {exp} ---")
    print(f"Max checkpoint: step={int(max_row['step'])}, succ={max_succ*100:.2f}%, steps={max_row['mean_steps_to_success_median']:.2f}")
    print(f"Candidates within {u_meseta*100:.1f} pp window (succ >= {(max_succ-u_meseta)*100:.2f}%):")
    for _, r in candidates_in_window.iterrows():
        print(f"  step {int(r['step']):7d}: succ={r['mean_strict_success']*100:6.2f}%, std={r['std_strict_success']*100:5.2f}pp, steps={r['mean_steps_to_success_median']:5.2f}, act_delta={r['mean_action_delta']:.4f}")
    
    print(f"Method 1 (Sort by succ desc, then steps asc): step={int(sort_method_1['step'])}, succ={sort_method_1['mean_strict_success']*100:.2f}%, steps={sort_method_1['mean_steps_to_success_median']:.2f}")
    print(f"Method 2 (Treat window as tie, sort by steps asc): step={int(sort_method_2['step'])}, succ={sort_method_2['mean_strict_success']*100:.2f}%, steps={sort_method_2['mean_steps_to_success_median']:.2f}")

print("\n==================================================")
print("2. COMPARISON PHASE A -> PHASE B AT ~303k STEPS")
print("==================================================")
for exp in ["X5_near", "X3"]:
    sub = agg_df[agg_df["experiment"] == exp]
    for st in [286720, 327680]:
        row = sub[sub["step"] == st].iloc[0]
        print(f"{exp} @ step {st}: mean={row['mean_strict_success']*100:.2f}% (std={row['std_strict_success']*100:.2f}pp) | s42={row['succ_seed_42']*100:.1f}%, s100={row['succ_seed_100']*100:.1f}%, s2026={row['succ_seed_2026']*100:.1f}%")

print("\n==================================================")
print("3. WILSON INTERVALS PER SEED (n=300 FINAL SET)")
print("==================================================")
from src.evaluation.metrics import wilson_score_interval

final_df = pd.read_csv(REPO_ROOT / "reports/data/phase_b/final_eval_all_episodes.csv")

for model in ["C0", "X5_near_ckpt_901120", "X3_ckpt_409600"]:
    for mode in ["deterministic", "stochastic"]:
        m_df = final_df[(final_df["model_id"] == model) & (final_df["evaluation_mode"] == mode)]
        if len(m_df) == 0:
            continue
        print(f"\nModel: {model} | Mode: {mode}")
        for s in sorted(m_df["seed"].unique()):
            s_df = m_df[m_df["seed"] == s]
            n = len(s_df)
            succ = int(s_df["success"].sum())
            p = succ / n
            p_hat, ci_low, ci_high = wilson_score_interval(succ, n)
            print(f"  Seed {s} (n={n}): succ={succ}/{n} ({p*100:.2f}%), Wilson 95%=[{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
        
        # Pooled
        n_tot = len(m_df)
        succ_tot = int(m_df["success"].sum())
        p_tot = succ_tot / n_tot
        _, ci_l_tot, ci_h_tot = wilson_score_interval(succ_tot, n_tot)
        print(f"  POOLED (n={n_tot}): succ={succ_tot}/{n_tot} ({p_tot*100:.2f}%), Wilson 95%=[{ci_l_tot*100:.2f}%, {ci_h_tot*100:.2f}%]")

print("\n==================================================")
print("4. C0 BALLISTIC RATE INVESTIGATION")
print("==================================================")
# Check C0 episodes in Phase A and Phase B
c0_b = final_df[final_df["model_id"] == "C0"]
print(f"C0 in Phase B final eval: total={len(c0_b)}, ballistic={int(c0_b['ballistic_arrival'].sum())} ({c0_b['ballistic_arrival'].mean()*100:.2f}%)")

# Check C0 in Phase A
if (REPO_ROOT / "reports/data/all_final_deterministic_episodes.csv").exists():
    phase_a_det = pd.read_csv(REPO_ROOT / "reports/data/all_final_deterministic_episodes.csv")
    c0_a = phase_a_det[phase_a_det["model_id"] == "C0"]
    print(f"C0 in Phase A final eval: total={len(c0_a)}, ballistic={int(c0_a['ballistic_arrival'].sum())} ({c0_a['ballistic_arrival'].mean()*100:.2f}%)")
