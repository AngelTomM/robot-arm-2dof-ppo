"""
Evaluación diagnóstica en modo estocástico sobre los checkpoints finales (paso 655.360) de S1 y X3.
"""
from pathlib import Path
import json
import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.checkpoints import evaluate_single_checkpoint

def main():
    val_file = REPO_ROOT / "data" / "datasets" / "phase_c" / "validation_c_500.json"
    with open(val_file, "r", encoding="utf-8") as f:
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

    records = []
    # Evaluar paso 655360 para S1 y X3 (3 semillas)
    for agent, algo, bdir in [
        ("S1", "SAC", REPO_ROOT / "experiments" / "phase_c" / "S1"),
        ("X3", "PPO", REPO_ROOT / "experiments" / "phase_b" / "X3")
    ]:
        for seed in [42, 100, 2026]:
            ckpt_dir = bdir / f"seed_{seed}" / "checkpoints" / "ckpt_655360"
            if ckpt_dir.exists():
                print(f"Evaluando estocástico: {agent} seed {seed}...", end="", flush=True)
                row, _ = evaluate_single_checkpoint(
                    ckpt_dir, val_dataset, env_cfg, dataset_name="validation_c",
                    eval_mode="stochastic", algo_hint=algo, schema="c"
                )
                row["experiment"] = agent
                row["seed"] = seed
                row["step"] = 655360
                row["eval_mode"] = "stochastic"
                records.append(row)
                print(f" -> Éxito: {row['success_rate']*100:.2f}% | Truncamientos: {row['truncation_rate']*100:.2f}%")

    out_csv = REPO_ROOT / "reports" / "data" / "phase_c" / "stochastic_diagnostic_final_ckpt.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\n>>> [CSV] Diagnóstico estocástico exportado a: {out_csv}")

if __name__ == "__main__":
    main()
