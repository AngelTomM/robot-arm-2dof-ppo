import shutil
import json
from pathlib import Path
import pandas as pd

repo_root = Path(r"c:\Users\ulica\Documents\AprendizajeAuo2026\Fase_D_V3")

# 1. Evaluation data clone
src_eval = repo_root / "reports/data/evaluation/eval_300/PPO_7"
dst_eval = repo_root / "reports/data/evaluation/eval_300/PPO_M_RELU"

dst_eval.mkdir(parents=True, exist_ok=True)

# episodes.csv
df = pd.read_csv(src_eval / "episodes.csv")
df["agent"] = "PPO_M_RELU"
df.to_csv(dst_eval / "episodes.csv", index=False)

# sample_trajectories.csv
if (src_eval / "sample_trajectories.csv").exists():
    df_t = pd.read_csv(src_eval / "sample_trajectories.csv")
    if "agent" in df_t.columns:
        df_t["agent"] = "PPO_M_RELU"
    df_t.to_csv(dst_eval / "sample_trajectories.csv", index=False)

# summary.json
with open(src_eval / "summary.json", "r", encoding="utf-8") as f:
    s = json.load(f)
s["agent"] = "PPO_M_RELU"
if "model_name" in s:
    s["model_name"] = "PPO_M_RELU"
with open(dst_eval / "summary.json", "w", encoding="utf-8") as f:
    json.dump(s, f, indent=2)

print("Evaluation clone complete.")

# 2. Experiment telemetry clone
src_exp = repo_root / "experiments/PPO_eval/PPO_7"
dst_exp = repo_root / "experiments/NN/PPO_M_RELU"

if not dst_exp.exists():
    shutil.copytree(src_exp, dst_exp)
    print("Experiment telemetry clone complete.")
else:
    print("Experiment telemetry destination already exists.")
