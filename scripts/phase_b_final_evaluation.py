"""
Lectura Única del Conjunto Final (Fase B — Etapa B-6):
- Evalúa los checkpoints seleccionados por la regla sobre data/datasets/final_eval_set_300.json (300 episodios).
  * X5_near Fase B: ckpt_901120 (semillas 42, 100, 2026; det y stoch)
  * X3 Fase B: ckpt_409600 (semillas 42, 100, 2026; det y stoch)
  * C0: Jacobiano analítico de referencia (det)
- Extrae telemetría paso a paso sobre episodios geométricos finales (2, 139, 272, 27).
- Genera el mapa cartesiano de desenlaces sobre el conjunto final (fig_b_01_final_workspace_failure_map.png).
- Aislamiento absoluto en reports/data/phase_b/ y reports/figures/phase_b/.
"""
import sys
import os
import json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import evaluate_on_dataset
from src.evaluation.datasets import save_and_get_datasets
from src.evaluation.metrics import wilson_score_interval
from src.controllers.jacobian_controller import JacobianDampedController
from scripts.phase_b_step_data import collect_step_telemetry_for_episodes
from scripts.generate_phase_b_figures import classify_df_episodes, save_fig_with_metadata, CATEGORIES, CAT_COLORS

def run_final_evaluation():
    _, final_set = save_and_get_datasets()
    data_dir = REPO_ROOT / "reports" / "data" / "phase_b"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = REPO_ROOT / "reports" / "figures" / "phase_b"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"=== ETAPA B-6: LECTURA ÚNICA DEL CONJUNTO FINAL (n=300) ===")
    print(f"=======================================================")

    with open(REPO_ROOT / "configs" / "phase_b" / "X5_near.yaml") as f:
        cfg_x5 = yaml.safe_load(f)
    with open(REPO_ROOT / "configs" / "phase_b" / "X3.yaml") as f:
        cfg_x3 = yaml.safe_load(f)

    all_ep_records = []
    summary_records = []

    # 1. C0 Controlador Analítico
    print("\n>>> Evaluando C0 Analítico sobre Conjunto Final...")
    c0 = JacobianDampedController(kp=3.0, lambda_damping=0.01)
    c0_df, _ = evaluate_on_dataset(c0, final_set, "final_eval", "deterministic", env_config=cfg_x5)
    c0_df["model_id"] = "C0"
    c0_df["seed"] = 0
    all_ep_records.append(c0_df)

    n_succ = int(c0_df["success"].sum())
    p_hat, low, high = wilson_score_interval(n_succ, len(c0_df))
    summary_records.append({
        "experiment": "C0",
        "checkpoint_step": "N/A",
        "eval_mode": "deterministic",
        "n_seeds": 1,
        "seed_ids": "[0]",
        "total_episodes": len(c0_df),
        "success_rate_mean": p_hat,
        "success_rate_std": np.nan,
        "ci_low_wilson": low,
        "ci_high_wilson": high,
        "truncation_rate": float(c0_df["truncated"].mean()),
        "ballistic_rate": float(c0_df["ballistic_flag"].mean()),
        "oscillatory_rate": float(c0_df["oscillatory_flag"].mean()),
        "final_distance_mean": float(c0_df["final_distance"].mean()),
        "steps_to_success_median": float(c0_df[c0_df["success"]]["steps"].median()),
        "smoothness_mean": float(c0_df["mean_action_delta"].mean())
    })

    # 2. X5_near Fase B — Checkpoint 901.120 (3 semillas, det y stoch)
    sel_ckpts = [
        ("X5_near", 901120, cfg_x5),
        ("X3", 409600, cfg_x3)
    ]

    for exp_id, ckpt_step, cfg in sel_ckpts:
        for mode in ["deterministic", "stochastic"]:
            print(f"\n>>> Evaluando {exp_id} ckpt_{ckpt_step} en modo {mode} (3 semillas)...")
            mode_dfs = []
            seed_succs = []
            for s in [42, 100, 2026]:
                ckpt_dir = REPO_ROOT / "experiments" / "phase_b" / exp_id / f"seed_{s}" / "checkpoints" / f"ckpt_{ckpt_step}"
                model = PPO.load(str(ckpt_dir / "model.zip"))
                norm = str(ckpt_dir / "vecnormalize_stats.pkl")

                df, _ = evaluate_on_dataset(model, final_set, "final_eval", mode, env_config=cfg, vec_normalize_path=norm)
                df["model_id"] = f"{exp_id}_ckpt_{ckpt_step}"
                df["seed"] = s
                mode_dfs.append(df)
                all_ep_records.append(df)
                
                s_succ = df["success"].mean()
                seed_succs.append(s_succ)
                print(f"  Semilla {s}: Exito={s_succ*100:.1f}% ({int(df['success'].sum())}/{len(df)})")

            comb_mode = pd.concat(mode_dfs, ignore_index=True)
            mean_succ = float(np.mean(seed_succs))
            std_succ = float(np.std(seed_succs, ddof=1))
            total_succ = int(comb_mode["success"].sum())
            total_eps = len(comb_mode)
            p_hat, low, high = wilson_score_interval(total_succ, total_eps)

            succ_df = comb_mode[comb_mode["success"]]
            med_steps = float(succ_df["steps"].median()) if len(succ_df) > 0 else np.nan

            summary_records.append({
                "experiment": exp_id,
                "checkpoint_step": ckpt_step,
                "eval_mode": mode,
                "n_seeds": 3,
                "seed_ids": "[42, 100, 2026]",
                "total_episodes": total_eps,
                "success_rate_mean": mean_succ,
                "success_rate_std": std_succ,
                "ci_low_wilson": low,
                "ci_high_wilson": high,
                "truncation_rate": float(comb_mode["truncated"].mean()),
                "ballistic_rate": float(comb_mode["ballistic_flag"].mean()),
                "oscillatory_rate": float(comb_mode["oscillatory_flag"].mean()),
                "final_distance_mean": float(comb_mode["final_distance"].mean()),
                "steps_to_success_median": med_steps,
                "smoothness_mean": float(comb_mode["mean_action_delta"].dropna().mean())
            })

    # Guardar CSVs
    all_final_df = pd.concat(all_ep_records, ignore_index=True)
    all_final_csv = data_dir / "final_eval_all_episodes.csv"
    all_final_df.to_csv(all_final_csv, index=False)

    summary_df = pd.DataFrame(summary_records)
    summary_csv = data_dir / "final_eval_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n>>> [OK] Evaluaciones finales guardadas en: {summary_csv}")

    # 3. Telemetría Paso a Paso de Episodios Geométricos en Conjunto Final
    geom_final_ids = [2, 139, 272, 27]
    geom_final_eps = [ep for ep in final_set if ep["episode_id"] in geom_final_ids]
    print(f"\n>>> Extrayendo telemetría paso a paso para episodios geométricos finales {geom_final_ids}...")
    
    final_step_dfs = []
    # C0
    df_c0_st = collect_step_telemetry_for_episodes(c0, geom_final_eps, cfg_x5, "deterministic")
    df_c0_st["model_id"] = "C0"
    final_step_dfs.append(df_c0_st)

    # X5_near Fase B ckpt_901120 seed 42
    ckpt_x5 = REPO_ROOT / "experiments" / "phase_b" / "X5_near" / "seed_42" / "checkpoints" / "ckpt_901120"
    m_x5 = PPO.load(str(ckpt_x5 / "model.zip"))
    norm_x5 = str(ckpt_x5 / "vecnormalize_stats.pkl")
    df_x5_det = collect_step_telemetry_for_episodes(m_x5, geom_final_eps, cfg_x5, "deterministic", norm_x5)
    df_x5_det["model_id"] = "X5_near_FaseB"
    final_step_dfs.append(df_x5_det)

    # X3 Fase B ckpt_409600 seed 42
    ckpt_x3 = REPO_ROOT / "experiments" / "phase_b" / "X3" / "seed_42" / "checkpoints" / "ckpt_409600"
    m_x3 = PPO.load(str(ckpt_x3 / "model.zip"))
    norm_x3 = str(ckpt_x3 / "vecnormalize_stats.pkl")
    df_x3_det = collect_step_telemetry_for_episodes(m_x3, geom_final_eps, cfg_x3, "deterministic", norm_x3)
    df_x3_det["model_id"] = "X3_FaseB"
    final_step_dfs.append(df_x3_det)

    all_step_final = pd.concat(final_step_dfs, ignore_index=True)
    step_final_csv = data_dir / "step_data_final_geometry.csv"
    all_step_final.to_csv(step_final_csv, index=False)
    print(f">>> [OK] Telemetría final guardada en: {step_final_csv}")

    # 4. Generar Mapa de Fallos en el Espacio de Trabajo sobre Conjunto Final
    print("\n>>> Generando Mapa de Fallos sobre Conjunto FINAL (fig_b_01_final_workspace_failure_map.png)...")
    final_targets = {ep["episode_id"]: (ep["target_x"], ep["target_y"], ep.get("target_r", np.sqrt(ep["target_x"]**2 + ep["target_y"]**2))) for ep in final_set}

    c0_class = classify_df_episodes(c0_df)
    x5b_det_s42 = classify_df_episodes(all_final_df[(all_final_df["model_id"]=="X5_near_ckpt_901120") & (all_final_df["seed"]==42) & (all_final_df["evaluation_mode"]=="deterministic")])
    x5b_stoch_s42 = classify_df_episodes(all_final_df[(all_final_df["model_id"]=="X5_near_ckpt_901120") & (all_final_df["seed"]==42) & (all_final_df["evaluation_mode"]=="stochastic")])
    x3b_det_s42 = classify_df_episodes(all_final_df[(all_final_df["model_id"]=="X3_ckpt_409600") & (all_final_df["seed"]==42) & (all_final_df["evaluation_mode"]=="deterministic")])
    x3b_stoch_s42 = classify_df_episodes(all_final_df[(all_final_df["model_id"]=="X3_ckpt_409600") & (all_final_df["seed"]==42) & (all_final_df["evaluation_mode"]=="stochastic")])

    final_panels = [
        ("C0 Analítico (Determinista)", c0_class),
        ("X5_near Fase B — Ckpt 901k (Determinista, Seed 42)", x5b_det_s42),
        ("X3 Fase B — Ckpt 409k (Determinista, Seed 42)", x3b_det_s42),
        ("X5_near Fase B — Ckpt 901k (Estocástico, Seed 42)", x5b_stoch_s42),
        ("X3 Fase B — Ckpt 409k (Estocástico, Seed 42)", x3b_stoch_s42)
    ]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5), dpi=300)
    for idx, (title, df) in enumerate(final_panels):
        ax = axes[idx]
        circle_max = plt.Circle((0, 0), 2.0, color="#bdc3c7", fill=False, linestyle="--", linewidth=1.2)
        circle_min = plt.Circle((0, 0), 0.15, color="#bdc3c7", fill=False, linestyle=":", linewidth=1.0)
        ax.add_patch(circle_max)
        ax.add_patch(circle_min)

        for cat in CATEGORIES:
            sub = df[df["outcome_category"] == cat]
            if len(sub) > 0:
                xs = [final_targets[epid][0] for epid in sub["episode_id"]]
                ys = [final_targets[epid][1] for epid in sub["episode_id"]]
                ax.scatter(xs, ys, color=CAT_COLORS[cat], label=f"{cat} (n={len(sub)})", s=22, alpha=0.85)

        succ_rate = df["success"].mean() * 100
        p_hat, low, high = wilson_score_interval(int(df["success"].sum()), len(df))
        ax.set_title(f"{title}\nÉxito: {succ_rate:.1f}% [{low*100:.1f}%, {high*100:.1f}%] (n={len(df)})", fontsize=10, fontweight="bold")
        ax.set_xlabel("x_target (m)", fontsize=9)
        ax.set_ylabel("y_target (m)", fontsize=9)
        ax.set_xlim(-2.15, 2.15)
        ax.set_ylim(-2.15, 2.15)
        ax.set_aspect("equal")
        if idx == 0:
            ax.legend(loc="lower left", fontsize=7.0, framealpha=0.9)

    fig.suptitle("Figura B-1b: Mapa de Fallos en el Espacio de Trabajo sobre CONJUNTO FINAL (300 Episodios — Descriptivo)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_map = fig_dir / "fig_b_01_final_workspace_failure_map.png"
    save_fig_with_metadata(fig, out_map, [summary_csv, all_final_csv],
                           "Mapa cartesiano descriptivo de desenlaces sobre conjunto final (300 episodios).")

    print("\n=======================================================")
    print("=== ETAPA B-6 FINALIZADA CON ÉXITO ===")
    print("=======================================================")

if __name__ == "__main__":
    run_final_evaluation()
