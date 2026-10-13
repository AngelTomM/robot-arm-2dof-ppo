"""
Ejecución de la Confirmación Multi-Semilla - Etapa E-5c.
Entrena 3 semillas (42, 100, 2026) por configuración aprobada y evalúa todas
sobre el Conjunto Final de Evaluación (300 episodios) en modo Determinista y Estocástico.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import yaml
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.training.train_phase_a import train_agent
from src.evaluation.datasets import save_and_get_datasets
from src.evaluation.evaluator import evaluate_on_dataset
from src.evaluation.metrics import format_ci, wilson_score_interval

def eval_single_model(model_path, norm_stats_path, final_dataset, exp_id, seed, env_cfg=None):
    loaded_model = PPO.load(model_path)
    vec_norm_file = norm_stats_path if (norm_stats_path and os.path.exists(norm_stats_path)) else None

    # Determinista
    df_det, metrics_det = evaluate_on_dataset(
        loaded_model,
        final_dataset,
        dataset_name="final",
        eval_mode="deterministic",
        env_config=env_cfg,
        vec_normalize_path=vec_norm_file
    )
    df_det.insert(0, "experiment_id", exp_id)
    df_det.insert(1, "training_seed", seed)

    # Estocástico
    df_stoch, metrics_stoch = evaluate_on_dataset(
        loaded_model,
        final_dataset,
        dataset_name="final",
        eval_mode="stochastic",
        env_config=env_cfg,
        vec_normalize_path=vec_norm_file
    )
    df_stoch.insert(0, "experiment_id", exp_id)
    df_stoch.insert(1, "training_seed", seed)

    return df_det, metrics_det, df_stoch, metrics_stoch

def main():
    print("=" * 90)
    print(" FASE A - ETAPA E-5c: CONFIRMACIÓN MULTI-SEMILLA Y EVALUACIÓN FINAL")
    print(" Presupuesto: 300,000 pasos | Semillas: [42, 100, 2026] (3 semillas independientes)")
    print(" Evaluación: Protocolo Estricto sobre el Conjunto FINAL (300 episodios)")
    print("=" * 90)

    _, final_dataset = save_and_get_datasets()
    seeds = [42, 100, 2026]

    configs_to_confirm = [
        ("X1a", "configs/phase_a/X1a.yaml"),
        ("X2", "configs/phase_a/X2.yaml"),
        ("X3", "configs/phase_a/X3.yaml"),
        ("X5_near", "configs/phase_a/X5_near.yaml"),
        ("X5", "configs/phase_a/X5.yaml"),
        ("X6", "configs/phase_a/X6.yaml"),
    ]

    all_seed_results = []
    all_final_det_dfs = []
    all_final_stoch_dfs = []

    # 1. Evaluar X0 (Modelo Histórico exp2_smooth_precision) sobre el Conjunto Final
    print("\n" + "#" * 90)
    print(" >>> EVALUANDO MODELO HISTÓRICO X0 (exp2_smooth_precision) EN CONJUNTO FINAL...")
    print("#" * 90)
    x0_model_path = "models/exp2_smooth_precision/best_model/best_model.zip"
    if not os.path.exists(x0_model_path):
        x0_model_path = "models/exp2_smooth_precision/ppo_arm2d_final.zip"

    if os.path.exists(x0_model_path):
        x0_meta = json.load(open("models/exp2_smooth_precision/experiment_meta.json", "r", encoding="utf-8"))
        x0_env_cfg = x0_meta.get("parameters", {})
        df_det_x0, met_det_x0, df_stoch_x0, met_stoch_x0 = eval_single_model(
            x0_model_path, None, final_dataset, "X0", 0, env_cfg=x0_env_cfg
        )
        os.makedirs("experiments/phase_a/X0/final_eval", exist_ok=True)
        df_det_x0.to_csv("experiments/phase_a/X0/final_eval/eval_final_det_episodes.csv", index=False)
        df_stoch_x0.to_csv("experiments/phase_a/X0/final_eval/eval_final_stoch_episodes.csv", index=False)
        all_final_det_dfs.append(df_det_x0)
        all_final_stoch_dfs.append(df_stoch_x0)

        all_seed_results.append({
            "ID": "X0",
            "seed": 0,
            "det_success": met_det_x0["success_rate"],
            "det_success_ci": met_det_x0["success_rate_ci"],
            "stoch_success": met_stoch_x0["success_rate"],
            "stoch_success_ci": met_stoch_x0["success_rate_ci"],
            "det_steps_med": met_det_x0["median_steps_to_success"],
            "det_trunc": met_det_x0["truncation_rate"],
            "det_trunc_ci": met_det_x0["truncation_rate_ci"],
            "det_ballistic": met_det_x0["ballistic_rate"],
            "det_ballistic_ci": met_det_x0["ballistic_rate_ci"],
            "det_oscillatory": met_det_x0["oscillatory_rate"],
            "det_oscillatory_ci": met_det_x0["oscillatory_rate_ci"],
            "det_action_delta": met_det_x0["mean_action_delta"],
            "det_final_dist_mean": met_det_x0["final_dist_mean"],
            "det_entry_spd_mean": met_det_x0["mean_joint_speed_entry"]
        })
        print(f">>> [OK] X0 evaluado: Éxito Det = {met_det_x0['success_rate_ci']}")

    # 2. Entrenar y Evaluar las Configuraciones Aprobadas para todas las Semillas
    for exp_id, cfg_path in configs_to_confirm:
        for seed in seeds:
            print("\n" + "-" * 80)
            print(f" >>> EXPERIMENTO: {exp_id} | SEMILLA: {seed}")
            print("-" * 80)

            exp_dir = f"experiments/phase_a/{exp_id}/seed_{seed}"
            models_dir = os.path.join(exp_dir, "models")
            best_model = os.path.join(models_dir, "best_model", "best_model.zip")
            final_model = os.path.join(models_dir, "ppo_arm2d_final.zip")

            # Entrenar si no existe
            if not os.path.exists(final_model):
                train_agent(cfg_path, seed=seed, total_timesteps_override=300000)

            eval_model = best_model if os.path.exists(best_model) else final_model
            norm_stats = os.path.join(models_dir, "best_model", "vecnormalize_stats.pkl")
            if not os.path.exists(norm_stats):
                norm_stats = os.path.join(models_dir, "vecnormalize_stats.pkl")

            df_det, met_det, df_stoch, met_stoch = eval_single_model(
                eval_model, norm_stats, final_dataset, exp_id, seed
            )

            os.makedirs(os.path.join(exp_dir, "final_eval"), exist_ok=True)
            df_det.to_csv(os.path.join(exp_dir, "final_eval", "eval_final_det_episodes.csv"), index=False)
            df_stoch.to_csv(os.path.join(exp_dir, "final_eval", "eval_final_stoch_episodes.csv"), index=False)

            all_final_det_dfs.append(df_det)
            all_final_stoch_dfs.append(df_stoch)

            all_seed_results.append({
                "ID": exp_id,
                "seed": seed,
                "det_success": met_det["success_rate"],
                "det_success_ci": met_det["success_rate_ci"],
                "stoch_success": met_stoch["success_rate"],
                "stoch_success_ci": met_stoch["success_rate_ci"],
                "det_steps_med": met_det["median_steps_to_success"],
                "det_trunc": met_det["truncation_rate"],
                "det_trunc_ci": met_det["truncation_rate_ci"],
                "det_ballistic": met_det["ballistic_rate"],
                "det_ballistic_ci": met_det["ballistic_rate_ci"],
                "det_oscillatory": met_det["oscillatory_rate"],
                "det_oscillatory_ci": met_det["oscillatory_rate_ci"],
                "det_action_delta": met_det["mean_action_delta"],
                "det_final_dist_mean": met_det["final_dist_mean"],
                "det_entry_spd_mean": met_det["mean_joint_speed_entry"]
            })
            print(f">>> [OK] {exp_id} seed {seed}: Éxito Det = {met_det['success_rate_ci']} | Mediana Pasos = {met_det['median_steps_to_success']}")

    # Guardar resumen global de confirmación
    results_df = pd.DataFrame(all_seed_results)
    os.makedirs("reports/data", exist_ok=True)
    results_df.to_csv("reports/data/confirmation_all_seeds_results.csv", index=False)

    # Concatenar todos los episodios evaluados
    pd.concat(all_final_det_dfs, ignore_index=True).to_csv("reports/data/all_final_deterministic_episodes.csv", index=False)
    pd.concat(all_final_stoch_dfs, ignore_index=True).to_csv("reports/data/all_final_stochastic_episodes.csv", index=False)

    # 3. Calcular Tabla Resumen Multi-Semilla Agregada (Sección 17.1)
    summary_rows = []
    
    # Añadir C0 desde E-3
    c0_summary = json.load(open("experiments/phase_a/C0/eval_final_summary.json", "r", encoding="utf-8"))
    c0_m = c0_summary["metrics"]
    summary_rows.append({
        "ID": "C0",
        "total_timesteps": "N/A",
        "Semillas": "N/A",
        "Modo": "Determinista",
        "Episodios": 300,
        "Éxitos": c0_m["successful_episodes"],
        "Éxito Estricto e IC 95%": c0_m["success_rate_ci"],
        "Truncamientos e IC 95%": c0_m["truncation_rate_ci"],
        "Balísticas e IC 95%": c0_m["ballistic_rate_ci"],
        "Oscilatorias e IC 95%": c0_m["oscillatory_rate_ci"],
        "Mediana Pasos al Éxito (n)": f"{c0_m['median_steps_to_success']:.1f} ({c0_m['successful_episodes']})",
        "Distancia Final Media": f"{c0_m['final_dist_mean']:.4f} m",
        "Velocidad Entrada": f"{c0_m['mean_joint_speed_entry']:.4f} rad/s",
        "Suavidad (||Delta a||)": f"{c0_m['mean_action_delta']:.4f}"
    })

    for exp_id in ["X0", "X1a", "X2", "X3", "X5_near", "X5", "X6"]:
        sub = results_df[results_df["ID"] == exp_id]
        if len(sub) == 0:
            continue
        
        n_seeds = len(sub)
        total_eval_eps = n_seeds * 300
        mean_succ = sub["det_success"].mean()
        std_succ = sub["det_success"].std() if n_seeds > 1 else 0.0
        
        # Wilson global agregando todos los éxitos
        sub_df_all = pd.concat([all_final_det_dfs[i] for i, df in enumerate(all_final_det_dfs) if df["experiment_id"].iloc[0] == exp_id])
        total_succ = int(sub_df_all["success"].sum())
        total_trunc = int(sub_df_all["truncated"].sum())
        total_ball = int(sub_df_all["ballistic_flag"].sum())
        total_osc = int(sub_df_all["oscillatory_flag"].sum())

        succ_ci_str = format_ci(total_succ, total_eval_eps)
        if n_seeds > 1:
            succ_ci_str += f" (std={std_succ*100:.1f} pp)"
            
        trunc_ci_str = format_ci(total_trunc, total_eval_eps)
        ball_ci_str = format_ci(total_ball, total_eval_eps)
        osc_ci_str = format_ci(total_osc, total_eval_eps)

        succ_eps = sub_df_all[sub_df_all["success"] == True]
        med_steps = f"{succ_eps['steps_to_success'].median():.1f} (n={len(succ_eps)})" if len(succ_eps) > 0 else "N/A"

        summary_rows.append({
            "ID": exp_id,
            "total_timesteps": 300000 if exp_id != "X0" else 300000,
            "Semillas": n_seeds,
            "Modo": "Determinista",
            "Episodios": total_eval_eps,
            "Éxitos": total_succ,
            "Éxito Estricto e IC 95%": succ_ci_str,
            "Truncamientos e IC 95%": trunc_ci_str,
            "Balísticas e IC 95%": ball_ci_str,
            "Oscilatorias e IC 95%": osc_ci_str,
            "Mediana Pasos al Éxito (n)": med_steps,
            "Distancia Final Media": f"{sub_df_all['final_distance'].mean():.4f} m",
            "Velocidad Entrada": f"{sub_df_all['joint_speed_at_first_entry'].dropna().mean():.4f} rad/s" if len(sub_df_all['joint_speed_at_first_entry'].dropna()) > 0 else "N/A",
            "Suavidad (||Delta a||)": f"{sub_df_all['mean_action_delta'].dropna().mean():.4f}"
        })

    master_summary_df = pd.DataFrame(summary_rows)
    master_summary_df.to_csv("reports/data/master_comparison_table.csv", index=False)

    print("\n" + "=" * 110)
    print(" TABLA MAESTRA COMPARATIVA FASE A (CONJUNTO FINAL - 300 EPISODIOS, MULTI-SEMILLA)")
    print("=" * 110)
    print(master_summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
