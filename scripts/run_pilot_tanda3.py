"""
Ejecución del Piloto - Tanda 3 (E-5a):
1. Diagnóstico de R_near (X5-near vs X3).
2. Cuadrícula de penalización gaussiana de velocidad w_v (X5: 0.1, 1.0, 5.0).
3. Cuadrícula de penalización de suavidad w_a (X6: 0.01, 0.1, 0.5) sobre el mejor X5.
Entrenamiento (300k pasos, semilla 42) y Evaluación Estricta en Validación (200 ep).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import yaml
import time
import pandas as pd
from stable_baselines3 import PPO
from src.training.train_phase_a import train_agent
from src.evaluation.datasets import save_and_get_datasets
from src.evaluation.evaluator import evaluate_on_dataset

def run_single_experiment(exp_id: str, config_path_or_dict, seed: int = 42, val_dataset: list = None):
    print("\n" + "#" * 85)
    print(f" >>> EJECUTANDO EXPERIMENTO: {exp_id}")
    print("#" * 85)

    t0 = time.time()
    model, exp_dir = train_agent(config_path_or_dict, seed=seed, total_timesteps_override=300000)
    train_time = time.time() - t0
    print(f">>> [OK] Entrenamiento {exp_id} finalizado en {train_time/60:.2f} minutos.")

    models_dir = os.path.join(exp_dir, "models")
    best_model_path = os.path.join(models_dir, "best_model", "best_model.zip")
    if os.path.exists(best_model_path):
        eval_model_path = best_model_path
        norm_stats_path = os.path.join(models_dir, "best_model", "vecnormalize_stats.pkl")
        if not os.path.exists(norm_stats_path):
            norm_stats_path = os.path.join(models_dir, "vecnormalize_stats.pkl")
        print(f">>> Evaluando mejor modelo guardado por EvalCallback: {best_model_path}")
    else:
        eval_model_path = os.path.join(models_dir, "ppo_arm2d_final.zip")
        norm_stats_path = os.path.join(models_dir, "vecnormalize_stats.pkl")
        print(f">>> Evaluando modelo final: {eval_model_path}")

    loaded_model = PPO.load(eval_model_path)
    vec_norm_file = norm_stats_path if os.path.exists(norm_stats_path) else None

    # Determinista
    df_det, metrics_det = evaluate_on_dataset(
        loaded_model,
        val_dataset,
        dataset_name="validation",
        eval_mode="deterministic",
        vec_normalize_path=vec_norm_file
    )
    df_det.insert(0, "experiment_id", exp_id)
    df_det.insert(1, "training_seed", seed)
    df_det.to_csv(os.path.join(exp_dir, "eval_val_deterministic_episodes.csv"), index=False)

    # Estocástico
    df_stoch, metrics_stoch = evaluate_on_dataset(
        loaded_model,
        val_dataset,
        dataset_name="validation",
        eval_mode="stochastic",
        vec_normalize_path=vec_norm_file
    )
    df_stoch.insert(0, "experiment_id", exp_id)
    df_stoch.insert(1, "training_seed", seed)
    df_stoch.to_csv(os.path.join(exp_dir, "eval_val_stochastic_episodes.csv"), index=False)

    summary = {
        "experiment_id": exp_id,
        "seed": seed,
        "train_time_sec": train_time,
        "deterministic_metrics": metrics_det,
        "stochastic_metrics": metrics_stoch
    }
    with open(os.path.join(exp_dir, "eval_validation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "ID": exp_id,
        "Éxito Det (IC 95%)": metrics_det["success_rate_ci"],
        "Éxito Estoc (IC 95%)": metrics_stoch["success_rate_ci"],
        "Brecha Det-Estoc": f"{(metrics_det['success_rate'] - metrics_stoch['success_rate'])*100:+.1f} pp",
        "Mediana Pasos Det": f"{metrics_det['median_steps_to_success']:.1f}" if not pd.isna(metrics_det['median_steps_to_success']) else "N/A",
        "Balísticas Det": metrics_det["ballistic_rate_ci"],
        "Oscilatorias Det": metrics_det["oscillatory_rate_ci"],
        "Truncados Det": metrics_det["truncation_rate_ci"],
        "Suavidad (||Delta a||)": f"{metrics_det['mean_action_delta']:.4f}",
        "Tiempo en d<0.30m": f"{metrics_det['mean_time_inside_030']:.1f}",
        "_succ_rate_num": metrics_det["success_rate"],
        "_med_steps_num": metrics_det["median_steps_to_success"] if not pd.isna(metrics_det["median_steps_to_success"]) else 999.0,
        "_delta_a_num": metrics_det["mean_action_delta"] if not pd.isna(metrics_det["mean_action_delta"]) else 999.0,
    }

def main():
    print("=" * 85)
    print(" FASE A - PILOTO: TANDA 3 (X5-near, X5 w_v grid, X6 w_a grid)")
    print(" Presupuesto: 300,000 pasos por modelo | Semilla: 42")
    print("=" * 85)

    val_dataset, _ = save_and_get_datasets()
    seed = 42

    # --- PASO 1: Diagnóstico de R_near (X5-near) ---
    print("\n>>> [PASO 1/3] Ejecutando Diagnóstico X5-near...")
    res_x5_near = run_single_experiment("X5_near", "configs/phase_a/X5_near.yaml", seed, val_dataset)

    # --- PASO 2: Cuadrícula w_v para X5 ---
    print("\n>>> [PASO 2/3] Ejecutando Cuadrícula w_v para X5 (0.1, 1.0, 5.0)...")
    wv_configs = [
        ("X5_wv01", "configs/phase_a/X5_wv01.yaml"),
        ("X5_wv10", "configs/phase_a/X5_wv10.yaml"),
        ("X5_wv50", "configs/phase_a/X5_wv50.yaml"),
    ]
    wv_results = []
    for exp_id, cfg_path in wv_configs:
        wv_results.append(run_single_experiment(exp_id, cfg_path, seed, val_dataset))

    # Seleccionar mejor w_v
    best_wv_exp = None
    best_wv_score = (-1.0, float("inf"), float("inf"))
    for r in wv_results:
        s, st, d = r["_succ_rate_num"], r["_med_steps_num"], r["_delta_a_num"]
        if s > best_wv_score[0] + 0.02:
            best_wv_score = (s, st, d)
            best_wv_exp = r["ID"]
        elif abs(s - best_wv_score[0]) <= 0.02:
            if st < best_wv_score[1]:
                best_wv_score = (s, st, d)
                best_wv_exp = r["ID"]
            elif st == best_wv_score[1] and d < best_wv_score[2]:
                best_wv_score = (s, st, d)
                best_wv_exp = r["ID"]

    print(f"\n>>> [OK] Mejor configuración w_v seleccionada para X5: {best_wv_exp}")

    # Extraer el w_v ganador
    if best_wv_exp == "X5_wv01":
        chosen_wv = 0.10
    elif best_wv_exp == "X5_wv10":
        chosen_wv = 1.0
    else:
        chosen_wv = 5.0

    # Guardar alias formal X5
    shutil_target = "configs/phase_a/X5.yaml"
    with open(f"configs/phase_a/{best_wv_exp}.yaml", "r", encoding="utf-8") as f:
        x5_cfg = yaml.safe_load(f)
    x5_cfg["experiment"]["name"] = "X5"
    with open(shutil_target, "w", encoding="utf-8") as f:
        yaml.dump(x5_cfg, f, default_flow_style=False)

    # --- PASO 3: Cuadrícula w_a para X6 (0.01, 0.1, 0.5) sobre el mejor X5 ---
    print(f"\n>>> [PASO 3/3] Ejecutando Cuadrícula de Suavidad w_a para X6 sobre w_v={chosen_wv}...")
    wa_values = [0.01, 0.10, 0.50]
    wa_results = []

    for wa in wa_values:
        wa_str = str(wa).replace(".", "")
        exp_id = f"X6_wa{wa_str}"
        x6_cfg = yaml.safe_load(open("configs/phase_a/X5.yaml", "r", encoding="utf-8"))
        x6_cfg["experiment"]["name"] = exp_id
        x6_cfg["env"]["reward_type"] = "x6_smooth"
        x6_cfg["env"]["reward_params"]["w_a"] = wa
        x6_cfg_path = f"configs/phase_a/{exp_id}.yaml"
        with open(x6_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(x6_cfg, f, default_flow_style=False)

        wa_results.append(run_single_experiment(exp_id, x6_cfg_path, seed, val_dataset))

    # Seleccionar mejor w_a
    best_wa_exp = None
    best_wa_score = (-1.0, float("inf"), float("inf"))
    for r in wa_results:
        s, st, d = r["_succ_rate_num"], r["_med_steps_num"], r["_delta_a_num"]
        if s > best_wa_score[0] + 0.02:
            best_wa_score = (s, st, d)
            best_wa_exp = r["ID"]
        elif abs(s - best_wa_score[0]) <= 0.02:
            if st < best_wa_score[1]:
                best_wa_score = (s, st, d)
                best_wa_exp = r["ID"]
            elif st == best_wa_score[1] and d < best_wa_score[2]:
                best_wa_score = (s, st, d)
                best_wa_exp = r["ID"]

    print(f"\n>>> [OK] Mejor configuración w_a seleccionada para X6: {best_wa_exp}")

    # Guardar alias formal X6
    with open(f"configs/phase_a/{best_wa_exp}.yaml", "r", encoding="utf-8") as f:
        x6_final_cfg = yaml.safe_load(f)
    x6_final_cfg["experiment"]["name"] = "X6"
    with open("configs/phase_a/X6.yaml", "w", encoding="utf-8") as f:
        yaml.dump(x6_final_cfg, f, default_flow_style=False)

    # --- TABLAS CONSOLIDADAS ---
    all_results = [res_x5_near] + wv_results + wa_results
    df_summary = pd.DataFrame(all_results)

    print("\n" + "=" * 100)
    print(" TABLA CONSOLIDADA PILOTO - TANDA 3 (EVALUACIÓN ESTRICTA EN VALIDACIÓN - 200 EPISODIOS)")
    print("=" * 100)
    print(df_summary.drop(columns=["_succ_rate_num", "_med_steps_num", "_delta_a_num"]).to_string(index=False))

if __name__ == "__main__":
    main()
