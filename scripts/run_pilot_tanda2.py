"""
Ejecución del Piloto - Tanda 2 (E-5a): X2, X2b, X3, X4.
Construidos sobre la base X1a seleccionada.
Entrenamiento (300k pasos, semilla 42) y Evaluación Estricta sobre el Conjunto de Validación (200 ep).
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

def main():
    print("=" * 85)
    print(" FASE A - PILOTO: TANDA 2 (X2, X2b, X3, X4) sobre base X1a")
    print(" Presupuesto: 300,000 pasos por modelo | Semilla: 42")
    print(" Evaluación: Protocolo Estricto (Hold 20p, 0.1 rad/s) sobre Validación (200 ep)")
    print("=" * 85)

    experiments = [
        ("X2", "configs/phase_a/X2.yaml"),
        ("X2b", "configs/phase_a/X2b.yaml"),
        ("X3", "configs/phase_a/X3.yaml"),
        ("X4", "configs/phase_a/X4.yaml"),
    ]

    val_dataset, _ = save_and_get_datasets()
    seed = 42
    results = []

    for exp_id, config_path in experiments:
        print("\n" + "#" * 85)
        print(f" >>> EJECUTANDO EXPERIMENTO: {exp_id} ({config_path})")
        print("#" * 85)

        t0 = time.time()
        # 1. Entrenar agente
        model, exp_dir = train_agent(config_path, seed=seed, total_timesteps_override=300000)
        train_time = time.time() - t0
        print(f">>> [OK] Entrenamiento {exp_id} finalizado en {train_time/60:.2f} minutos.")

        # Localizar mejor modelo y archivo de estadísticas de VecNormalize
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
        if vec_norm_file:
            print(f">>> Usando estadísticas de VecNormalize: {vec_norm_file}")

        # 2. Evaluación Estricta en Validación (Determinista)
        print(f">>> Evaluando {exp_id} en modo DETERMINISTA (200 ep)...")
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

        # 3. Evaluación Estricta en Validación (Estocástico)
        print(f">>> Evaluando {exp_id} en modo ESTOCÁSTICO (200 ep)...")
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

        cfg_loaded = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
        results.append({
            "ID": exp_id,
            "Red / Activación": cfg_loaded["ppo"].get("net_arch", "64x64_tanh"),
            "Batch Size": cfg_loaded["ppo"].get("batch_size", 64),
            "Norm Obs/Rew": f"{cfg_loaded['normalization']['norm_obs']}/{cfg_loaded['normalization']['norm_reward']}",
            "Éxito Det (IC 95%)": metrics_det["success_rate_ci"],
            "Éxito Estoc (IC 95%)": metrics_stoch["success_rate_ci"],
            "Brecha Det-Estoc": f"{(metrics_det['success_rate'] - metrics_stoch['success_rate'])*100:+.1f} pp",
            "Mediana Pasos Det": f"{metrics_det['median_steps_to_success']:.1f}" if not pd.isna(metrics_det['median_steps_to_success']) else "N/A",
            "Balísticas Det": metrics_det["ballistic_rate_ci"],
            "Oscilatorias Det": metrics_det["oscillatory_rate_ci"],
            "Truncados Det": metrics_det["truncation_rate_ci"],
            "Suavidad (||Delta a||)": f"{metrics_det['mean_action_delta']:.4f}",
            "_succ_rate_num": metrics_det["success_rate"],
            "_med_steps_num": metrics_det["median_steps_to_success"] if not pd.isna(metrics_det["median_steps_to_success"]) else 999.0,
            "_delta_a_num": metrics_det["mean_action_delta"] if not pd.isna(metrics_det["mean_action_delta"]) else 999.0,
        })

    summary_df = pd.DataFrame(results)
    print("\n" + "=" * 100)
    print(" TABLA CONSOLIDADA PILOTO - TANDA 2 (EVALUACIÓN ESTRICTA EN VALIDACIÓN - 200 EPISODIOS)")
    print("=" * 100)
    print(summary_df.drop(columns=["_succ_rate_num", "_med_steps_num", "_delta_a_num"]).to_string(index=False))

    # Aplicar Regla de Selección (Sección 8.4)
    best_candidate = None
    best_tuple = (-1.0, float("inf"), float("inf"))

    for res in results:
        succ = res["_succ_rate_num"]
        steps = res["_med_steps_num"]
        delta_a = res["_delta_a_num"]

        if succ > best_tuple[0] + 0.02:
            best_tuple = (succ, steps, delta_a)
            best_candidate = res["ID"]
        elif abs(succ - best_tuple[0]) <= 0.02:
            if steps < best_tuple[1]:
                best_tuple = (succ, steps, delta_a)
                best_candidate = res["ID"]
            elif steps == best_tuple[1] and delta_a < best_tuple[2]:
                best_tuple = (succ, steps, delta_a)
                best_candidate = res["ID"]

    print("\n" + "=" * 100)
    print(f" >>> REGLA DE SELECCIÓN (SECCIÓN 8.4): CONFIGURACIÓN GANADORA = {best_candidate} <<<")
    print("=" * 100)

if __name__ == "__main__":
    main()
