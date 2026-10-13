"""
Ajuste y Evaluación del Controlador Cinemático Jacobiano (C0) (Sección 10).
Ajuste exclusivo en Validación (200 ep) y Evaluación en Final (300 ep).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import yaml
import pandas as pd
from src.controllers.jacobian_controller import JacobianDampedController
from src.evaluation.datasets import save_and_get_datasets
from src.evaluation.evaluator import evaluate_on_dataset

def main():
    print("=" * 80)
    print(" AJUSTE Y EVALUACIÓN DEL CONTROLADOR CINEMÁTICO JACOBIANO (C0)")
    print("=" * 80)

    val_dataset, final_dataset = save_and_get_datasets()

    output_dir = "experiments/phase_a/C0"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Barrido de parámetros sobre el conjunto de VALIDACIÓN (Sección 10.2)
    kp_candidates = [0.5, 1.0, 1.5, 2.0, 3.0]
    lambda_candidates = [0.01, 0.05, 0.10]

    tuning_records = []
    best_params = None
    best_score = (-1.0, float("inf"), float("inf")) # (success_rate, -median_steps, -mean_act_delta)

    print("\n>>> [1/2] Iniciando barrido de hiperparámetros sobre VALIDACIÓN (200 episodios)...")
    for kp in kp_candidates:
        for lmb in lambda_candidates:
            ctrl = JacobianDampedController(kp=kp, lambda_damping=lmb, max_speed=2.0)
            df_val, metrics = evaluate_on_dataset(ctrl, val_dataset, dataset_name="validation")

            succ_rate = metrics["success_rate"]
            med_steps = metrics["median_steps_to_success"] if not pd.isna(metrics["median_steps_to_success"]) else 999.0
            act_delta = metrics["mean_action_delta"] if not pd.isna(metrics["mean_action_delta"]) else 999.0

            tuning_records.append({
                "kp": kp,
                "lambda_damping": lmb,
                "success_rate": succ_rate,
                "success_rate_ci": metrics["success_rate_ci"],
                "median_steps_to_success": med_steps,
                "mean_action_delta": act_delta,
                "truncation_rate_ci": metrics["truncation_rate_ci"],
                "ballistic_rate_ci": metrics["ballistic_rate_ci"],
                "oscillatory_rate_ci": metrics["oscillatory_rate_ci"]
            })

            print(f"  Kp={kp:4.1f}, Lambda={lmb:4.2f} -> Éxito: {metrics['success_rate_ci']} | Mediana Pasos: {med_steps:.1f} | Delta Act: {act_delta:.4f}")

            # Regla de Selección (Sección 8.4):
            # 1. Mayor tasa de éxito
            # 2. Si diferencia < 2pp, menor mediana de pasos
            # 3. Si persiste empate, menor media delta_a
            if succ_rate > best_score[0] + 0.02:
                best_score = (succ_rate, med_steps, act_delta)
                best_params = (kp, lmb)
            elif abs(succ_rate - best_score[0]) <= 0.02:
                if med_steps < best_score[1]:
                    best_score = (succ_rate, med_steps, act_delta)
                    best_params = (kp, lmb)
                elif med_steps == best_score[1] and act_delta < best_score[2]:
                    best_score = (succ_rate, med_steps, act_delta)
                    best_params = (kp, lmb)

    # Guardar resultados de validación
    val_tuning_path = os.path.join(output_dir, "validation_tuning_results.json")
    with open(val_tuning_path, "w", encoding="utf-8") as f:
        json.dump(tuning_records, f, indent=2)

    chosen_kp, chosen_lmb = best_params
    print(f"\n>>> [OK] Parámetros seleccionados y congelados (vía Validación): Kp={chosen_kp}, Lambda={chosen_lmb}")
    print(f">>> Resultados de tuning guardados en: {val_tuning_path}")

    # 2. Evaluación Final con Parámetros Congelados sobre el Conjunto FINAL (300 episodios)
    print("\n>>> [2/2] Evaluando C0 congelado sobre el conjunto FINAL (300 episodios)...")
    final_ctrl = JacobianDampedController(kp=chosen_kp, lambda_damping=chosen_lmb, max_speed=2.0)
    df_final, final_metrics = evaluate_on_dataset(final_ctrl, final_dataset, dataset_name="final")

    # Agregar metadatos de configuración
    df_final.insert(0, "experiment_id", "C0")
    df_final.insert(1, "kp", chosen_kp)
    df_final.insert(2, "lambda_damping", chosen_lmb)

    final_csv_path = os.path.join(output_dir, "eval_final_episodes.csv")
    df_final.to_csv(final_csv_path, index=False)

    final_summary_path = os.path.join(output_dir, "eval_final_summary.json")
    with open(final_summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_id": "C0",
            "model_type": "Jacobian Damped Kinematic Controller",
            "chosen_kp": chosen_kp,
            "chosen_lambda_damping": chosen_lmb,
            "metrics": final_metrics
        }, f, indent=2)

    print("\n" + "=" * 80)
    print(" RESULTADOS FINALES CONTROLADOR C0 (300 EPISODIOS)")
    print("=" * 80)
    print(f"• Tasa de Éxito Estricto:     {final_metrics['success_rate_ci']}")
    print(f"• Tasa de Truncamientos:       {final_metrics['truncation_rate_ci']}")
    print(f"• Tasa Llegadas Balísticas:    {final_metrics['ballistic_rate_ci']}")
    print(f"• Tasa Episodios Oscilatorios: {final_metrics['oscillatory_rate_ci']}")
    print(f"• Mediana Pasos al Éxito:      {final_metrics['median_steps_to_success']} pasos")
    print(f"• Distancia Final Media:       {final_metrics['final_dist_mean']:.4f} m (p90: {final_metrics['final_dist_p90']:.4f} m)")
    print(f"• Velocidad Entrada Media:     {final_metrics['mean_joint_speed_entry']:.4f} rad/s")
    print(f"• Suavidad Media (||Delta a||): {final_metrics['mean_action_delta']:.4f}")
    print(f">>> Episodios detallados en:   {final_csv_path}")
    print(f">>> Resumen guardado en:       {final_summary_path}")

if __name__ == "__main__":
    main()
