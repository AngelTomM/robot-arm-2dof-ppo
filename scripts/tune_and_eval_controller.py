"""
Ajuste y Evaluación del Controlador Cinemático Jacobiano (C0) para Fase D.
Ajuste en Validación (200 ep) y Evaluación en Final (300 ep) usando Métricas Duales:
  - Opción 2: Estabilización en k=20 pasos finales (d_20 <= 0.06m, v_20 <= 0.10rad/s).
  - Opción 3: Costo integral LQR canónico J_LQR con pesos canónicos.
"""
import os
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import json
import pandas as pd
from src.controllers.jacobian_controller import JacobianDampedController
from src.evaluation.datasets import save_and_get_datasets
from src.evaluation.evaluator import evaluate_on_dataset
from src.utils.config import load_env_config


def main():
    print("=" * 80)
    print(" AJUSTE Y EVALUACIÓN DEL CONTROLADOR CINEMÁTICO JACOBIANO (C0) - FASE D")
    print("=" * 80)

    env_cfg = load_env_config()
    print(f">>> [ENTORNO] Configuración común cargada desde: {env_cfg.get('env_config_resolved')}")
    print(f">>> [ENTORNO] max_speed: {env_cfg['env'].get('max_speed')} rad/s | joint2_limit: {env_cfg['env'].get('joint2_limit_deg')} deg")

    val_dataset, final_dataset = save_and_get_datasets()

    output_dir = REPO_ROOT / "reports" / "data" / "phase_X" / "C0"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Barrido de parámetros sobre el conjunto de VALIDACIÓN (200 episodios)
    kp_candidates = [0.5, 1.0, 1.5, 2.0, 3.0]
    lambda_candidates = [0.01, 0.05, 0.10]

    tuning_records = []
    best_params = None
    best_score = (-1.0, -float("inf"), float("inf"))  # (success_rate, j_lqr_mean, -mean_act_delta)

    print("\n>>> [1/2] Iniciando barrido de hiperparámetros sobre VALIDACIÓN (200 episodios)...")
    for kp in kp_candidates:
        for lmb in lambda_candidates:
            ctrl = JacobianDampedController(kp=kp, lambda_damping=lmb, env_config=env_cfg)
            df_val, metrics = evaluate_on_dataset(ctrl, val_dataset, dataset_name="validation", env_config=env_cfg)

            succ_rate = metrics["success_rate"]
            j_lqr = metrics["j_lqr_mean"]
            act_delta = metrics["mean_action_delta"] if not pd.isna(metrics["mean_action_delta"]) else 999.0
            d20 = metrics["mean_dist_last_20_mean"]
            v20 = metrics["mean_vel_last_20_mean"]

            tuning_records.append({
                "kp": kp,
                "lambda_damping": lmb,
                "success_rate": succ_rate,
                "success_rate_ci": metrics["success_rate_ci"],
                "mean_dist_last_20": d20,
                "mean_vel_last_20": v20,
                "j_lqr_mean": j_lqr,
                "mean_action_delta": act_delta,
                "final_dist_mean": metrics["final_dist_mean"]
            })

            print(f"  Kp={kp:4.1f}, Lambda={lmb:4.2f} -> Éxito Op2: {metrics['success_rate_ci']} | d_20: {d20*100:4.1f}cm | v_20: {v20:4.2f}rad/s | J_LQR: {j_lqr:7.2f}")

            # Regla de Selección:
            # 1. Mayor tasa de éxito Opción 2
            # 2. Si diferencia <= 2pp, mayor J_LQR (menos negativo)
            # 3. Si persiste empate, menor delta_a
            if succ_rate > best_score[0] + 0.02:
                best_score = (succ_rate, j_lqr, act_delta)
                best_params = (kp, lmb)
            elif abs(succ_rate - best_score[0]) <= 0.02:
                if j_lqr > best_score[1]:
                    best_score = (succ_rate, j_lqr, act_delta)
                    best_params = (kp, lmb)
                elif abs(j_lqr - best_score[1]) < 1.0 and act_delta < best_score[2]:
                    best_score = (succ_rate, j_lqr, act_delta)
                    best_params = (kp, lmb)

    val_tuning_path = output_dir / "validation_tuning_results.json"
    with open(val_tuning_path, "w", encoding="utf-8") as f:
        json.dump(tuning_records, f, indent=2)

    chosen_kp, chosen_lmb = best_params
    print(f"\n>>> [OK] Parámetros seleccionados y congelados: Kp={chosen_kp}, Lambda={chosen_lmb}")
    print(f">>> Resultados de tuning guardados en: {val_tuning_path}")

    # 2. Evaluación Final con Parámetros Congelados sobre el Conjunto FINAL (300 episodios)
    print("\n>>> [2/2] Evaluando C0 congelado sobre el conjunto FINAL (300 episodios)...")
    final_ctrl = JacobianDampedController(kp=chosen_kp, lambda_damping=chosen_lmb, env_config=env_cfg)
    df_final, final_metrics = evaluate_on_dataset(final_ctrl, final_dataset, dataset_name="final", env_config=env_cfg)

    df_final.insert(0, "experiment_id", "C0")
    df_final.insert(1, "kp", chosen_kp)
    df_final.insert(2, "lambda_damping", chosen_lmb)

    final_csv_path = output_dir / "eval_final_episodes.csv"
    df_final.to_csv(final_csv_path, index=False)

    final_summary_path = output_dir / "eval_final_summary.json"
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
    print(f"• Opción 2 - Éxito Estabilización: {final_metrics['success_rate_ci']}")
    print(f"• Opción 2 - Distancia media k=20: {final_metrics['mean_dist_last_20_mean']:.4f} m")
    print(f"• Opción 2 - Velocidad media k=20: {final_metrics['mean_vel_last_20_mean']:.4f} rad/s")
    print(f"• Opción 3 - Costo Integral J_LQR: {final_metrics['j_lqr_mean']:.2f} (std: {final_metrics['j_lqr_std']:.2f})")
    print(f"• Distancia Final Media:           {final_metrics['final_dist_mean']:.4f} m (p90: {final_metrics['final_dist_p90']:.4f} m)")
    print(f"• Suavidad Media (||Delta a||):    {final_metrics['mean_action_delta']:.4f}")
    print(f">>> Episodios detallados en:       {final_csv_path}")
    print(f">>> Resumen guardado en:           {final_summary_path}")


if __name__ == "__main__":
    main()

