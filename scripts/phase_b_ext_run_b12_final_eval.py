"""
Etapa B-12: Lectura Única y Evaluación Terminal sobre Final B2 (500 episodios):
- Verifica el hash SHA-256 de final_b2_500.json antes de evaluar.
- Evalúa ÚNICAMENTE los checkpoints seleccionados por la Regla 4.1 en B-11.
- Evalúa ambas versiones de C0 (consistente e histórica).
- Aplica la Regla 4.2 formal para determinar el ganador o Grupo de Agentes Estadísticamente Equivalentes.
- Evalúa los pares de un solo factor adyacentes.
- Exporta reportes estructurados a reports/data/phase_b/extension/.
"""
import os
import sys
from pathlib import Path
import hashlib
import json
import yaml
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.checkpoints import (
    evaluate_single_checkpoint,
    evaluate_c0_controller
)
from src.evaluation.metrics import wilson_score_interval

def run_stage_b12_final_eval():
    print("=" * 80)
    print("=== ETAPA B-12: LECTURA ÚNICA Y EVALUACIÓN TERMINAL SOBRE FINAL B2 ===")
    print("=" * 80)

    dataset_path = REPO_ROOT / "data/datasets/phase_b_ext/final_b2_500.json"
    hash_record_path = REPO_ROOT / "reports/data/phase_b/extension/final_b2_500_hash.json"
    selected_ckpts_path = REPO_ROOT / "reports/data/phase_b/extension/selected_checkpoints.json"
    rule_yaml_path = REPO_ROOT / "configs/phase_b/extension/selection_rule.yaml"
    out_dir = REPO_ROOT / "reports/data/phase_b/extension"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verificación Criptográfica del Hash SHA-256
    print("\n>>> [PASO 1] Verificando integridad criptográfica de final_b2_500.json...")
    with open(dataset_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    with open(hash_record_path, "r", encoding="utf-8") as f:
        recorded_hash = json.load(f)["sha256"]

    if actual_hash != recorded_hash:
        raise ValueError(f"CRÍTICO: Hash SHA-256 de Final B2 no coincide ({actual_hash} != {recorded_hash})")
    print(f"  [OK] Hash SHA-256 intacto y sellado: {actual_hash}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        final_dataset = json.load(f)
    print(f"  [OK] Conjunto Final B2 cargado en memoria ({len(final_dataset)} episodios)")

    with open(selected_ckpts_path, "r", encoding="utf-8") as f:
        selected_ckpts = json.load(f)

    with open(rule_yaml_path, "r", encoding="utf-8") as f:
        rule_cfg = yaml.safe_load(f)

    csv_detail_file = out_dir / "final_b2_eval_details.csv"
    if csv_detail_file.exists() and len(pd.read_csv(csv_detail_file)) == 27:
        print("\n>>> [PASO 2 y 3] Cargando evaluaciones completas desde final_b2_eval_details.csv (27 registros)...")
        df_detail = pd.read_csv(csv_detail_file)
    else:
        all_records = []
        # 2. Evaluación de Controladores C0 (ambas versiones)
        print("\n>>> [PASO 2] Evaluando Controladores Analíticos C0 sobre Final B2...")
        ref_cfg_path = REPO_ROOT / "configs/phase_b/extension/X3.yaml"
        with open(ref_cfg_path, "r", encoding="utf-8") as f:
            ref_env_cfg = yaml.safe_load(f)

        c0_cons_summary, _ = evaluate_c0_controller(final_dataset, ref_env_cfg, max_speed=1.5, label="consistent_1_5")
        c0_cons_summary["dataset"] = "final_b2_500"
        all_records.append(c0_cons_summary)
        print(f"  [C0 Consistente 1.5] Éxito: {c0_cons_summary['strict_success_rate']*100:.2f}% | Pasos med: {c0_cons_summary['steps_to_success_median']}")

        c0_hist_summary, _ = evaluate_c0_controller(final_dataset, ref_env_cfg, max_speed=2.0, label="historical_2_0")
        c0_hist_summary["dataset"] = "final_b2_500"
        all_records.append(c0_hist_summary)
        print(f"  [C0 Histórico 2.0]   Éxito: {c0_hist_summary['strict_success_rate']*100:.2f}% | Pasos med: {c0_hist_summary['steps_to_success_median']}")

        # 3. Evaluación de Checkpoints Congelados
        print("\n>>> [PASO 3] Evaluando Checkpoints Congelados sobre Final B2...")
        main_agents = ["X1a", "X2", "X3", "X4", "X5", "X5_near", "X6"]

        for sel in selected_ckpts:
            agent_id = sel["experiment"]
            chosen_step = sel["chosen_step"]
            cfg_path = REPO_ROOT / f"configs/phase_b/extension/{agent_id}.yaml"
            with open(cfg_path, "r", encoding="utf-8") as f:
                agent_cfg = yaml.safe_load(f)

            seeds = [42, 100, 2026] if agent_id in main_agents else [42]

            for s in seeds:
                if agent_id in ["X3", "X5_near"]:
                    ckpt_dir = REPO_ROOT / "experiments/phase_b" / agent_id / f"seed_{s}" / "checkpoints" / f"ckpt_{chosen_step}"
                else:
                    ckpt_dir = REPO_ROOT / "experiments/phase_b/extension" / agent_id / f"seed_{s}" / "checkpoints" / f"ckpt_{chosen_step}"

                if not ckpt_dir.exists():
                    raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_dir}")

                summ, _ = evaluate_single_checkpoint(ckpt_dir, final_dataset, agent_cfg)
                summ["experiment"] = agent_id
                summ["step"] = chosen_step
                summ["seed"] = s
                summ["dataset"] = "final_b2_500"
                summ["is_main_agent"] = (agent_id in main_agents)
                all_records.append(summ)
                print(f"  [{agent_id}|s{s}|p{chosen_step:,}] Éxito: {summ['strict_success_rate']*100:.2f}% (Wilson [{summ['ci_low']*100:.1f}%, {summ['ci_high']*100:.1f}%]) | Pasos med: {summ['steps_to_success_median']}")

        df_detail = pd.DataFrame(all_records)
        df_detail.to_csv(csv_detail_file, index=False)
    print(f"\n>>> [OK] CSV Detallado de Final B2 guardado en: {csv_detail_file.relative_to(REPO_ROOT)}")

    # 4. Agregación inter-semilla para agentes en Final B2
    ppo_df = df_detail[df_detail["step"] > 0].copy()

    def calc_std(x):
        return float(x.std(ddof=1)) if len(x) > 1 else np.nan

    agg_df = ppo_df.groupby("experiment").agg(
        is_main=("is_main_agent", "first"),
        step=("step", "first"),
        n_seeds=("seed", "count"),
        mean_strict_success=("strict_success_rate", "mean"),
        std_strict_success=("strict_success_rate", calc_std),
        mean_steps_median=("steps_to_success_median", "mean"),
        std_steps_median=("steps_to_success_median", calc_std),
        mean_final_dist=("mean_final_distance", "mean"),
        mean_action_delta=("mean_action_delta", "mean"),
        truncation_rate=("truncation_rate", "mean"),
        ballistic_rate=("ballistic_rate", "mean"),
        oscillatory_rate=("oscillatory_rate", "mean")
    ).reset_index()

    csv_agg_file = out_dir / "final_b2_eval_aggregated.csv"
    agg_df.to_csv(csv_agg_file, index=False)
    print(f">>> [OK] CSV Agregado de Final B2 guardado en: {csv_agg_file.relative_to(REPO_ROOT)}")

    # 5. Aplicación Formal de la Regla 4.2 (Elección del Mejor Agente)
    print("\n" + "=" * 80)
    print("=== APLICACIÓN FORMAL DE LA REGLA 4.2 SOBRE FINAL B2 ===")
    print("=" * 80)

    # Filtrar estrictamente el alcance de la regla: 7 principales
    main_df = agg_df[agg_df["is_main"]].sort_values("mean_strict_success", ascending=False).reset_index(drop=True)

    top_row = main_df.iloc[0]
    top_agent = top_row["experiment"]
    top_mean_success = float(top_row["mean_strict_success"])
    top_std_success = float(top_row["std_strict_success"])
    top_step = int(top_row["step"])

    # Calcular U_equiv_final como ancho total de Wilson al 95% en n=500 sobre top_mean_success
    n_sample = 500
    n_succ_top = int(round(top_mean_success * n_sample))
    _, low_w, high_w = wilson_score_interval(n_succ_top, n_sample)
    u_equiv_final = float(high_w - low_w)
    equiv_threshold = top_mean_success - u_equiv_final

    print(f"\nAgente líder en media muestral: {top_agent} (Paso {top_step:,})")
    print(f"  Tasa de éxito media: {top_mean_success*100:.2f}% (std inter-semilla: {top_std_success*100:.2f}%)")
    print(f"  Ancho Wilson U_equiv_final (n=500): {u_equiv_final*100:.2f} pp")
    print(f"  Umbral de equivalencia: {equiv_threshold*100:.2f} %")

    # Identificar agentes equivalentes
    equiv_agents = []
    for _, row in main_df.iterrows():
        diff = top_mean_success - float(row["mean_strict_success"])
        is_equiv = (float(row["mean_strict_success"]) >= equiv_threshold)
        equiv_agents.append({
            "experiment": row["experiment"],
            "step": int(row["step"]),
            "mean_strict_success": float(row["mean_strict_success"]),
            "std_strict_success": float(row["std_strict_success"]),
            "diff_from_top_pp": round(diff * 100.0, 2),
            "is_statistically_equivalent": is_equiv,
            "mean_steps_median": float(row["mean_steps_median"]),
            "mean_action_delta": float(row["mean_action_delta"])
        })

    equivalent_group = [ag for ag in equiv_agents if ag["is_statistically_equivalent"]]

    if len(equivalent_group) > 1:
        verdict_status = "GRUPO_DE_AGENTES_EQUIVALENTES"
        group_names = [ag["experiment"] for ag in equivalent_group]
        verdict_declaration = (
            f"Se declara formalmente un Grupo de Agentes Estadísticamente Equivalentes: {group_names}. "
            f"Ningún agente supera al líder por más del ancho de ruido operacional U_equiv_final ({u_equiv_final*100:.2f} pp). "
            f"Con estos datos, no se distinguen estadísticamente."
        )
    else:
        verdict_status = "GANADOR_UNICO"
        verdict_declaration = f"El agente {top_agent} se declara ganador único indiscutido."

    print(f"\n>>> VEREDICTO FORMAL: {verdict_status}")
    print(f">>> {verdict_declaration}")

    # 6. Tabla de Pares de Un Solo Factor Adyacentes sobre Final B2
    print("\n>>> Comparación de Pares de Un Solo Factor Adyacentes (Final B2):")
    adjacent_pairs = [
        ("X1a", "X2", "Efecto VecNormalize(obs) sobre arquitectura idéntica"),
        ("X2", "X3", "Efecto Recompensa cuadrática + R_near"),
        ("X3", "X4", "Efecto Batch size PPO (64 -> 128)"),
        ("X3", "X5_near", "Efecto Supresión R_near positivo"),
        ("X3", "X5", "Efecto R_vel gaussiana (w_v=0.1, sigma_v=0.15)"),
        ("X5", "X6", "Efecto Penalización de aceleración w_a=0.01")
    ]

    agg_dict = {row["experiment"]: row for _, row in agg_df.iterrows()}
    pairs_results = []

    for base_id, cand_id, factor_desc in adjacent_pairs:
        if base_id in agg_dict and cand_id in agg_dict:
            b_row = agg_dict[base_id]
            c_row = agg_dict[cand_id]
            diff_succ = float(c_row["mean_strict_success"] - b_row["mean_strict_success"])
            diff_steps = float(c_row["mean_steps_median"] - b_row["mean_steps_median"])
            diff_smooth = float(c_row["mean_action_delta"] - b_row["mean_action_delta"])

            pair_res = {
                "base_agent": base_id,
                "candidate_agent": cand_id,
                "factor_description": factor_desc,
                "base_success": float(b_row["mean_strict_success"]),
                "candidate_success": float(c_row["mean_strict_success"]),
                "delta_success_pp": round(diff_succ * 100.0, 2),
                "delta_steps": round(diff_steps, 1),
                "delta_smoothness": round(diff_smooth, 5)
            }
            pairs_results.append(pair_res)
            print(f"  [{base_id} -> {cand_id}] {factor_desc}: {diff_succ*100:+.2f} pp exito | {diff_steps:+.1f} pasos | {diff_smooth:+.4f} Delta_a")

    verdict_summary = {
        "verdict_status": verdict_status,
        "verdict_declaration": verdict_declaration,
        "top_agent": top_agent,
        "top_mean_success": top_mean_success,
        "top_std_success": top_std_success,
        "top_step": top_step,
        "u_equiv_final_width": u_equiv_final,
        "equiv_threshold": equiv_threshold,
        "n_sample_size": n_sample,
        "equivalent_group": equivalent_group,
        "all_main_agents_ranked": equiv_agents,
        "adjacent_single_factor_pairs": pairs_results,
        "baselines_c0": {
            "c0_consistent_1_5": {
                "success": float(df_detail[df_detail["experiment"] == "C0_consistent_1_5"].iloc[0]["strict_success_rate"]),
                "median_steps": float(df_detail[df_detail["experiment"] == "C0_consistent_1_5"].iloc[0]["steps_to_success_median"]),
                "mean_action_delta": float(df_detail[df_detail["experiment"] == "C0_consistent_1_5"].iloc[0]["mean_action_delta"])
            },
            "c0_historical_2_0": {
                "success": float(df_detail[df_detail["experiment"] == "C0_historical_2_0"].iloc[0]["strict_success_rate"]),
                "median_steps": float(df_detail[df_detail["experiment"] == "C0_historical_2_0"].iloc[0]["steps_to_success_median"]),
                "mean_action_delta": float(df_detail[df_detail["experiment"] == "C0_historical_2_0"].iloc[0]["mean_action_delta"])
            }
        }
    }

    verdict_file = out_dir / "final_b2_selection_verdict.json"
    with open(verdict_file, "w", encoding="utf-8") as f:
        json.dump(verdict_summary, f, indent=2)

    print(f"\n>>> [OK] Veredicto completo exportado a: {verdict_file.relative_to(REPO_ROOT)}")
    return verdict_summary

if __name__ == "__main__":
    run_stage_b12_final_eval()
