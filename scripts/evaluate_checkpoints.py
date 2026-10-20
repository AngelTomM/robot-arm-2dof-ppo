"""
PUNTO ÚNICO DE EVALUACIÓN DE CHECKPOINTS
=========================================
Despachador de los runners de evaluación de checkpoints (Fases B y C).
La lógica vive en src/evaluation/checkpoints.py; los parámetros
configurables y los datasets requeridos están documentados en su
docstring. La Extended de Fase B (B/ext) se orquesta con sus propios
scripts batch (phase_b_ext_batch_eval_validation.py,
phase_b_ext_run_b12_final_eval.py), que consumen el mismo módulo.

USO
===
  python scripts/evaluate_checkpoints.py --help

  # Fase B: todos los checkpoints de una corrida sobre 'validation' (200)
  python scripts/evaluate_checkpoints.py --phase b \\
      --exp_dir experiments/phase_b/X5_near/seed_42 \\
      --config configs/phase_b/X5_near.yaml \\
      [--output_csv reports/data/phase_b/mi_salida.csv]

  # Fase C: S1 (SAC) + X3 (PPO) + C0 sobre 'validation_c' (500)
  python scripts/evaluate_checkpoints.py --phase c \\
      [--dataset data/datasets/phase_c/validation_c_500.json] \\
      [--output-dir reports/data/phase_c] \\
      [--mode deterministic|stochastic] [--max-step 655360]

PARÁMETROS GLOBALES
===================
  --phase       (obligatorio) 'b' o 'c'.
  Fase B:  --exp_dir / --config / --output_csv (idénticos al CLI antiguo de
           Fase B; --output_csv es opcional).
  Fase C:  --dataset / --output-dir / --mode / --max-step.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.checkpoints import (
    evaluate_experiment_checkpoints,
    run_phase_c_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evaluate_checkpoints.py",
        description="Punto único de evaluación de checkpoints (Fases B y C).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Ejemplos:\n"
               "  python scripts/evaluate_checkpoints.py --phase b --exp_dir experiments/phase_b/X5_near/seed_42 --config configs/phase_b/X5_near.yaml\n"
               "  python scripts/evaluate_checkpoints.py --phase c --mode stochastic\n")
    parser.add_argument("--phase", required=True, choices=["b", "c"],
                        help="Fase de evaluación (b: corrida individual; c: S1+X3+C0).")
    # Fase B
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="[b] Directorio de la corrida (experiments/.../seed_<seed>).")
    parser.add_argument("--config", type=str, default=None,
                        help="[b] YAML de configuración de la fase.")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="[b] CSV de salida (opcional).")
    # Fase C
    parser.add_argument("--dataset", type=str,
                        default="data/datasets/phase_c/validation_c_500.json",
                        help="[c] JSON del conjunto de validación.")
    parser.add_argument("--output-dir", type=str, default="reports/data/phase_c",
                        help="[c] Directorio de los CSV de salida.")
    parser.add_argument("--mode", type=str, default="deterministic",
                        choices=["deterministic", "stochastic"],
                        help="[c] Modo de evaluación.")
    parser.add_argument("--max-step", type=int, default=655360,
                        help="[c] Presupuesto máximo de pasos a evaluar.")
    args = parser.parse_args()

    if args.phase == "b":
        if not args.exp_dir or not args.config:
            parser.error("la fase 'b' requiere --exp_dir y --config")
        evaluate_experiment_checkpoints(
            exp_dir=Path(args.exp_dir),
            cfg_path=Path(args.config),
            output_csv=Path(args.output_csv) if args.output_csv else None
        )
    else:
        run_phase_c_evaluation(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            max_step=args.max_step,
            eval_mode=args.mode,
        )


if __name__ == "__main__":
    main()
