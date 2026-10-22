"""
PUNTO ÚNICO DE EVALUACIÓN DE CHECKPOINTS (FASE D LQR)
======================================================
Despachador de evaluación de checkpoints para Fase D (MDP Continuo con LQR)
utilizando evaluación dual desacoplada:
  - Opción 2 (Principal): Estabilización en los últimos k=20 pasos (d_20 <= 0.06m, v_20 <= 0.10rad/s).
  - Opción 3 (Comparativa Canónica): Costo integral LQR canónico J_LQR.

USO
===
  python scripts/evaluate_checkpoints.py --help

  # Evaluar una corrida individual:
  python scripts/evaluate_checkpoints.py --exp_dir experiments/phase_X/prueba/PPO_1/seed_42 \\
      [--config configs/phase_X/prueba/PPO_1.yaml] [--output_csv reports/data/phase_X/eval_ppo.csv]

  # Evaluar suite completa Fase D (PPO_1, SAC_1, C0):
  python scripts/evaluate_checkpoints.py --phase d
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.checkpoints import (
    evaluate_experiment_checkpoints,
    run_phase_d_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evaluate_checkpoints.py",
        description="Evaluación de checkpoints Fase D (Métricas Duales: Opción 2 + Opción 3 Canónica).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase", type=str, default="d", choices=["d", "b", "c"],
                        help="Fase de evaluación (por defecto: 'd').")
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="Directorio de la corrida (experiments/.../seed_<seed>).")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML de configuración del experimento.")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="CSV de salida (opcional).")
    parser.add_argument("--dataset", type=str,
                        default="data/datasets/validation_set_200.json",
                        help="JSON del conjunto de validación.")
    parser.add_argument("--output-dir", type=str, default="reports/data/phase_X",
                        help="Directorio de salida para evaluación agregada.")
    parser.add_argument("--mode", type=str, default="deterministic",
                        choices=["deterministic", "stochastic"],
                        help="Modo de evaluación.")
    args = parser.parse_args()

    if args.exp_dir:
        evaluate_experiment_checkpoints(
            exp_dir=Path(args.exp_dir),
            cfg_path=Path(args.config) if args.config else None,
            output_csv=Path(args.output_csv) if args.output_csv else None,
            dataset_path=args.dataset,
        )
    else:
        run_phase_d_evaluation(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            eval_mode=args.mode,
        )


if __name__ == "__main__":
    main()

