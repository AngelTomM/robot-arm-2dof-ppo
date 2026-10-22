"""
PUNTO ÚNICO DE GENERACIÓN DE FIGURAS (FASE D LQR)
=================================================
Despachador de generadores de figuras para Fase D.

USO
===
  python scripts/generate_figures.py --help

  # Comparativa entre agentes (PPO_1 vs SAC_1):
  python scripts/generate_figures.py --phase custom \\
      --configs configs/phase_X/prueba/PPO_1.yaml,configs/phase_X/prueba/SAC_1.yaml

  # Trayectorias cartesianas con ventana de estabilización k=20:
  python scripts/generate_figures.py --phase trajectories
"""
import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PHASE_CHOICES = ["custom", "trajectories"]

ENTRY_POINTS = {
    "custom": "src.visualization.phase_custom",
    "trajectories": "src.visualization.trajectories",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate_figures.py",
        description="Punto único de generación de figuras (Fase D LQR).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase", default="custom", choices=PHASE_CHOICES,
                        help="Tipo de figuras a generar (custom o trajectories). Por defecto: custom.")
    parser.add_argument("--out-dir", default=None,
                        help="Ruta de salida alternativa.")
    parser.add_argument("--configs", default=None,
                        help="[custom] YAMLs de train.py separados por comas.")
    parser.add_argument("--experiments", default=None,
                        help="[custom] Nombres de experimento.")
    parser.add_argument("--seeds", default=None,
                        help="[custom] Semillas separadas por comas.")
    parser.add_argument("--figures", default=None,
                        help="[custom] Subconjunto de figuras separadas por comas.")
    parser.add_argument("--exps-dir", default=None,
                        help="[custom] Raíz alternativa de experimentos.")
    parser.add_argument("--skip-checkpoint-eval", action="store_true", default=False,
                        help="[custom] Omite la evaluación de checkpoints.")
    parser.add_argument("--refresh-checkpoint-eval", action="store_true", default=False,
                        help="[custom] Reevalúa los checkpoints aunque exista el CSV.")
    args = parser.parse_args()

    module = importlib.import_module(ENTRY_POINTS[args.phase])
    if args.phase == "custom":
        if not args.configs:
            # Si no se pasan configs, usar los 2 configs vigentes de Phase D por defecto
            default_cfgs = "configs/phase_X/prueba/PPO_1.yaml,configs/phase_X/prueba/SAC_1.yaml"
            args.configs = default_cfgs
        module.run(
            out_dir=args.out_dir,
            configs=[s.strip() for s in args.configs.split(",")],
            experiments=([s.strip() for s in args.experiments.split(",")]
                         if args.experiments else None),
            seeds=([int(s) for s in args.seeds.split(",")] if args.seeds else None),
            figures=([int(f) for f in args.figures.split(",")] if args.figures else None),
            exps_dir=args.exps_dir,
            skip_checkpoint_eval=args.skip_checkpoint_eval,
            refresh_checkpoint_eval=args.refresh_checkpoint_eval)
    else:
        module.run(args.out_dir)


if __name__ == "__main__":
    main()

