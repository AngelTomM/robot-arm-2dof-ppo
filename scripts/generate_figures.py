"""
PUNTO ÚNICO DE GENERACIÓN DE FIGURAS
====================================
Despachador de todos los generadores de figuras del repositorio. La lógica
de cada fase vive en src/visualization/<modulo>.py, donde se documentan
las figuras generadas, los insumos requeridos y los parámetros
configurables de cada módulo.

USO
===
  python scripts/generate_figures.py --help            # referencia completa
  python scripts/generate_figures.py --phase b         # figuras de Fase B
  python scripts/generate_figures.py --phase a_agg --out-dir <ruta>

FASES DISPONIBLES
=================
  a             Figuras históricas de Fase A por experimento
                (reports/data/all_final_*.csv -> reports/figures/).
  a_agg         Auditoría agregada multi-semilla de Fase A con
                validaciones V01-V18 (TensorBoard de experiments/phase_a).
  trajectories  Trayectorias cartesianas reales (requiere modelos PPO
                entrenados en experiments/phase_a/*/seed_42/models/).
  b             8 figuras de Fase B (TensorBoard de experiments/phase_b).
  b_ext         Figuras de la extensión de Fase B
                (reports/data/phase_b/extension/*.csv|json).
  c             6 figuras de Fase C (SAC)
                (reports/data/phase_c/*.csv + experiments/phase_c/S1).
  c_s3          4 figuras del análisis de cierre S3_combined
                (reports/phase_c_s2_analysis y phase_c_s3_analysis).

PARÁMETROS GLOBALES
===================
  --phase    (obligatorio) una de las fases listadas arriba.
  --out-dir  (opcional) ruta de salida alternativa. Cada módulo la
             interpreta sobre su ruta histórica; ver el docstring del
             módulo correspondiente en src/visualization/.
"""
import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PHASE_CHOICES = ["a", "a_agg", "trajectories", "b", "b_ext", "c", "c_s3"]

ENTRY_POINTS = {
    "a": "src.visualization.phase_a",
    "a_agg": "src.visualization.phase_a_agg",
    "trajectories": "src.visualization.trajectories",
    "b": "src.visualization.phase_b",
    "b_ext": "src.visualization.phase_b_ext",
    "c": "src.visualization.phase_c",
    "c_s3": "src.visualization.phase_c_s3",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate_figures.py",
        description="Punto único de generación de figuras del repositorio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Ejemplos:\n"
               "  python scripts/generate_figures.py --phase b\n"
               "  python scripts/generate_figures.py --phase a_agg --out-dir tmp/figs\n")
    parser.add_argument("--phase", required=True, choices=PHASE_CHOICES,
                        help="Fase/Dataset cuyas figuras se generan.")
    parser.add_argument("--out-dir", default=None,
                        help="Ruta de salida alternativa (ver docstring del módulo).")
    args = parser.parse_args()

    module = importlib.import_module(ENTRY_POINTS[args.phase])
    module.run(args.out_dir)


if __name__ == "__main__":
    main()
