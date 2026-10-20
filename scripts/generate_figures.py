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

  # Modelo recién entrenado (ejemplo: corrida corta de una config Fase C):
  python train.py --config configs/phase_c/S1_smooth_200.yaml --total_timesteps 8192
  python scripts/evaluate_checkpoints.py --phase c  # o runner biblioteca -> CSVs
  python scripts/generate_figures.py --phase c --data-dir <csvs> --experiments S1_smooth_200 --out-dir <figs>

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
  --phase      (obligatorio) una de las fases listadas arriba.
  --out-dir    (opcional) ruta de salida alternativa. Cada módulo la
               interpreta sobre su ruta histórica; ver el docstring del
               módulo correspondiente en src/visualization/.
  --experiments (solo b|c) modelos a dibujar, separados por comas; sin
               la bandera se dibujan los históricos de cada fase.
  --seeds      (solo b|c) semillas separadas por comas.
  --exps-dir   (solo b|c) raíz alternativa de corridas para el
               TensorBoard (b: figs 2/8; c: figura 2 SAC).
  --data-dir   (solo b|c) directorio alternativo de los CSV de evaluación
               (mismo esquema de columnas histórico).
  --figures    (solo b|c) subconjunto de figuras, separadas por comas.

Las fases a, a_agg, trajectories, b_ext y c_s3 son monográficas: no
aceptan banderas de selección (parser.error si se aportan).
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
               "  python scripts/generate_figures.py --phase a_agg --out-dir tmp/figs\n"
               "  python scripts/generate_figures.py --phase c --experiments S1_smooth_200 --figures 1,3\n")
    parser.add_argument("--phase", required=True, choices=PHASE_CHOICES,
                        help="Fase/Dataset cuyas figuras se generan.")
    parser.add_argument("--out-dir", default=None,
                        help="Ruta de salida alternativa (ver docstring del módulo).")
    parser.add_argument("--experiments", default=None,
                        help="[b|c] Modelos a dibujar separados por comas "
                             "(columna 'experiment' de los CSV o nombre del directorio de corrida).")
    parser.add_argument("--seeds", default=None,
                        help="[b|c] Semillas separadas por comas (por defecto 42,100,2026 en Fase B).")
    parser.add_argument("--exps-dir", default=None,
                        help="[b|c] Raíz alternativa de corridas para el TensorBoard "
                             "(b: figuras 2/8; c: figura 2; p. ej. experiments/phase_X/<exp>).")
    parser.add_argument("--data-dir", default=None,
                        help="[b|c] Directorio alternativo de los CSV de evaluación "
                             "(mismo esquema histórico de columnas).")
    parser.add_argument("--figures", default=None,
                        help="[b|c] Subconjunto de figuras separadas por comas (p. ej. '1,3').")
    args = parser.parse_args()

    selection_args = {"experiments": args.experiments, "seeds": args.seeds,
                      "exps_dir": args.exps_dir, "data_dir": args.data_dir, "figures": args.figures}
    selectable_phase = args.phase in ("b", "c")
    if any(v is not None for v in selection_args.values()) and not selectable_phase:
        parser.error("las banderas de selección (--experiments, --seeds, --exps-dir, "
                     "--data-dir, --figures) solo aplican a --phase b o c")

    module = importlib.import_module(ENTRY_POINTS[args.phase])
    if selectable_phase:
        experiments = ([s.strip() for s in args.experiments.split(",")]
                       if args.experiments else None)
        seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else None)
        figures = ([int(f) for f in args.figures.split(",")] if args.figures else None)
        kwargs = dict(out_dir=args.out_dir, experiments=experiments, figures=figures)
        if args.phase == "b":
            kwargs.update(seeds=seeds, exps_dir=args.exps_dir, data_dir=args.data_dir)
        else:
            kwargs.update(data_dir=args.data_dir, sac_exp_dir=args.exps_dir)
        module.run(**kwargs)
    else:
        module.run(args.out_dir)


if __name__ == "__main__":
    main()
