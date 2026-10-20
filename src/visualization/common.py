"""
Infraestructura compartida de generación de figuras (edición de estilo,
guardado a 300 dpi con metadatos JSON de trazabilidad y clasificación de
episodios).

USO
===
Módulo de uso interno por los generadores de ``src/visualization``; no se
invoca directamente desde la línea de comandos. Los generadores importan:

    from src.visualization.common import (
        apply_style, save_fig_with_metadata, classify_episodes,
    )

PARÁMETROS CONFIGURABLES
========================
apply_style()
    Aplica la configuración estética común a las figuras del proyecto:
    estilo ``seaborn-v0_8-whitegrid`` (si está disponible), fuente
    ``DejaVu Sans``/``Arial``, borde de ejes ``#cccccc`` y grosor 0.8.
    No acepta parámetros: la estética es idéntica en todas las fases.

save_fig_with_metadata(fig, out_png_path, input_files=None, description="",
                       phase=None, metadata=None, verbose=False)
    fig           : ``matplotlib.figure.Figure`` a guardar (300 dpi).
    out_png_path  : ``Path`` del PNG de salida; junto a él se escribe el
                    JSON de trazabilidad con el mismo nombre base.
    input_files   : lista de rutas de los archivos de entrada cuyas huellas
                    SHA-256 se registran; los inexistentes se omiten.
                    Valores posibles: ``None`` (sin hashes, modo metadatos
                    libres) o lista de ``str``/``Path``.
    description   : texto descriptivo de la figura (campo ``description``).
    phase         : etiqueta corta de fase incluida en el JSON
                    (p. ej. ``"B"``). ``None`` omite el campo, tal como en
                    el pipeline agregado de Fase A.
    metadata      : ``dict`` con metadatos libres (variantes de extensión y
                    Fase C). Si se aporta, prevalece sobre los campos
                    anteriores y se serializa con conversión de tipos
                    numpy (``np.integer``→int, ``np.floating``→float,
                    ``np.ndarray``→list).
    verbose       : ``True`` imprime ``">>> [GUARDADO] <nombre>"`` (Fase B);
                    ``False`` (por defecto) no imprime.

classify_episodes(df, column="category")
    Clasificación jerárquica de desenlaces definida en la Fase A
    (Sección 10.1); ver las cadenas de categorías en el cuerpo. Además
    asigna ``"No clasificable"`` cuando las banderas requeridas son NaN.
    column    : nombre de la columna resultante, ``"outcome_category"``
                por defecto (nombre compartido con los generadores
                históricos de Fase A agregada y Fase B).
    Devuelve una copia del DataFrame con la columna indicada añadida.
"""
import hashlib
import inspect
import json
import platform
import sys
import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Raíz del repositorio (src/visualization/common.py → 2 niveles arriba)
REPO_ROOT = Path(__file__).resolve().parents[2]


def apply_style():
    """Configuración estética global compartida por todos los generadores."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def _numpy_default_conv(o):
    """Conversor de tipos numpy para json.dump (admite ints, floats y arrays)."""
    if isinstance(o, (np.integer, np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.floating, np.float64, np.float32)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_fig_with_metadata(fig, out_png_path, input_files=None, description="",
                           phase=None, metadata=None, verbose=False):
    """Guarda figura a 300 dpi junto con un JSON de trazabilidad completa.

    Compatibiliza los tres esquemas vigentes del repositorio:
    1. Fase B / Fase A agregada (hint por campos ``description``/``phase``).
    2. Extensión y Fase C (hint por ``metadata`` dict libre).
    El script de procedencia registrado en el JSON es el módulo llamante.
    """
    out_png_path = Path(out_png_path)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    json_path = out_png_path.with_suffix(".json")

    if metadata is not None:
        # Esquema de metadatos libres (extensión de Fase B / Fase C / S3).
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=_numpy_default_conv)
        print(f"  [FIG OK] Guardada: {out_png_path.name}")
        return

    # Esquema canónico Fase B / Fase A agregada.
    file_hashes = {}
    for file in input_files or []:
        p = Path(file)
        if p.exists() and p.is_file():
            file_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

    caller = Path(inspect.stack()[1].filename).resolve()
    script_hash = hashlib.sha256(caller.read_bytes()).hexdigest() if caller.exists() else "N/A"

    meta = {
        "figure_name": out_png_path.name,
        "description": description,
    }
    if phase is not None:
        meta["phase"] = phase
    meta.update({
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "os_platform": platform.platform(),
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "script_path": str(caller.relative_to(REPO_ROOT)),
        "script_sha256": script_hash,
        "input_files_sha256": file_hashes,
    })

    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    if verbose:
        print(f">>> [GUARDADO] {out_png_path.name} y metadatos JSON.")


# ciclo determinista para experimentos/agentes sin color canónico (p. ej.
# un modelo recién entrenado): orden de aparición, estable por ejecución
DEFAULT_COLOR_CYCLE = [
    "#e67e22", "#2980b9", "#27ae60", "#c0392b",
    "#8e44ad", "#16a085", "#f1c40f", "#7f8c8d",
]


def color_resolver(base_palette: dict, extra_cycle: list = None):
    """
    Construye una función ``f(nombre) -> color`` que resuelve la estética
    compartida: los nombres canónicos usan su color histórico del
    ``base_palette`` y los desconocidos (modelos nuevos no presentes en la
    monografía) reciben por orden de primera aparición un color del ciclo
    por defecto (estable dentro de la misma ejecución).
    """
    cycle = list(extra_cycle) if extra_cycle else list(DEFAULT_COLOR_CYCLE)
    assigned: dict = {}

    def resolver(name: str) -> str:
        if name in base_palette:
            return base_palette[name]
        if name not in assigned:
            assigned[name] = cycle[len(assigned) % len(cycle)]
        return assigned[name]

    return resolver


def classify_episodes(df: pd.DataFrame, column: str = "outcome_category") -> pd.DataFrame:
    """
    Aplica la clasificación jerárquica de desenlaces de la Sección 10.1:
    1. Éxito limpio: success & ~ballistic_flag & ~oscillatory_flag
    2. Éxito con patología: success & (ballistic_flag | oscillatory_flag)
    3. Fallo oscilatorio: ~success & oscillatory_flag
    4. Fallo balístico: ~success & ballistic_flag & ~oscillatory_flag
    5. Fallo sin aproximación: ~success & ~osc & ~ball & (time_inside_030 == 0)
    6. Fallo sin estabilizar: ~success & ~osc & ~ball & (time_inside_030 > 0)
    7. No clasificable: NaNs en banderas requeridas o inconsistencias.
    """
    res = df.copy()
    # Detección explícita de banderas no disponibles (NaN) sin castear a bool
    # Evita que bool(np.nan) o .astype(bool) conviertan accidentalmente NaN en True
    nan_flags = res["ballistic_flag"].isna() | res["oscillatory_flag"].isna() | res["success"].isna()

    succ = res["success"].fillna(False).astype(bool)
    ball = res["ballistic_flag"].fillna(False).astype(bool)
    osc = res["oscillatory_flag"].fillna(False).astype(bool)
    t_near = res["time_inside_030"].fillna(0).astype(int)

    cat = np.full(len(res), "No clasificable", dtype=object)

    valid_mask = ~nan_flags
    mask_cat1 = valid_mask & succ & (~ball) & (~osc)
    mask_cat2 = valid_mask & succ & (ball | osc)
    mask_cat3 = valid_mask & (~succ) & osc
    mask_cat4 = valid_mask & (~succ) & ball & (~osc)
    mask_cat5 = valid_mask & (~succ) & (~osc) & (~ball) & (t_near == 0)
    mask_cat6 = valid_mask & (~succ) & (~osc) & (~ball) & (t_near > 0)

    cat[mask_cat1] = "1. Éxito limpio"
    cat[mask_cat2] = "2. Éxito con patología"
    cat[mask_cat3] = "3. Fallo oscilatorio"
    cat[mask_cat4] = "4. Fallo balístico"
    cat[mask_cat5] = "5. Fallo sin aproximación"
    cat[mask_cat6] = "6. Fallo sin estabilizar"

    res[column] = cat
    return res
