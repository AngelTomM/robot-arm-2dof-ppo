"""Paquete de visualización del proyecto (Robot Arm 2 DOF).

Módulos de figuras por fase, invocados a través del punto único
``scripts/generate_figures.py``:

======================  =====================================================
Módulo                  Contenido
======================  =====================================================
``common``              Estilo, paletas y guardado con metadatos JSON.
``phase_a``             Figuras históricas de Fase A (por experimento).
``phase_a_agg``         Auditoría agregada multi-semilla de Fase A (V01–V18).
``trajectories``        Trayectorias cartesianas reales (simulación en vivo).
``phase_b``             8 figuras de Fase B con trazabilidad.
``phase_b_ext``         Figuras de la extensión de Fase B.
``phase_c``             6 figuras de Fase C (SAC).
``phase_c_s3``          4 figuras del análisis de cierre S3.
======================  =====================================================

Uso:
    python scripts/generate_figures.py --phase b
    python scripts/generate_figures.py --phase a_agg --out-dir <ruta>
"""
