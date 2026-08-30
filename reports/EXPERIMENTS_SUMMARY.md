# 📊 Resumen Comparativo de Experimentos y Rendimiento

Esta tabla consolida las métricas de rendimiento reales obtenidas desde los logs de entrenamiento de cada modelo.

| Experimento           | Autor    | Recompensa       |   Max Speed |   Radio Meta |      LR |   n_steps |   Timesteps | Recompensa Máx   |   Recompensa Final |   Pasos/Episodio | Fecha               |
|:----------------------|:---------|:-----------------|------------:|-------------:|--------:|----------:|------------:|:-----------------|-------------------:|-----------------:|:--------------------|
| exp1_baseline         | Equipo   | distance_only    |         2   |         0.08 | 0.0003  |      1024 |      150000 | -124.97 *        |            -137.12 |            144.6 | 2026-08-30 11:20:09 |
| exp2_smooth_precision | Uli/Tomi | smooth_precision |         1.5 |         0.06 | 0.00015 |      2048 |      300000 | -36.51 *         |             -84.19 |            132.9 | 2026-08-30 12:12:33 |

### 💡 Guía de Interpretación de Métricas:
- **Recompensa Máx / Final**: Cuanto más alta o cercana a valores positivos, mejor aprendió el robot a alcanzar el objetivo.
- **Pasos/Episodio (Promedio)**: Comienza en el límite (200 pasos). Un valor bajo (ej. 30-60 pasos) indica que el robot llega rápidamente a la meta sin dar vueltas.
- **V. Máx / Radio Meta**: Parámetros de dificultad y precisión física configurados.

*Actualizado automáticamente con `python compare_experiments.py`*
