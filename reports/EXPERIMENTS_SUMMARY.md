# 📊 Comparación de Experimentos

Resumen consolidado de parámetros de configuración y métricas de desempeño por experimento.

| Experimento           | Autor    | Recompensa       |   Max Speed |   Radio Meta |      LR |   n_steps |   Timesteps | Recompensa Máx   |   Pasos/Episodio | Fecha               |
|:----------------------|:---------|:-----------------|------------:|-------------:|--------:|----------:|------------:|:-----------------|-----------------:|:--------------------|
| exp1_baseline         | Equipo   | distance_only    |         2   |         0.08 | 0.0003  |      1024 |      150000 | -124.97 *        |            144.6 | 2026-08-30 11:20:09 |
| exp2_baseline         | Tomi     | distance_only    |         2   |         0.08 | 0.00015 |      2048 |      300000 | -70.47 *         |            121.6 | 2026-08-30 12:58:12 |
| exp2_smooth_precision | Uli/Tomi | smooth_precision |         1.5 |         0.06 | 0.00015 |      2048 |      300000 | -36.51 *         |            132.9 | 2026-08-30 12:12:33 |

---
### 💡 Interpretación de Métricas:
- **Recompensa Máx**: Mejor retorno medio obtenido durante la evaluación del modelo (valores más cercanos a 0 o positivos indican mejor desempeño).
- **Pasos/Episodio**: Duración media del episodio (un valor menor indica resolución más rápida de la tarea).
