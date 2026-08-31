# 📊 Comparación de Experimentos (PPO / SAC)

Resumen consolidado de parámetros y métricas de desempeño (Ordenado por: **REWARD**).

| Experimento      | Algoritmo   | Autor   | Recompensa       |   Max Speed |   Radio Meta |     LR | Config Detalle   |   Timesteps | Recompensa Máx   |   Pasos/Episodio | Fecha               |
|:-----------------|:------------|:--------|:-----------------|------------:|-------------:|-------:|:-----------------|------------:|:-----------------|-----------------:|:--------------------|
| exp_sac_baseline | SAC         | Tomi    | smooth_precision |         1.5 |         0.06 | 0.0003 | buf:100000       |      120000 | 43.79 *          |            103   | 2026-08-31 10:44:37 |
| exp1_baseline    | PPO         | Tomi    | distance_only    |         2   |         0.08 | 0.0003 | n_steps:1024     |      150000 | -89.32 *         |            123.4 | 2026-08-31 00:04:55 |

---
### 💡 Interpretación de Métricas:
- **Algoritmo**: `PPO` (On-Policy) o `SAC` (Off-Policy con Replay Buffer).
- **Recompensa Máx**: Mejor retorno medio obtenido (valores más cercanos a 0 o positivos = mejor modelo).
- **Pasos/Episodio**: Duración media del episodio (menor valor = resolución más rápida).
