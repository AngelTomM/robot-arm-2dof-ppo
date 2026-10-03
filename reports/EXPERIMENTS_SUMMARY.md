# 📊 Comparación Integral de Modelos — Brazo Robótico 2 DOF

Resumen consolidado de todos los modelos entrenados y evaluados en el proyecto (Fase Histórica y Fase A).

---

## 1. 🏆 Ranking de Rendimiento Físico Real (Evaluación Hold Estricto)

> **Protocolo de Evaluación Riguroso (300 Episodios de Prueba):**  
> Para declarar éxito, el efector debe permanecer dentro de $r_{tgt} = 0.06\text{ m}$ con velocidad angular $\|\dot{\boldsymbol{\theta}}\| \le 0.10\text{ rad/s}$ durante **20 pasos consecutivos** ($1.0\text{ s}$).  
> *Los modelos confirmados (`C0`, `X5_near`, `X5`, `X6`, `X3`, `X1a`, `X2`, `X0`) fueron evaluados en 3 semillas independientes (900 episodios).*

| # | Modelo | Tipo / Modificación Principal | Éxito Estricto | Truncamientos | Error Final | Pasos al Éxito | Suavidad \|\|Δa\|\| |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **`C0`** | **Jacobiano Damped Least Squares** | **100.0%** | **0.0%** | **0.001 m** | **54.0** | 0.053 |
| 🥈 | **`X5_near`** ⭐ | **Red 256x256 + VecNormalize + Sin $R_{near}$** | **70.8%** | **29.2%** | **0.050 m** | **58.0** | 0.030 |
| 🥉 | **`X5`** (`wv01`) | Penalización de velocidad gaussiana ($w_v=0.10$) | **55.3%** | 44.7% | 0.065 m | 54.5 | 0.028 |
| 4 | **`X6`** (`wa001`) | Penalización de suavidad de acción ($w_a=0.01$) | **45.6%** | 54.4% | 0.084 m | 58.5 | **0.023** |
| 5 | **`X3`** | Red $256\times 256$ ReLU + VecNormalize (Batch 64) | **44.1%** | 55.9% | 0.081 m | 60.0 | 0.023 |
| 6 | **`X2b`** | Normalización de recompensa (`norm_reward=True`) | **45.0%** | 55.0% | 0.088 m | 66.0 | 0.025 |
| 7 | **`X6_wa05`** | Suavidad alta ($w_a=0.05$, excesiva rigidez) | **27.5%** | 72.5% | 0.125 m | 68.0 | **0.018** |
| 8 | **`X6_wa01`** | Suavidad media ($w_a=0.02$) | **27.0%** | 73.0% | 0.134 m | 67.5 | 0.020 |
| 9 | **`X4`** | Batch size grande (256, menos updates) | **15.0%** | 85.0% | 0.220 m | 64.0 | 0.015 |
| 10 | **`X1a`** | Hold moderado (5 pasos, 0.3 rad/s) sin VecNorm | **8.6%** | 91.4% | 0.414 m | 63.0 | 0.013 |
| 11 | **`X2`** | VecNormalize con red pequeña $64\times 64$ Tanh | **5.9%** | 94.1% | 0.375 m | 70.0 | 0.012 |
| 12 | **`X1s`** | Hold estricto directo (20p, 0.1 rad/s) sin VecNorm | **5.0%** | 95.0% | 0.430 m | 118.5 | 0.011 |
| 13 | **`X1b`** | Hold 10 pasos sin VecNormalize | **3.5%** | 96.5% | 0.485 m | 124.0 | 0.011 |
| 14 | **`X1c`** | Hold 15 pasos sin VecNormalize | **3.0%** | 97.0% | 0.510 m | 130.0 | 0.011 |
| 15 | **`X0`** | **Modelo Histórico Original (`exp2_smooth_precision`)** | **1.0%** | **99.0%** | **0.675 m** | **92.0** | **0.011** |
| 16 | **`X5_wv10`** | Penalización de velocidad excesiva ($w_v=1.0$) | **0.5%** | 99.5% | 0.720 m | 145.0 | 0.009 |
| 17 | **`X5_wv50`** | Penalización de velocidad parálisis ($w_v=5.0$) | **0.0%** | 100.0% | 0.850 m | N/A | 0.005 |

---

## 2. 📋 Registro Completo de Entrenamientos y Parámetros

Historial de todos los entrenamientos ejecutados con sus hiperparámetros y métricas registradas en TensorBoard:

| Experimento | Autor | Tipo Recompensa | Vel. Máx | Radio | Learning Rate | Batch | Timesteps | Recompensa Máx | Pasos/Episodio | Fecha de Ejecución |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`exp2_smooth_precision`** | Uli/Tomi | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -46.80 * | 187.1 | 2026-08-30 23:31 |
| **`exp2_smooth_precisiong`** | Yo/Gabi | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -51.69 * | 179.1 | 2026-08-30 23:35 |
| **`exp1_baselineg`** | Equipo | `distance_only` | 2.0 rad/s | 0.08 m | 0.00015 | 64 | 200,000 | -99.74 * | 165.6 | 2026-08-30 23:42 |
| **`exp1_baseline`** | Equipo | `distance_only` | 2.0 rad/s | 0.08 m | 0.00030 | 64 | 150,000 | -62.41 * | 174.5 | 2026-09-03 19:42 |
| **`smoke_test`** | Fase A | `smooth_precision` | 2.0 rad/s | 0.06 m | 0.00030 | 64 | 2,048 | -74.83 * | 50.0 | 2026-10-02 19:44 |
| **`X1a (seed_42)`** | Fase A | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -56.71 * | 190.5 | 2026-10-02 19:46 |
| **`X1b (seed_42)`** | Fase A | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -56.40 * | 200.0 | 2026-10-02 19:50 |
| **`X1c (seed_42)`** | Fase A | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -55.65 * | 200.0 | 2026-10-02 19:53 |
| **`X1s (seed_42)`** | Fase A | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -109.32 * | 187.6 | 2026-10-02 19:56 |
| **`X2 (seed_42)`** | Fase A | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -39.04 * | 183.7 | 2026-10-02 20:02 |
| **`X2b (seed_42)`** | Fase A | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | 9.41 * | 104.2 | 2026-10-02 20:06 |
| **`X3 (seed_42)`** | Fase A | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | 32.08 * | 63.7 | 2026-10-02 20:09 |
| **`X4 (seed_42)`** | Fase A | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 256 | 300,000 | 9.06 * | 58.9 | 2026-10-02 20:13 |
| **`X5_wv10 (seed_42)`** | Fase A | `x5_vel` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -55.13 * | 200.0 | 2026-10-02 20:25 |
| **`X5_wv50 (seed_42)`** | Fase A | `x5_vel` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -118.15 * | 200.0 | 2026-10-02 20:29 |
| **`X6_wa001 (seed_42)`**| Fase A | `x6_smooth` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | 17.74 * | 41.9 | 2026-10-02 20:33 |
| **`X6_wa01 (seed_42)`** | Fase A | `x6_smooth` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | 34.67 * | 64.7 | 2026-10-02 20:37 |
| **`X6_wa05 (seed_42)`** | Fase A | `x6_smooth` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | 29.44 * | 64.5 | 2026-10-02 20:41 |
| **`X1a (seed_100)`** | Fase A | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -1.28 * | 147.9 | 2026-10-02 20:49 |
| **`X1a (seed_2026)`** | Fase A | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -62.57 * | 166.9 | 2026-10-02 20:52 |
| **`X2 (seed_100)`** | Fase A | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -15.13 * | 189.1 | 2026-10-02 20:56 |
| **`X2 (seed_2026)`** | Fase A | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -58.72 * | 200.0 | 2026-10-02 21:00 |
| **`X3 (seed_100)`** | Fase A | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | 19.84 * | 56.0 | 2026-10-02 21:04 |
| **`X3 (seed_2026)`** | Fase A | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | 19.36 * | 43.6 | 2026-10-02 21:08 |
| **`X5_near (seed_100)`**| Fase A | `x5_near` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -25.42 * | 140.8 | 2026-10-02 21:12 |
| **`X5_near (seed_2026)`**| Fase A | `x5_near` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -19.70 * | 99.4 | 2026-10-02 21:16 |
| **`X5 (seed_42)`** | Fase A | `x5_vel` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | 17.91 * | 78.1 | 2026-10-02 21:20 |
| **`X5 (seed_100)`** | Fase A | `x5_vel` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | 16.66 * | 49.9 | 2026-10-02 21:31 |
| **`X5 (seed_2026)`** | Fase A | `x5_vel` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | 19.13 * | 44.7 | 2026-10-02 21:35 |
| **`X6 (seed_100)`** | Fase A | `x6_smooth` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | 22.89 * | 54.5 | 2026-10-02 21:40 |
| **`X6 (seed_2026)`** | Fase A | `x6_smooth` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | 21.04 * | 52.1 | 2026-10-02 21:44 |
| **`X5_near (seed_42)`** | Fase A | `x5_near` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -13.64 * | 68.9 | 2026-10-02 22:37 |
| **`exp2_smooth_precision (seed_42)`** | Uli/Tomi | `smooth_precision` | 1.5 rad/s | 0.06 m | 0.00015 | 64 | 300,000 | -30.94 * | 186.9 | 2026-10-02 22:46 |

---

## 3. 💡 Claves para Interpretar los Resultados

1. **La Recompensa Máx no define al mejor modelo:** Modelos como `X1a (seed_100)` obtuvieron `-1.28` porque su función de recompensa regalaba puntos por orbitar cerca del objetivo ($R_{near}$). Al quitar ese término viciado en `X5_near`, el modelo aprende a frenar en serio y su tasa de éxito físico sube al **70.8%**.
2. **Métricas Físicas Reales (Tabla 1):**
   - **Éxito Estricto (%):** Mide la capacidad real de frenar y sostener la posición durante 1 segundo ($20\text{ pasos}$) con $v \le 0.10\text{ rad/s}$.
   - **Truncamientos (%):** Episodios donde el brazo se quedó sin tiempo (200 pasos) sin estabilizarse.
   - **Error Final:** Distancia cartesiana al centro de la meta ($< 0.06\text{ m}$ indica precisión dentro del radio).
   - **Pasos al Éxito:** Tiempo que tarda en completar la tarea (menos pasos = más rápido).
3. **Modelos Destacados:**
   - 🥇 **`C0` (Jacobiano Clásico):** 100% de éxito y error de $0.001\text{ m}$ (Benchmark y Cota Superior).
   - 🥈 **`X5_near` (RL PPO):** 70.8% de éxito real y 29.2% de truncamientos (El mejor agente por refuerzo).
