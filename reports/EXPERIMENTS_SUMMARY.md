# 📊 Comparación Integral de Modelos y Descubrimientos — Brazo Robótico 2 DOF

**Fecha de Actualización:** 2026-10-12  
**Alcance del Proyecto:** Síntesis consolidada de Fase Histórica, Fase A, Fase B y Fase B (Extensión — Presupuesto Unificado 655 360 Pasos).  
**Protocolo Estricto de Evaluación:** Hold durante 20 pasos consecutivos ($1.0\text{ s}$) con velocidad angular conjunta $\|\dot{\boldsymbol{\theta}}\| \le 0.10\text{ rad/s}$ dentro de $r_{tgt} = 0.06\text{ m}$ antes de 200 pasos.

---

## 1. 🏆 Ranking Definitivo de Rendimiento Físico (Lectura Terminal en Final B2)

Los resultados corresponden a la evaluación determinista terminal sobre el conjunto final sellado ([`data/datasets/phase_b_ext/final_b2_500.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/data/datasets/phase_b_ext/final_b2_500.json), $n=500$ episodios por corrida, 1 500 episodios para agentes con $n=3$ semillas), ejecutada con el presupuesto unificado de **655 360 timesteps** y checkpoints seleccionados formalmente por la Regla 4.1:

| # | Agente / Controlador | Tipo / Configuración Clave | Éxito Estricto Medio | Desv. Semillas ($s_R$) | Pasos al Éxito (Mediana) | Suavidad $\overline{\Delta a}$ | Distancia Final (m) | Estado Formal (Regla 4.2) |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **`C0` (Consistente)** | Jacobiano analítico ($K_p=3.0, v_{max}=1.5$) | **100.0 %** | — | **59.0** | 0.0508 | **0.0009 m** | Benchmark analítico |
| 🥇 | **`C0` (Histórico)** | Jacobiano analítico ($K_p=3.0, v_{max}=2.0$) | **100.0 %** | — | 63.0 | 0.0435 | 0.0028 m | Benchmark analítico |
| ⭐ | **`X4`** | PPO batch 128 + red $256^2$ + VecNorm + $X3$ | **96.73 %** | 2.81 % | 59.7 | 0.0464 | 0.0193 m | **Grupo Equivalente (Líder)** |
| ⭐ | **`X3`** | Red $256^2$ + VecNorm + cuadrática + $R_{near}$ | **94.53 %** | **0.83 %** | **57.7** | 0.0422 | 0.0288 m | **Grupo Equivalente** |
| ⭐ | **`X5`** | Idem X3 + penalización vel. $R_{vel}$ gaussiana | **93.93 %** | 4.37 % | 60.0 | **0.0392** | 0.0273 m | **Grupo Equivalente** |
| 4 | **`X6`** | Idem X5 + penalización aceleración $w_a=0.01$ | **92.67 %** | 3.37 % | 59.3 | 0.0387 | 0.0305 m | A 0.91 pp del umbral |
| 5 | **`X2b`** *(Piloto $n=1$)* | Red pequeña $64\times 64$ Tanh + **`VecNormalize`** | **85.00 %** | — | 67.0 | 0.0373 | 0.0432 m | Informativo (rescate Tanh) |
| 6 | **`X5_near`** | Idem X5 pero **sin** término positivo $R_{near}$ | **79.60 %** | 7.63 % | 67.2 | 0.0289 | 0.0455 m | Degradación por falta de $R_{near}$ |
| 7 | **`X1a`** | Red $256^2$ ReLU base sin `VecNormalize` | **77.00 %** | 5.30 % | 71.5 | 0.0318 | 0.0653 m | No convergió a 655k |
| 8 | **`X2`** | Red $256^2$ ReLU base + `VecNormalize` | **74.73 %** | 1.81 % | 69.3 | 0.0311 | 0.0651 m | No convergió a 655k |
| 9 | **`X1b`** *(Piloto $n=1$)* | Red $64\times 64$ Tanh base sin `VecNormalize` | **71.20 %** | — | 71.0 | 0.0298 | 0.0671 m | Informativo |
| 10 | **`X1c`** *(Piloto $n=1$)* | Red $128\times 128$ ReLU base sin `VecNormalize` | **71.20 %** | — | 71.0 | 0.0298 | 0.0671 m | Informativo |
| 11 | **`X1s`** *(Piloto $n=1$)* | Red $256^2$ base con entropía alta ($\text{ent}=0.01$) | **47.00 %** | — | 86.0 | 0.0160 | 0.1220 m | Ruptura por sobre-exploración |

---

## 2. 🔬 Descubrimientos Fundamentales del Proyecto

### 1. Resolución del Enigma de X3 (Fase A vs Fase B)
* **El hallazgo en Fase A:** En la Fase A se reportó erróneamente que `X3` apenas alcanzaba un 44.1 % de éxito frente al 70.8 % de `X5_near`.
* **El descubrimiento en Fase B:** Aquel resultado se debió a un **artefacto procedimental de guardado de checkpoints**. El callback de Fase A guardó los pesos de la red `best_model.zip` en los pasos 120k–160k, pero el archivo `vecnormalize_stats.pkl` fue sobreescrito al paso 300k. Al evaluar esa red con estadísticas cinemáticas desfasadas, la política colapsó.
* **La realidad física:** Cuando se evalúa `X3` con su par acoplado atómico de `VecNormalize`, alcanza **94.53 %** en Final B2 a 286k pasos y supera el **98.0 %** a 410k pasos.

### 2. El Factor Más Determinante: Densificación de Recompensa Cuadrática + $R_{near}$ ($X2 \to X3$)
* Aislando este único factor (red y normalización idénticas), la recompensa cuadrática con bonificación de proximidad generó un salto de **$+19.80$ pp en tasa de éxito** (de 74.73 % a 94.53 %) y redujo en **$-11.7$ pasos** el tiempo de estabilización.
* Al suprimir el término positivo de $R_{near}$ en `X5_near`, el rendimiento cae inmediatamente en **$-14.93$ pp** (a 79.60 %), demostrando que el brazo necesita una densificación geométrica explícita para aprender a desacelerar antes de entrar a la zona de tolerancia.

### 3. Veredicto Formal de la Fase B (Extensión): Grupo de Equivalentes ['X4', 'X3', 'X5']
Bajo la regla formal congelada antes del entrenamiento (anchura Wilson de ruido $U_{equiv\_final} = 3.16$ pp):
* **`X4` lidera en tasa bruta (96.73 %):** La duplicación del tamaño de lote a 128 estabilizó las estimaciones del gradiente PPO, llevando la política a un techo del 99.4 % en su mejor semilla, aunque requirió más pasos para converger (paso 450k).
* **`X3` lidera en robustez y velocidad (94.53 %):** Presenta la menor dispersión entre semillas de todo el estudio ($s_R = \mathbf{0.83\,\%}$) y el tiempo de llegada más rápido (**57.7 pasos**, superando los 59 pasos de $C0$), convergiendo a solo 286k pasos.
* **`X5` lidera en suavidad cinemática (93.93 %):** La penalización de velocidad gaussiana modela eficazmente el frenado fino ($\overline{\Delta a} = \mathbf{0.0392}$ rad/s) sin sacrificar la tasa de éxito dentro del margen estadístico.

### 4. El Rescate de Redes Tanh Pequeñas (`X2b`)
* En la Fase A se hipotetizó que las redes $64\times 64$ Tanh fallaban debido a saturación de gradientes.
* En la extensión, el piloto **`X2b`** demostró que al incorporar `VecNormalize(obs)` a esa misma arquitectura de 64 neuronas, el éxito se dispara de **71.20 % a 85.00 %** ($+13.8$ pp). La normalización de las observaciones alivia la saturación en la primera capa.

### 5. Colapso por Entropía Excesiva (`X1s`)
* Elevar el coeficiente de entropía de $0.002$ a $0.01$ en PPO provocó que el agente `X1s` colapsara al **47.00 %** de éxito, sufriendo un 53.0 % de truncamientos. Para tareas de posicionamiento de precisión milimétrica, una varianza estocástica forzada impide la convergencia asintótica requerida para cumplir el criterio de hold durante 20 pasos.

---

## 3. ⚖️ Tabla de Comparación de Pares de Un Solo Factor Adyacentes

Para evitar inferencias indirectas, se evaluaron las diferencias estrictas entre configuraciones contiguas en su árbol genealógico sobre el conjunto final:

| Par Adyacente | Factor Aislado | Impacto en Éxito | Impacto en Pasos | Impacto en Suavidad ($\overline{\Delta a}$) | Conclusión |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **X1a $\to$ X2** | `VecNormalize(obs)` sobre red $256^2$ | $-2.27\text{ pp}$ | $-2.2\text{ pasos}$ | $-0.0007$ | La normalización por sí sola no compensa una recompensa lineal sin guía espacial a 655k pasos. |
| **X2 $\to$ X3** | Recompensa cuadrática + $R_{near}$ | **$+19.80\text{ pp}$** | **$-11.7\text{ pasos}$** | $+0.0110$ | **Factor de mayor impacto:** transforma una política insuficiente en casi óptima. |
| **X3 $\to$ X4** | Batch size PPO ($64 \to 128$) | $+2.20\text{ pp}$ | $+2.0\text{ pasos}$ | $+0.0042$ | Mayor estabilidad del gradiente PPO que eleva el techo asintótico requiriendo más timesteps. |
| **X3 $\to$ X5_near**| Supresión de $R_{near}$ positivo | **$-14.93\text{ pp}$** | **$+9.5\text{ pasos}$** | $-0.0133$ | La ausencia de proximidad degrada severamente el frenado y la retención estricta. |
| **X3 $\to$ X5** | $R_{vel}$ gaussiana ($w_v=0.10, \sigma_v=0.15$) | $-0.60\text{ pp}$ | $+2.3\text{ pasos}$ | **$-0.0030$** | Modela frenado suave en el efector final a costo despreciable en éxito estricto. |
| **X5 $\to$ X6** | Penalización aceleración $w_a=0.01$ | $-1.27\text{ pp}$ | $-0.7\text{ pasos}$ | $-0.0005$ | Sobre-penalización: ganancia nula en suavidad sobre X5 y reducción marginal en éxito. |

---

## 4. 📁 Trazabilidad de Reportes y Datos en el Repositorio

* **Informe General Integrado de Fase B:** [`reports/phase_b_report.md`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/phase_b_report.md) (Secciones 1 a 13 originales + Sección 14 Addenda de la Extensión).
* **Veredicto Formal Estructurado:** [`reports/data/phase_b/extension/final_b2_selection_verdict.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/extension/final_b2_selection_verdict.json).
* **Evaluaciones en Conjunto Final:** [`reports/data/phase_b/extension/final_b2_eval_details.csv`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/extension/final_b2_eval_details.csv) y [`final_b2_eval_aggregated.csv`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/extension/final_b2_eval_aggregated.csv).
* **Galería de Figuras a 300 dpi con Trazabilidad JSON:** [`reports/figures/phase_b/extension/`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/figures/phase_b/extension/).
