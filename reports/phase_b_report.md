# INFORME FINAL — FASE B: DIAGNÓSTICO EN PROFUNDIDAD, PRESUPUESTO EXTENDIDO Y DINÁMICA DE CHECKPOINTS EN UN BRAZO ROBÓTICO 2 DOF

**Fecha:** 2026-10-09  
**Autor:** Antigravity AI — DeepMind Agentic Pair Programming  
**Rama Git:** `fase-b-diagnostics`  
**Entorno de Ejecución:** Python 3.11.16 | PyTorch 2.13.0+cpu | Gymnasium 1.3.0 | Stable-Baselines3 2.9.0 | NumPy 1.26.4 | Pandas 3.0.5 | Matplotlib 3.11.1 | Plataforma: Windows-10-10.0.19045-SP0  

---

## 1. Resumen Ejecutivo

La **Fase B** tuvo como objetivo central realizar un diagnóstico en profundidad de la estabilidad cinemática, la reproducibilidad y el comportamiento asintótico de políticas de Aprendizaje por Refuerzo Profundo (PPO) en un brazo planar de 2 Grados de Libertad (2 DOF) bajo un presupuesto extendido a **1.000.000 de pasos de interacción**, comparando su evolución contra el controlador cinemático diferencial analítico (**`C0`**).

El protocolo mantuvo las especificaciones físicas ($L_1 = L_2 = 1.0\text{ m}$, $dt = 0.05\text{ s}$, $v_{max} = 1.5\text{ rad/s}$) y el criterio estricto de estabilización (*hold* durante 20 pasos consecutivos con $\|\dot{\boldsymbol{\theta}}\| \le 0.10\text{ rad/s}$ en $r_{tgt} = 0.06\text{ m}$). Se entrenaron dos configuraciones representativas a lo largo de 3 semillas independientes (`42`, `100`, `2026`): **`X5_near`** (red $256\times 256$ ReLU + `VecNormalize` + recompensa sin término de proximidad) y **`X3`** (red $256\times 256$ ReLU + `VecNormalize` + función de recompensa con término de proximidad $R_{near}$). Se registraron 25 checkpoints periódicos acoplados por corrida (150 checkpoints en total), evaluados sistemáticamente sobre el conjunto de validación ($n=200$) para la toma de decisiones, seguido de una lectura única terminal sobre el conjunto final ($n=300$).

### Principales Resultados Medidos y Síntesis:
1. [HECHO] **Rendimiento Máximo Determinista en Evaluación Final:**  
   En la lectura única del conjunto final ($n=300$ episodios por semilla, 900 episodios totales):
   * El controlador analítico **`C0`** alcanzó **100.0%** de éxito estricto determinista (IC Wilson 95%: [98.74%, 100.00%]), con $0.0028\text{ m}$ de distancia media y 0.0% de truncamientos.
   * El modelo **`X3`** (`ckpt_409600`) alcanzó **98.00%** de éxito estricto medio inter-semilla (std=0.67 pp; semillas: 97.33%, 98.67%, 98.00%), con una mediana de **56.0 pasos** al éxito y 2.0% de truncamientos.
   * El modelo **`X5_near`** (`ckpt_901120`) alcanzó **94.89%** de éxito estricto medio inter-semilla (std=2.04 pp; semillas: 95.33%, 92.67%, 96.67%), con una mediana de **67.0 pasos** al éxito y 5.11% de truncamientos.
2. [HECHO] **Colapso en Evaluación Estocástica:**  
   Todos los modelos PPO registraron **0.0%** de éxito estricto en modo estocástico sobre el conjunto final (0/300 éxitos en todas las semillas; IC Wilson agrupado: [0.00%, 0.43%]).
3. [HECHO] **Diferencias de Procedimiento Identificadas:**  
   * La discrepancia reportada en Fase A para `X3` (44.1%) se debió a evaluar `best_model.zip` (guardado en los pasos 120k–160k) con estadísticas de `VecNormalize` desfasadas del paso 300k. El modelo final real de Fase A a 300k pasos ya alcanzaba **94.33%** en validación.
   * La variación en llegadas balísticas de `C0` (36.3% en Fase A vs 10.7% en Fase B) se debió al cambio en el entorno de `max_speed = 2.0 rad/s` a `1.5 rad/s` y al escalado resultante de la ganancia efectiva a $K_{p,eff} = 2.25$.
4. [HECHO] **Bandera Balística Desacoplada del Fracaso:**  
   La condición de entrada balística ($\|\dot{\boldsymbol{\theta}}\| > 0.30\text{ rad/s}$) no impidió la estabilización: `X3` registró un **71.89%** de llegadas balísticas manteniendo un 98.00% de éxito estricto, mientras que `C0` registró un **10.67%** de entradas balísticas con un 100.0% de éxito.

---

## 2. Alcance y Relación con la Fase A

### 2.1 Qué se Mantiene Inalterado respecto a la Fase A
* **Física y Cinemática:** Brazo planar de 2 DOF con eslabones rígidos $L_1 = 1.0\text{ m}$, $L_2 = 1.0\text{ m}$, paso de integración $\Delta t = 0.05\text{ s}$, espacio de acciones continuas $[-1.0, 1.0]^2$ mapeadas a velocidad angular conjunta $\dot{\boldsymbol{\theta}} = \mathbf{a} \cdot v_{max}$.
* **Criterio Estricto de Hold:** Para registrar éxito, el efector final debe ingresar en $r_{tgt} = 0.06\text{ m}$ y permanecer durante **20 pasos consecutivos** ($1.0\text{ s}$) con velocidad angular conjunta $\|\dot{\boldsymbol{\theta}}\| \le 0.10\text{ rad/s}$ antes del paso 200.
* **Integridad de Archivos Históricos:** Todos los modelos, configuraciones y reportes de la Fase A se mantuvieron en modo de solo lectura, verificando su integridad mediante hashes criptográficos SHA-256 (37 archivos auditados en [`phase_a_reference_hashes.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_a_reference_hashes.json)).

### 2.2 Qué Cambia Exclusivamente en la Fase B
* **Presupuesto Temporal de Interacción:** Se extendió el entrenamiento de 300.000 pasos nominales a **1.000.000 de pasos nominales** (exactamente 123 rollouts $\times$ 8.192 = **1.007.616 timesteps reales**).
* **Esquema de Checkpointing Atómico:** Guardado periódico cada $k=5$ rollouts (**40.960 pasos**), acoplando en cada punto temporal la red neuronal (`model.zip`) y el vector móvil de estadísticas cinemáticas (`vecnormalize_stats.pkl`).
* **Alcance de Modelos:** La exploración se concentró en dos arquitecturas PPO de red $256\times 256$ ReLU (`X5_near` y `X3`) y el controlador analítico `C0`. Los modelos secundarios de Fase A (`X1a`, `X1b`, `X1c`, `X1s`, `X2`, `X2b`, `X4`, `X6`, `X5_wv10`, `X5_wv50`) no fueron reentrenados bajo el presupuesto extendido de 1M pasos.

---

## 3. Diseño Experimental y Presupuesto Real

### 3.1 Configuraciones Evaluadas
* **`X5_near` ([`configs/phase_b/X5_near.yaml`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/configs/phase_b/X5_near.yaml)):**  
  Función de recompensa $R_t = -d_t - 0.02\|\mathbf{a}_t\|^2 + 15.0 \cdot \mathbb{I}_{succ}$. Sin término de proximidad $R_{near}$. Wrapper `VecNormalize(norm_obs=True, norm_reward=False)`.
* **`X3` ([`configs/phase_b/X3.yaml`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/configs/phase_b/X3.yaml)):**  
  Función de recompensa histórica `smooth_precision` con término de proximidad $R_{near} = (1 - d/0.30)$ activo en $d < 0.30\text{ m}$, penalización de velocidad en campo cercano ($0.15 \cdot v^2$) y esfuerzo de acción ($0.02\|\mathbf{a}\|^2$). Wrapper `VecNormalize(norm_obs=True, norm_reward=False)`.
* **`C0` ([`src/controllers/jacobian_controller.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/src/controllers/jacobian_controller.py)):**  
  Control cinemático por pseudoinversa analítica del Jacobiano con amortiguamiento DLS ($K_p = 3.0, \lambda = 0.01$).

### 3.2 Tabla de Parámetros de Entrenamiento

| Parámetro | Configuración `X5_near` | Configuración `X3` | Controlador `C0` |
| :--- | :---: | :---: | :---: |
| **Arquitectura de Red** | MLP $256\times 256$ ReLU | MLP $256\times 256$ ReLU | No aplica (Analítico) |
| **Entornos Paralelos (`num_envs`)** | 4 | 4 | No aplica |
| **Pasos de Rollout (`n_steps`)** | 2.048 | 2.048 | No aplica |
| **Tamaño de Lote (`batch_size`)** | 64 | 64 | No aplica |
| **Épocas de Optimización (`n_epochs`)** | 10 | 10 | No aplica |
| **Tasa de Aprendizaje (`learning_rate`)** | 0.00015 | 0.00015 | No aplica |
| **Factor de Descuento ($\gamma$)** | 0.99 | 0.99 | No aplica |
| **Coeficiente GAE ($\lambda_{gae}$)** | 0.95 | 0.95 | No aplica |
| **Rango de Clip PPO** | 0.20 | 0.20 | No aplica |
| **Coeficiente de Entropía** | 0.002 | 0.002 | No aplica |
| **Rollouts Totales Ejecutados** | 123 | 123 | No aplica |
| **Presupuesto Real de Timesteps** | **1.007.616** | **1.007.616** | No aplica |
| **Frecuencia de Checkpoints** | Cada 40.960 pasos (5 rollouts) | Cada 40.960 pasos (5 rollouts) | No aplica |
| **Checkpoints por Corrida / Total** | 25 por semilla / 75 total | 25 por semilla / 75 total | No aplica |

### 3.3 Tiempos de Cómputo y Trazabilidad de Ejecución
*Fuente:* [`reports/data/phase_b/tanda_1_x5_near_summary.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/tanda_1_x5_near_summary.json) y [`reports/data/phase_b/tanda_2_x3_summary.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/tanda_2_x3_summary.json).

| Experimento | Semilla | Tiempo de Ejecución (s) | Tiempo (min) | FPS Medio | Checkpoints Generados | Integridad Archivos |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`X5_near`** | 42 | 660.29 s | 11.00 min | 1.526,0 | 25 / 25 | Íntegros (0 faltantes) |
| **`X5_near`** | 100 | 677.83 s | 11.30 min | 1.486,5 | 25 / 25 | Íntegros (0 faltantes) |
| **`X5_near`** | 2026 | 630.29 s | 10.50 min | 1.598,7 | 25 / 25 | Íntegros (0 faltantes) |
| *Total `X5_near`* | — | 1.968,41 s | **32.81 min** | 1.537,1 | **75 / 75** | Íntegros |
| **`X3`** | 42 | 628.05 s | 10.47 min | 1.604,3 | 25 / 25 | Íntegros (0 faltantes) |
| **`X3`** | 100 | 636.72 s | 10.61 min | 1.582,5 | 25 / 25 | Íntegros (0 faltantes) |
| **`X3`** | 2026 | 611.56 s | 10.19 min | 1.647,6 | 25 / 25 | Íntegros (0 faltantes) |
| *Total `X3`* | — | 1.876,33 s | **31.27 min** | 1.611,5 | **75 / 75** | Íntegros |

---

## 4. Reglas Fijadas Antes de Ver Resultados

### 4.1 Regla Formal de Selección de Checkpoints ([`selection_rule.yaml`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/configs/phase_b/selection_rule.yaml))
La regla previa definió los siguientes criterios de selección:
1. `primary`: Mayor tasa media de éxito estricto determinista entre las 3 semillas (`max_mean_strict_success_rate`).
2. `tie_break_window`: Ventana de $U_{meseta} = 5.0\text{ pp}$ por debajo del máximo.
3. `tie_break_1`: Menor mediana de pasos al éxito (`min_steps_to_success_median`).
4. `tie_break_2`: Menor cambio medio de acción $\|\Delta a\|$ (`min_mean_action_delta`).
5. `single_common_checkpoint`: Selección de un único timestep común para las 3 semillas.

### 4.2 Selección de Episodios Geométricos Fijos
Sobre el conjunto de validación (`validation_set_200.json`), se fijaron a priori 4 episodios cubriendo la distribución de radios del espacio de trabajo:
* **Q10 (Radio Corto):** Episodio 34 ($r = 0.3097\text{ m}$), cercano a la base.
* **Q50 (Radio Medio):** Episodio 195 ($r = 1.0913\text{ m}$), espacio de trabajo intermedio representativo.
* **Q90 (Radio Largo):** Episodio 40 ($r = 1.7457\text{ m}$), extensión cinemática alta.
* **Frontera Singular:** Episodio 90 ($r = 1.8995\text{ m}$), radio extremo próximo a la singularidad de $2.0\text{ m}$.

### 4.3 Aislamiento de Datasets y Lectura Única
Se estableció que las decisiones de selección y análisis de mesetas se ejecutarían **exclusivamente sobre el conjunto de validación** ($n=200$). El conjunto final ($n=300$) se mantuvo sellado y se evaluó una única vez en la Etapa B-6 tras congelar los checkpoints ganadores.

### 4.4 Discrepancia Procedimental en la Aplicación de la Regla de Selección
* [HECHO]: En el script de selección [`scratch/analyze_eval.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/scratch/analyze_eval.py#L44-L47), los candidatos dentro de la ventana de 5 pp se ordenaron ejecutando:
  ```python
  tie_candidates.sort_values(
      by=["mean_strict_success", "mean_steps_to_success_median", "mean_action_delta"],
      ascending=[False, True, True]
  )
  ```
  Al ordenar de forma primaria por `mean_strict_success` descendente, el script seleccionó el **máximo absoluto exacto** de la curva (`ckpt_901120` para `X5_near` y `ckpt_409600` para `X3`). El criterio de desempate por pasos solo habría tenido efecto ante una igualdad numérica exacta en la tasa de éxito media.
* Si la ventana de 5 pp se hubiera tratado como una región de estricta indiferencia estadística desempatando en primera instancia por menor mediana de pasos, los checkpoints seleccionados habrían sido el paso **`1,007,616`** para `X5_near` (mediana de 68.3 pasos vs 69.3 pasos; éxito 93.00%) y el paso **`327,680`** para `X3` (mediana de 57.3 pasos vs 57.8 pasos; éxito 95.17%).
* Como la lectura del conjunto final en B-6 se realizó sobre `ckpt_901120` y `ckpt_409600` y no se repite, este aspecto se documenta formalmente como una discrepancia procedimental en la aplicación de la regla.

---

## 5. Reproducibilidad e Integridad

### 5.1 Determinismo y Corridas Réplica
* [HECHO]: La fijación de semillas aleatorias en PyTorch, NumPy y Gymnasium produjo trayectorias bit a bit idénticas al reejecutar evaluaciones deterministas sobre los datasets congelados con un error cuadrático medio de transición $0.0$.

### 5.2 Verificación de Checkpointing Atómico
* [HECHO]: El test unitario [`tests/test_phase_b_validations.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/tests/test_phase_b_validations.py) verificó que en cada guardado periódico, `model.zip` y `vecnormalize_stats.pkl` se almacenan en la misma carpeta (`ckpt_<paso>/`) y son recuperables de forma independiente sin desfasaje de normalización.

### 5.3 Integridad Criptográfica de Archivos de Fase A
* [HECHO]: Se verificaron los hashes SHA-256 de los 37 archivos de referencia de Fase A ([`reports/data/phase_a_reference_hashes.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_a_reference_hashes.json)). El 100% de los archivos (37/37) coincidió exactamente con el registro original, garantizando la ausencia de alteraciones en código, configuraciones o resultados de la fase previa.

---

## 6. Resultados sobre Validación (Curva de Aprendizaje por Checkpoints)

### 6.1 Evolución Comparativa a Igual Presupuesto
*Fuente:* [`reports/data/phase_b/checkpoints_eval_validation_aggregated.csv`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/checkpoints_eval_validation_aggregated.csv). Evaluado en modo determinista sobre 200 episodios por semilla (media de las 3 semillas: 42, 100, 2026):

| Checkpoint (Paso) | Éxito Medio `X3` | Éxito Medio `X5_near` | Brecha (`X3` − `X5_near`) | Mediana Pasos `X3` | Mediana Pasos `X5_near` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `40,960` | 0.50% | 0.33% | +0.17 pp | 108.0 | 125.5 |
| `81,920` | 10.33% | 9.33% | +1.00 pp | 90.2 | 104.5 |
| `122,880` | 34.17% | 7.83% | +26.33 pp | 73.3 | 73.7 |
| `163,840` | 66.33% | 18.50% | +47.83 pp | 66.3 | 73.3 |
| `204,800` | 89.33% | 40.83% | +48.50 pp | 61.3 | 69.7 |
| `245,760` | 93.00% | 60.67% | +32.33 pp | 58.7 | 67.0 |
| **`286,720`** *(~300k)* | **94.50%** | **74.67%** | **+19.83 pp** | **58.0** | **67.8** |
| **`327,680`** *(~300k)* | **95.17%** | **77.17%** | **+18.00 pp** | **57.3** | **66.0** |
| `368,640` | 97.00% | 82.17% | +14.83 pp | 57.8 | 65.3 |
| **`409,600`** *(Pico X3)* | **98.00%** | **76.17%** | **+21.83 pp** | **57.8** | **66.7** |
| `450,560` | 97.00% | 82.50% | +14.50 pp | 58.3 | 68.0 |
| `491,520` | 92.50% | 81.00% | +11.50 pp | 59.2 | 67.5 |
| `532,480` | 92.17% | 79.00% | +13.17 pp | 60.3 | 68.0 |
| `573,440` | 86.50% | 77.67% | +8.83 pp | 60.0 | 68.7 |
| `614,400` | 87.50% | 83.83% | +3.67 pp | 63.7 | 69.3 |
| `655,360` | 89.00% | 85.83% | +3.17 pp | 61.8 | 70.0 |
| `696,320` | 94.50% | 90.33% | +4.17 pp | 64.2 | 69.0 |
| `737,280` | 92.83% | 89.50% | +3.33 pp | 63.7 | 71.7 |
| `778,240` | 92.50% | 87.83% | +4.67 pp | 63.0 | 71.2 |
| **`819,200`** | **92.00%** | **92.00%** | **0.00 pp** | **62.7** | **70.7** |
| `860,160` | 94.00% | 92.33% | +1.67 pp | 63.5 | 71.7 |
| **`901,120`** *(Pico X5)* | **92.83%** | **94.50%** | **-1.67 pp** | **62.5** | **69.3** |
| `942,080` | 94.33% | 93.50% | +0.83 pp | 63.7 | 69.5 |
| `983,040` | 95.50% | 92.00% | +3.50 pp | 63.3 | 69.2 |
| **`1,007,616`** *(Final)* | **90.83%** | **93.00%** | **-2.17 pp** | **62.7** | **68.3** |

* [HECHO]: De los 25 checkpoints evaluados a lo largo del entrenamiento:
  * En **22 checkpoints**, la tasa de éxito de `X3` fue superior a la de `X5_near`.
  * En **1 checkpoint** (paso 819.200), ambas configuraciones empataron exactamente en **92.00%**.
  * En **2 checkpoints** (pasos 901.120 y 1.007.616), la tasa de éxito de `X5_near` fue superior a la de `X3` (+1.67 pp y +2.17 pp, respectivamente).
* [HECHO]: La mediana de pasos al éxito de `X3` fue inferior a la de `X5_near` en 24 de los 25 checkpoints (en el paso 122.880 registraron 73.3 vs 73.7 pasos).

### 6.2 Detección de Mesetas y Variaciones en Iteraciones Tardías
* **Meseta de `X5_near`:** La variación absoluta entre checkpoints consecutivos cumplió la condición de meseta ($|\Delta p| < 5.0\text{ pp}$ durante 2 transiciones) a partir del paso **`532,480`**, manteniéndose en un régimen estable entre 89.5% y 94.5% hasta el final.
* **Variaciones de `X3`:** Cumplió la condición de meseta a partir del paso **`286,720`** (alcanzando 98.00% en el paso 409.600). En el paso **`573,440`**, registró una caída de **$-5.67\text{ pp}$** (de 92.17% a 86.50%), recuperándose posteriormente por encima del 94% en el paso 696.320.

### 6.3 Comparación de la Curva a $\sim 303\text{k}$ Pasos
* En los checkpoints inmediatos a los 303.104 pasos de la Fase A (`286,720` y `327,680`):
  * `X5_near` registró **74.67%** y **77.17%** de éxito medio en validación. Este valor es cercano al 70.8% registrado en la Fase A, evidenciando que `X5_near` **requirió la extensión del presupuesto hasta 901k pasos** para alcanzar el 94.50%.
  * `X3` registró **94.50%** y **95.17%** de éxito medio en validación, evidenciando que ya a los 300k pasos poseía un rendimiento elevado cuando se evalúa con normalización acoplada.

### 6.4 Análisis de Sensibilidad de la Selección de Checkpoints
*Fuente:* Evaluación sobre validación variando la ventana de tolerancia $U_{window}$ entre 1.0 y 10.0 pp:

| Experimento | Ventana $U$ | Candidatos | Menor Mediana de Pasos | Mayor Suavidad (Min $\|\Delta a\|$) | Más Temprano (Min Timesteps) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`X5_near`** | 1.0 pp ($\ge 93.5\%$) | 2 | Paso 901,120 (94.5%, 69.3 p) | Paso 901,120 (94.5%, $\Delta a=0.0324$) | Paso 901,120 (94.5%) |
| **`X5_near`** | 2.0 pp ($\ge 92.5\%$) | 3 | Paso 1,007,616 (93.0%, 68.3 p) | Paso 901,120 (94.5%, $\Delta a=0.0324$) | Paso 901,120 (94.5%) |
| **`X5_near`** | 5.0 pp ($\ge 89.5\%$) | 8 | Paso 1,007,616 (93.0%, 68.3 p) | Paso 737,280 (89.5%, $\Delta a=0.0297$) | Paso 696,320 (90.3%) |
| **`X3`** | 1.0 pp ($\ge 97.0\%$) | 3 | Paso 368,640 (97.0%, 57.8 p) | Paso 368,640 (97.0%, $\Delta a=0.0444$) | Paso 368,640 (97.0%) |
| **`X3`** | 3.0 pp ($\ge 95.0\%$) | 5 | Paso 327,680 (95.2%, 57.3 p) | Paso 327,680 (95.2%, $\Delta a=0.0432$) | Paso 327,680 (95.2%) |
| **`X3`** | 5.0 pp ($\ge 93.0\%$) | 10 | Paso 327,680 (95.2%, 57.3 p) | Paso 245,760 (93.0%, $\Delta a=0.0392$) | Paso 245,760 (93.0%) |

### 6.5 Subsección: Sesgo de Selección Optimista (*Optimistic Selection Bias*)
* [HECHO]: Los checkpoints evaluados sobre el conjunto final fueron elegidos seleccionando el valor máximo observado sobre el conjunto de validación (`ckpt_901120` con 94.5% y `ckpt_409600` con 98.0%).
* [HECHO]: Entre los checkpoints de la meseta (pasos 696k a 1007k para `X5_near`), la desviación estándar inter-semilla es de aproximadamente $\pm 2\text{--}5\text{ pp}$ y la diferencia entre candidatos adyacentes está comprendida dentro de un rango de ruido muestral de $\sim 7\text{ pp}$.
* [HIPÓTESIS]: Al seleccionar sistemáticamente el punto que obtuvo la realización muestral más alta dentro de una meseta de candidatos con rendimiento subyacente similar, existe un sesgo de selección que puede hacer que la estimación del rendimiento sobre el conjunto de validación y la consiguiente evaluación final reflejen una fluctuación favorable, resultando potencialmente optimista respecto al valor medio esperado de la política en ese régimen.

---

## 7. Resultados sobre el Conjunto Final (Lectura Única B-6)

### 7.1 Tabla Maestra Consolidada por Modelo y Modo de Evaluación
*Fuente:* [`reports/data/phase_b/final_eval_summary.csv`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/final_eval_summary.csv) y [`reports/data/phase_b/final_eval_all_episodes.csv`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/final_eval_all_episodes.csv).  
Protocolo estricto evaluado sobre los 300 episodios de [`data/datasets/final_eval_set_300.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/data/datasets/final_eval_set_300.json).

| Modelo | Checkpoint | Modo | Episodios | Éxito Estricto Medio | Desv. Est. (std) | Truncamientos | Balísticas | Oscilatorias | Mediana Pasos | Distancia Final Media | Suavidad $\|\Delta a\|$ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C0`** | N/A | Determinista | 300 | **100.0%** | N/A | 0.00% | 10.67% | 0.00% | 64.0 | 0.0028 m | 0.0444 |
| **`X5_near`** | `901,120` | Determinista | 900 | **94.89%** | 2.04 pp | 5.11% | 19.11% | 0.00% | 67.0 | 0.0318 m | 0.0332 |
| **`X5_near`** | `901,120` | Estocástico | 900 | **0.00%** | 0.00 pp | 100.0% | 53.22% | 74.89% | No aplica | 0.0385 m | 0.1494 |
| **`X3`** | `409,600` | Determinista | 900 | **98.00%** | 0.67 pp | 2.00% | 71.89% | 0.00% | 56.0 | 0.0224 m | 0.0481 |
| **`X3`** | `409,600` | Estocástico | 900 | **0.00%** | 0.00 pp | 100.0% | 96.67% | 99.22% | No aplica | 0.0411 m | 0.5899 |

### 7.2 Desglose por Semilla e Intervalos de Confianza de Wilson al 95%
Para evitar la subestimación de incertidumbre derivada de tratar los 900 episodios como réplicas independientes (dado que las 3 semillas evalúan exactamente las mismas 300 metas), se reportan los intervalos de Wilson por semilla individual ($n=300$):

#### Agente `C0` (Determinista):
* Semilla 0 ($n=300$): **100.00%** (300/300) $\to$ IC Wilson 95%: **[98.74%, 100.00%]**.

#### Agente `X5_near` (`ckpt_901120`, Determinista):
* Semilla 42 ($n=300$): **95.33%** (286/300) $\to$ IC Wilson 95%: **[92.32%, 97.20%]**.
* Semilla 100 ($n=300$): **92.67%** (278/300) $\to$ IC Wilson 95%: **[89.15%, 95.11%]**.
* Semilla 2026 ($n=300$): **96.67%** (290/300) $\to$ IC Wilson 95%: **[93.97%, 98.18%]**.
* *Media inter-semilla:* **94.89%** $\pm$ 2.04 pp (puntos muestrales: 95.33%, 92.67%, 96.67%).  
  *(Referencia agrupada $n=900$: [93.25%, 96.15%]; subestima la dispersión inter-semilla).*

#### Agente `X5_near` (`ckpt_901120`, Estocástico):
* Semillas 42, 100 y 2026 ($n=300$ cada una): **0.00%** (0/300) $\to$ IC Wilson 95%: **[0.00%, 1.26%]**.

#### Agente `X3` (`ckpt_409600`, Determinista):
* Semilla 42 ($n=300$): **97.33%** (292/300) $\to$ IC Wilson 95%: **[94.83%, 98.64%]**.
* Semilla 100 ($n=300$): **98.67%** (296/300) $\to$ IC Wilson 95%: **[96.62%, 99.48%]**.
* Semilla 2026 ($n=300$): **98.00%** (294/300) $\to$ IC Wilson 95%: **[95.71%, 99.08%]**.
* *Media inter-semilla:* **98.00%** $\pm$ 0.67 pp (puntos muestrales: 97.33%, 98.67%, 98.00%).  
  *(Referencia agrupada $n=900$: [96.86%, 98.73%]; subestima la dispersión inter-semilla).*

#### Agente `X3` (`ckpt_409600`, Estocástico):
* Semillas 42, 100 y 2026 ($n=300$ cada una): **0.00%** (0/300) $\to$ IC Wilson 95%: **[0.00%, 1.26%]**.

---

## 8. Diagnóstico Visual en Profundidad

*Nota metodológica:* Las representaciones visuales espaciales ilustran realizaciones concretas de trayectorias, pero la validez cuantitativa descansa en las métricas tabuladas con intervalos de confianza.

### 8.1 Análisis de las 9 Figuras Generadas

#### 8.1.1 `fig_b_01_workspace_failure_map.png` (Mapa de Fallos en Validación, $n=200$)
* *Contenido medido:* 6 paneles cartesianos 2D ($x, y$) en coordenadas de la meta sobre el conjunto de validación para `C0`, `X5_near` Fase A (semilla 42), `X5_near` Fase B det y stoch (semilla 42), y `X3` Fase B det y stoch (semilla 42). Clasifica los desenlaces en 6 categorías discretas.
* *Desviación respecto al diseño:* Utiliza únicamente la semilla 42 para evitar la superposición ilegible de 600 puntos por panel.

#### 8.1.2 `fig_b_01_final_workspace_failure_map.png` (Mapa de Fallos en Conjunto Final, $n=300$)
* *Contenido medido:* 5 paneles cartesianos sobre el conjunto final terminal evaluado en la Etapa B-6.
* *Desviación respecto al diseño:* Generada como figura complementaria no prevista en el listado inicial de 8 figuras de B-2 para documentar la lectura final B-6. Emplea exclusivamente la semilla 42.

#### 8.1.3 `fig_b_02_agent_training_curves.png` (Curvas de Entrenamiento en TensorBoard)
* *Contenido medido:* 4 paneles temporales a lo largo de 1.000.000 de pasos: recompensa media acumulada, longitud de episodio, pérdida de valor (`value_loss`) y entropía (`entropy_loss`).
* *Desviación respecto al diseño:* Solo grafica los rollouts de la **Fase B**; no superpone la Fase A debido a diferencias en la frecuencia de registro y longitud de corrida.

#### 8.1.4 `fig_b_03_learning_curve_checkpoints.png` (Curva por Checkpoints vs Pasos)
* *Contenido medido:* Panel superior: Éxito estricto determinista (%) en validación para los 25 checkpoints de `X5_near` y `X3` (media ± std inter-semilla y curvas individuales), línea de referencia de `C0` (100%) y línea vertical de Fase A (303k). Panel inferior: Velocidad angular media articular al primer ingreso ($rad/s$).
* *Desviación respecto al diseño:* El panel inferior representa la velocidad angular de ingreso en lugar de las tasas de truncamiento y balísticas previstas originalmente.

#### 8.1.5 `fig_b_04_phase_plane_distance_speed.png` (Plano de Fase Cartesiano e Histograma)
* *Contenido medido:* Panel izquierdo: Distancia cartesiana $d(t)$ vs velocidad cartesiana del efector final $v_{ee}(t)$ para episodios geométricos representativos. Panel derecho: Histogramas de densidad de velocidad angular al primer ingreso a la zona meta con línea de corte en $u_b = 0.30\text{ rad/s}$.

#### 8.1.6 `fig_b_05_time_profiles_geometry.png` (Perfiles Temporales por Geometría)
* *Contenido medido:* Telemetría paso a paso ($t \in [0, 200]$) sobre los 4 episodios geométricos fijados (34, 195, 40, 90) sobre validación: distancia euclidiana, velocidad angular conjunta, acciones comandadas $a_1, a_2$ y contador de hold.

#### 8.1.7 `fig_b_06_cartesian_trajectories_gradient.png` (Trayectorias 2D en Degradé)
* *Contenido medido:* Trayectorias cartesianas 2D del efector final para `X5_near` Fase B (semilla 42) comparadas con `C0` sobre los 4 episodios geométricos, con degradé de color indicando avance temporal.
* *Desviación respecto al diseño:* Solo incluye `X5_near` frente a `C0`; `X3` se omitió para evitar saturación de trazos superpuestos. Emplea únicamente la semilla 42.

#### 8.1.8 `fig_b_07_det_vs_stoch_same_episode.png` (Comparativa Determinista vs Estocástica)
* *Contenido medido:* Episodio 195 (semilla 42) de `X5_near`: Panel 1 muestra la trayectoria 2D suave vs ruidosa. Panel 2 muestra la velocidad angular conjunta a lo largo del tiempo frente al umbral $v_{tol} = 0.10\text{ rad/s}$.

#### 8.1.9 `fig_b_08_ppo_diagnostics_phase_b.png` (Diagnósticos Internos de PPO)
* *Contenido medido:* 4 paneles con métricas internas de optimización: evolución de la dispersión de acción ($\sigma$), divergencia KL aproximada (`approx_kl`), fracción de recorte (`clip_fraction`) y varianza explicada del crítico (`explained_variance`).

---

## 9. Evaluación de Criterios de Aceptación

Contraste individualizado contra los umbrales formales de [`configs/phase_a/thresholds.yaml`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/configs/phase_a/thresholds.yaml#L19-L26) evaluados sobre el conjunto final (300 episodios):

| Criterio | Umbral Formal | `C0` (Medido) | `X3` (`ckpt_409600`) | `X5_near` (`ckpt_901120`) |
| :--- | :---: | :---: | :---: | :---: |
| **Éxito estricto determinista medio** | $\ge 95.0\%$ | **100.0%** (Cumple) | **98.00%** (Cumple) | **94.89%** (No cumple: 94.89% < 95.00%) |
| **Éxito mínimo en cualquier semilla** | $\ge 90.0\%$ | **100.0%** (Cumple) | **97.33%** (Cumple) | **92.67%** (Cumple) |
| **Dispersión inter-semilla (std)** | $\le 5.0\text{ pp}$ | **0.00 pp** (Cumple) | **0.67 pp** (Cumple) | **2.04 pp** (Cumple) |
| **Llegadas balísticas** | $\le 5.0\%$ | **10.67%** (No cumple) | **71.89%** (No cumple) | **19.11%** (No cumple) |
| **Episodios oscilatorios** | $\le 5.0\%$ | **0.00%** (Cumple) | **0.00%** (Cumple) | **0.00%** (Cumple) |
| **Tasa de truncamientos** | $\le 5.0\%$ | **0.00%** (Cumple) | **2.00%** (Cumple) | **5.11%** (No cumple: 5.11% > 5.00%) |
| **Brecha determinista–estocástica** | $\le 10.0\text{ pp}$ | No aplica | **98.00 pp** (No cumple) | **94.89 pp** (No cumple) |

*Criterios añadidos en Fase B:*
* **Criterio de Meseta en Validación:** $|\Delta p| < 5.0\text{ pp}$ durante 2 transiciones consecutivas. Cumplido por `X5_near` a partir del paso 532.480 y por `X3` a partir del paso 286.720.

---

## 10. Balance HECHO / HIPÓTESIS

### 10.1 Hechos Medidos

| Identificador | Afirmación Basada en Datos | Fuente de Verificación |
| :--- | :--- | :--- |
| **H-1** | `X3` supera a `X5_near` en validación en 22 de los 25 checkpoints evaluados a igual presupuesto. | [`checkpoints_eval_validation_aggregated.csv`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/checkpoints_eval_validation_aggregated.csv) |
| **H-2** | En el conjunto final, `X3` alcanza 98.00% de éxito frente a 94.89% de `X5_near` en modo determinista. | [`final_eval_summary.csv`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/final_eval_summary.csv) |
| **H-3** | Ambos modelos PPO obtienen 0/300 éxitos en modo estocástico bajo evaluación estricta en todas las semillas. | [`final_eval_all_episodes.csv`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/final_eval_all_episodes.csv) |
| **H-4** | `eval/success_rate` en línea (15 episodios, determinista, hold relajado) finalizó en 95.56% para `X3` y 86.67% para `X5_near`. | TensorBoard logs en `experiments/phase_b/` |
| **H-5** | La dispersión articular `train/std` al final promedió 0.2278 para `X3` y 0.0821 para `X5_near`. | TensorBoard logs y parámetros de red |
| **H-6** | `ppo_arm2d_final.zip` de Fase A (300k) alcanza 94.33% en validación con normalización acoplada. | [`phase_a_models_val_comparison.csv`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/phase_a_models_val_comparison.csv) |
| **H-7** | `C0` redujo sus llegadas balísticas de 36.3% a 10.7% al evaluarse con `max_speed = 1.5 rad/s`. | [`final_eval_summary.csv`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/final_eval_summary.csv) y [`eval_final_summary.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/experiments/phase_a/C0/eval_final_summary.json) |

### 10.2 Hipótesis sin Demostración Directa

| Identificador | Afirmación / Explicación Propuesta | Estado de Demostración |
| :--- | :--- | :--- |
| **HP-1** | *El término $R_{near}$ impide que la política aprenda a estabilizarse.* | **Debilitada por los datos:** `X3` (con $R_{near}$) alcanzó 98.00% de éxito estricto y superó a `X5_near` en la mayoría de los checkpoints. |
| **HP-2** | *El mecanismo de frenado de `X3` se basa en la compensación entre el gradiente de distancia y la penalización de acción.* | **Hipótesis no aislada:** No se realizó una ablación factorial separando cada término de costo en Fase B. |
| **HP-3** | *La red neuronal $64\times 64$ carecía de capacidad representacional frente a $256\times 256$.* | **Hipótesis no validada en Fase B:** No se entrenaron redes $64\times 64$ a 1M pasos con checkpoints acoplados. |
| **HP-4** | *La brecha determinista–estocástica se debe a la probabilidad compuesta de violación del umbral $0.10\text{ rad/s}$ bajo ruido normal.* | **Hipótesis analítica:** Se deduce de la magnitud de $\sigma$, pero no se ejecutó un barrido de $\sigma \to 0$ en inferencia estocástica. |

---

## 11. Limitaciones y Caveats (Obligatorios)

1. **Alcance Restringido de Modelos y Efecto en el Ranking:**  
   La Fase B solo entrenó y evaluó `X5_near` y `X3`. Los demás agentes explorados en Fase A (`X1`, `X2`, `X4`, `X5`, `X6`) no fueron entrenados a 1M pasos. Dado que la evaluación de Fase A para `X3` estuvo distorsionada por desfasaje de normalización, el ranking histórico comparativo de la Fase A no es directamente extrapolable a presupuestos extendidos.
2. **Comparación Basada en Checkpoints Desacoplados Temporalmente:**  
   La comparación entre `X3` y `X5_near` en el conjunto final compara checkpoints seleccionados en puntos temporales distintos (paso 409.600 vs paso 901.120). Si bien refleja el punto óptimo identificado para cada configuración, no constituye una comparación a igual número de iteraciones en la evaluación final.
3. **Desacople entre Bandera Balística y Fallo de Tarea:**  
   La métrica operativa de llegada balística ($v_{entry} > 0.30\text{ rad/s}$) clasifica como balísticos al 71.89% de los episodios de `X3` y al 10.67% de `C0`, a pesar de que ambos modelos logran estabilizarse con éxito estricto en $\ge 98\%$ de los casos. La bandera refleja la cinemática de primer orden (acercamiento rápido sin costo físico por frenado brusco), no una falla operativa.
4. **Colapso Sistemático del Modo Estocástico:**  
   El éxito de 0.0% en modo estocástico no representa una degradación del control medio, sino una incompatibilidad entre la inyección de ruido de acción ($\sigma \approx 0.08\text{--}0.23$) y la exigencia de mantener velocidad por debajo de $0.10\text{ rad/s}$ durante 20 pasos ininterrumpidos.
5. **Sesgo de Selección Optimista:**  
   La elección del checkpoint con mayor media en validación dentro de una meseta de fluctuación estadística puede inducir un sesgo optimista en las cifras finales reportadas.
6. **Diferencias de Procedimiento Declaradas:**  
   * En Fase A, `best_model.zip` fue evaluado con estadísticas de `VecNormalize` del final de entrenamiento, subestimando el rendimiento de `X3` en 50.16 pp.
   * En `C0`, la velocidad máxima del entorno difirió entre Fase A (2.0 rad/s) y Fase B (1.5 rad/s), alterando la ganancia efectiva de control.
7. **Restricción de Gráficos Espaciales:**  
   Los mapas cartesianos (Fig B-1 y B-1b) y las trayectorias temporales (Fig B-6 y B-7) grafican únicamente la semilla 42 para preservar la claridad visual.

---

## 12. Ideas para Fases Futuras (No Evaluadas)

*Estas propuestas constituyen hipótesis conceptuales para trabajo futuro; no fueron evaluadas experimentalmente en la Fase B y no forman parte de las conclusiones:*

1. **Envolvente de Velocidad Admisible (*Velocity Funnel*):**  
   Definir un perfil de velocidad límite dependiente de la distancia al objetivo $v_{lim}(d) = v_{max} \cdot \text{clip}((d - r_{tgt})/d_0, 0, 1)$ y penalizar exclusivamente el exceso respecto a dicho perfil, incentivando la desaceleración suave sin provocar parálisis cinemática en campo lejano.
2. **Restricción Física de Aceleración (*Rate Limiter* en el Entorno):**  
   Implementar una cota sobre la derivada temporal de la acción $|\mathbf{a}_t - \mathbf{a}_{t-1}| \le \Delta a_{max}$ o un filtro de primer orden en la velocidad articular, impidiendo cambios instantáneos de velocidad y forzando al algoritmo de RL a anticipar la desaceleración.
3. **Planificación de Trayectorias y Modulación de Ganancia en `C0`:**  
   Incorporar un perfil asintótico $\tanh(d / d_{brake})$ en la velocidad comandada por el controlador analítico para eliminar las entradas balísticas remanentes ($10.67\%$).
4. **Decaimiento Programado del Ruido de Exploración:**  
   Programar un decaimiento explícito del parámetro `log_std` en la política gaussiana durante las épocas finales para posibilitar la estabilización estocástica bajo tolerancias estrechas.

---

## 13. Apéndice Metodológico y Trazabilidad

### 13.1 Inventario de Archivos Nuevos Generados en la Fase B
* **Configuraciones:**
  * [`configs/phase_b/X5_near.yaml`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/configs/phase_b/X5_near.yaml)
  * [`configs/phase_b/X3.yaml`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/configs/phase_b/X3.yaml)
  * [`configs/phase_b/selection_rule.yaml`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/configs/phase_b/selection_rule.yaml)
* **Scripts de Pipeline:**
  * [`scripts/phase_b_train.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/scripts/phase_b_train.py) (Entrenamiento centralizado PPO a 1M)
  * [`scripts/phase_b_run_tanda1.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/scripts/phase_b_run_tanda1.py) (Tanda 1: `X5_near`, 3 semillas)
  * [`scripts/phase_b_run_tanda2.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/scripts/phase_b_run_tanda2.py) (Tanda 2: `X3`, 3 semillas)
  * [`scripts/phase_b_eval_checkpoints.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/scripts/phase_b_eval_checkpoints.py) (Evaluador modular de checkpoints)
  * [`scripts/phase_b_batch_eval_validation.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/scripts/phase_b_batch_eval_validation.py) (Evaluación masiva en validación)
  * [`scripts/phase_b_prepare_eval_and_telemetry.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/scripts/phase_b_prepare_eval_and_telemetry.py) (Preparación de telemetría de validación)
  * [`scripts/phase_b_step_data.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/scripts/phase_b_step_data.py) (Extractor de telemetría paso a paso)
  * [`scripts/phase_b_final_evaluation.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/scripts/phase_b_final_evaluation.py) (Lectura única del conjunto final B-6)
  * [`scripts/generate_phase_b_figures.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/scripts/generate_phase_b_figures.py) (Pipeline integral de figuras a 300 dpi)
* **Módulos de Código Fuente y Tests:**
  * [`src/training/phase_b_callbacks.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/src/training/phase_b_callbacks.py) (Callback de checkpointing atómico)
  * [`tests/test_phase_b_validations.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/tests/test_phase_b_validations.py) (Suite de validaciones y determinismo)
* **Datos Consolidados en [`reports/data/phase_b/`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/):**
  * `tanda_1_x5_near_summary.json` y `tanda_2_x3_summary.json`
  * `checkpoints_eval_validation.csv` y `checkpoints_eval_validation_aggregated.csv`
  * `final_eval_all_episodes.csv` y `final_eval_summary.csv`
  * `step_data_val_geometry.csv` y `step_data_final_geometry.csv`
  * `phase_a_models_val_comparison.csv`
* **Figuras Generadas en [`reports/figures/phase_b/`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/figures/phase_b/):**
  * `fig_b_01_workspace_failure_map.png` y `.json`
  * `fig_b_01_final_workspace_failure_map.png` y `.json`
  * `fig_b_02_agent_training_curves.png` y `.json`
  * `fig_b_03_learning_curve_checkpoints.png` y `.json`
  * `fig_b_04_phase_plane_distance_speed.png` y `.json`
  * `fig_b_05_time_profiles_geometry.png` y `.json`
  * `fig_b_06_cartesian_trajectories_gradient.png` y `.json`
  * `fig_b_07_det_vs_stoch_same_episode.png` y `.json`
  * `fig_b_08_ppo_diagnostics_phase_b.png` y `.json`

### 13.2 Versiones de Librerías del Entorno
* **Python:** 3.11.16 (64-bit)
* **Sistema Operativo:** Windows-10-10.0.19045-SP0
* **PyTorch:** 2.13.0+cpu
* **Gymnasium:** 1.3.0
* **Stable-Baselines3:** 2.9.0
* **NumPy:** 1.26.4
* **Pandas:** 3.0.5
* **Matplotlib:** 3.11.1

### 13.3 Auditoría de Integridad de Hashes (37 Archivos de Fase A)
Auditoría ejecutada contra [`reports/data/phase_a_reference_hashes.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_a_reference_hashes.json):
* Archivos comprobados: 37
* Coincidencias exactas: **37 / 37 (100.0%)**
* Archivos modificados o ausentes: **0**

### 13.4 Comandos para Reproducir
Ejecución secuencial desde la raíz del repositorio con el entorno virtual activo:

```powershell
# 1. Ejecutar suites de entrenamiento a 1.000.000 de pasos
python scripts/phase_b_run_tanda1.py
python scripts/phase_b_run_tanda2.py

# 2. Evaluación masiva de checkpoints sobre validación (n=200)
python scripts/phase_b_batch_eval_validation.py

# 3. Preparación de telemetría y datasets geométricos de validación
python scripts/phase_b_prepare_eval_and_telemetry.py

# 4. Lectura única terminal del conjunto final (n=300)
python scripts/phase_b_final_evaluation.py

# 5. Generación de las figuras diagnósticas de Fase B a 300 dpi
python scripts/generate_phase_b_figures.py

# 6. Ejecutar tests unitarios de validación de Fase B
python -m unittest tests/test_phase_b_validations.py
```

---

## 14. Addenda: Fase B (Extensión) — Comparación Exhaustiva de Candidatas y Selección Formal del Mejor Agente (Presupuesto 655 360 Pasos)

### 14.1 Alcance, Presupuesto y Protocolo de Decisión Pre-registrado
En esta extensión de la Fase B (Etapas B-7 a B-13) se unificó el presupuesto de interacción a **655 360 timesteps** (80 rollouts de 8 192 pasos, 16 checkpoints periódicos cada 40 960 pasos) para **todas las configuraciones candidatas**:
* **7 agentes principales (n=3 semillas: 42, 100, 2026):** `X1a`, `X2`, `X3`, `X4`, `X5`, `X5_near`, `X6`.
* **4 agentes pilotos (n=1 semilla: 42, informativos fuera de la regla):** `X1b`, `X1c`, `X1s`, `X2b`.
* **2 versiones del baseline analítico:** `C0 (consistente, max_speed=1.5)` y `C0 (histórico, max_speed=2.0)`.

Se generaron dos conjuntos de 500 episodios con 0 % de solapamiento entre sí y con los conjuntos históricos:
1. `validation_b2_500.json` (semilla maestra 202611): Utilizado para la selección del checkpoint óptimo bajo la **Regla 4.1**.
2. `final_b2_500.json` (semilla maestra 202622, SHA-256 `c02716de...`): Sellado criptográficamente hasta la evaluación única terminal bajo la **Regla 4.2**.

La reutilización de checkpoints para `X3` y `X5_near` se rotula formalmente como: *"verificado hasta 655360 para X3 y X5_near (semilla 42); inferido para semillas 100 y 2026"*, tras comprobar empíricamente que la diferencia paramétrica y de `VecNormalize` respecto a un entrenamiento nuevo fue exactamente 0.0.

---

### 14.2 Selección de Checkpoints por Agente en Validación B2 (Regla 4.1)
Bajo la Regla 4.1 (media móvil de 3 puntos, extremos truncados, umbral de indiferencia $U_{equiv}$ calculado como ancho Wilson al 95 % en $n=500$ sobre el pico, y desempate por checkpoint más temprano):

| Agente | Tipo | Checkpoint Elegido | Éxito Suavizado | Éxito Raw ($\pm$ std) | Pico Suavizado (Paso) | Banda $U_{equiv}$ | Pendiente Final [573k $\to$ 655k] | Diagnóstico Asintótico |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **X4** | Principal | **450 560** | **97.18 %** | **97.00 %** $\pm$ 2.31 % | 98.40 % (655k) | 2.31 pp | +0.53 pp | Meseta (>98 %) |
| **X3** | Principal | **286 720** | **94.11 %** | **94.13 %** $\pm$ 2.16 % | 97.00 % (410k) | 3.06 pp | +1.53 pp | Meseta (~95 %) |
| **X6** | Principal | **286 720** | **93.73 %** | **93.67 %** $\pm$ 2.81 % | 96.24 % (369k) | 3.41 pp | +1.93 pp | Meseta (~95 %) |
| **X5** | Principal | **286 720** | **92.76 %** | **94.73 %** $\pm$ 3.72 % | 95.09 % (451k) | 3.87 pp | -4.40 pp | Degradando al final |
| **X2b** | Piloto ($n=1$) | **532 480** | **86.33 %** | **87.60 %** | 89.80 % (655k) | 5.32 pp | +2.60 pp | Creciendo |
| **X5_near**| Principal | **327 680** | **80.47 %** | **79.00 %** $\pm$ 7.83 % | 85.00 % (655k) | 6.26 pp | +7.53 pp | Creciendo |
| **X1a** | Principal | **614 400** | **75.96 %** | **77.53 %** $\pm$ 5.00 % | 78.73 % (655k) | 7.15 pp | +9.53 pp | Creciendo |
| **X2** | Principal | **573 440** | **76.20 %** | **75.27 %** $\pm$ 1.67 % | 82.80 % (655k) | 6.61 pp | +9.67 pp | Creciendo |
| **X1b** | Piloto ($n=1$) | **614 400** | **70.93 %** | **71.80 %** | 74.10 % (655k) | 7.67 pp | +11.80 pp | Creciendo |
| **X1c** | Piloto ($n=1$) | **614 400** | **70.93 %** | **71.80 %** | 74.10 % (655k) | 7.67 pp | +11.80 pp | Creciendo |
| **X1s** | Piloto ($n=1$) | **614 400** | **49.73 %** | **48.40 %** | 53.80 % (655k) | 8.71 pp | +17.60 pp | Creciendo |

---

### 14.3 Evaluación Terminal sobre Final B2 y Veredicto Formal (Regla 4.2)
En la lectura única del conjunto final (`final_b2_500.json`, $n=500$ episodios por semilla):

| Pos. | Agente | Checkpoint Evaluado | Éxito Medio ($n=3$) | Desv. Semillas ($s_R$) | Semillas Raw (`42`, `100`, `2026`) | Distancia al Líder | ¿Equivalente Estadístico? | Mediana Pasos | Suavidad $\overline{\Delta a}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **X4** | Paso 450 560 | **96.73 %** | 2.81 % | $93.8\,\%,\,97.0\,\%,\,99.4\,\%$ | $0.00\text{ pp}$ | **SÍ (Líder)** | 59.7 | 0.0464 |
| **2** | **X3** | Paso 286 720 | **94.53 %** | **0.83 %** | $93.6\,\%,\,94.8\,\%,\,95.2\,\%$ | **2.20 pp** | **SÍ** | **57.7** | 0.0422 |
| **3** | **X5** | Paso 286 720 | **93.93 %** | 4.37 % | $89.2\,\%,\,97.8\,\%,\,94.8\,\%$ | **2.80 pp** | **SÍ** | 60.0 | **0.0392** |
| 4 | **X6** | Paso 286 720 | **92.67 %** | 3.37 % | $95.0\,\%,\,94.2\,\%,\,88.8\,\%$ | 4.07 pp | No ($> U_{equiv}$) | 59.3 | 0.0387 |
| 5 | **X5_near**| Paso 327 680| **79.60 %** | 7.63 % | $88.4\,\%,\,75.6\,\%,\,74.8\,\%$ | 17.13 pp | No | 67.2 | 0.0289 |
| 6 | **X1a** | Paso 614 400 | **77.00 %** | 5.30 % | $71.2\,\%,\,78.2\,\%,\,81.6\,\%$ | 19.73 pp | No | 71.5 | 0.0318 |
| 7 | **X2** | Paso 573 440 | **74.73 %** | 1.81 % | $76.4\,\%,\,72.8\,\%,\,75.0\,\%$ | 22.00 pp | No | 69.3 | 0.0311 |

* **Baselines C0 en Final B2:** Ambos alcanzan **100.0 %** de éxito; $C0$ consistente ($\text{max\_speed}=1.5$) registra mediana de **59.0 pasos** y $\overline{\Delta a}=0.0508$; $C0$ histórico ($\text{max\_speed}=2.0$) registra mediana de **63.0 pasos** y $\overline{\Delta a}=0.0435$.
* **Pilotos $n=1$ en Final B2:** `X2b` (85.00 %), `X1b` (71.20 %), `X1c` (71.20 %), `X1s` (47.00 %).

#### Veredicto Formal Emitido:
* Líder muestral: **`X4`** ($\bar{p}_{top} = 96.73\,\%$).
* Banda de ruido operacional ($n=500$): $U_{equiv\_final} = \mathbf{3.16\text{ pp}}$ (Umbral: **93.58 %**).
* **Declaración:** Se declara formalmente un **Grupo de Agentes Estadísticamente Equivalentes: `['X4', 'X3', 'X5']`**. Con los datos y presupuesto disponibles, no se distinguen estadísticamente.
* **Caracterización Multicriterio del Grupo:**
  * **X4:** Líder en tasa bruta ($96.73\,\%$, pico semilla $99.4\,\%$), pero requiere más timesteps para converger (450k) y exhibe mayor esfuerzo de acción ($\overline{\Delta a} = 0.0464$).
  * **X3:** Líder en consistencia ($s_R = \mathbf{0.83\,\%}$ inter-semilla) y velocidad de alcance (mediana de **57.7 pasos**, superando a $C0$), con convergencia temprana (paso 286k).
  * **X5:** Líder en suavidad de acción ($\overline{\Delta a} = \mathbf{0.0392}$), a costo de mayor dispersión entre semillas ($s_R = 4.37\,\%$).

---

### 14.4 Aislamiento de Factores: Pares Adyacentes (Final B2)

| Par Adyacente | Factor Aislado | $\Delta$ Éxito | $\Delta$ Pasos | $\Delta$ Suavidad ($\overline{\Delta a}$) | Conclusión |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **X1a $\to$ X2** | `VecNormalize(obs)` sobre red idéntica | $-2.27\text{ pp}$ | $-2.2\text{ pasos}$ | $-0.0007$ | La normalización por sí sola no compensa una recompensa sin guía espacial a 655k. |
| **X2 $\to$ X3** | Recompensa cuadrática + $R_{near}$ | **$+19.80\text{ pp}$** | **$-11.7\text{ pasos}$** | $+0.0110$ | **Factor más determinante del proyecto:** transforma una política mediocre en casi óptima. |
| **X3 $\to$ X4** | Batch size PPO ($64 \to 128$) | $+2.20\text{ pp}$ | $+2.0\text{ pasos}$ | $+0.0042$ | Mayor estabilidad de gradiente PPO que eleva el techo asintótico requiriendo más pasos. |
| **X3 $\to$ X5_near**| Supresión de $R_{near}$ positivo | **$-14.93\text{ pp}$** | **$+9.5\text{ pasos}$** | $-0.0133$ | La ausencia de proximidad degrada severamente el frenado y la retención estricta. |
| **X3 $\to$ X5** | $R_{vel}$ gaussiana ($w_v=0.10, \sigma_v=0.15$) | $-0.60\text{ pp}$ | $+2.3\text{ pasos}$ | **$-0.0030$** | Modela frenado suave en el extremo a costo despreciable en éxito estricto. |
| **X5 $\to$ X6** | Penalización aceleración $w_a=0.01$ | $-1.27\text{ pp}$ | $-0.7\text{ pasos}$ | $-0.0005$ | Sobre-penalización: ganancia nula en suavidad sobre X5 y reducción marginal en éxito. |

---

### 14.5 Documentación de Advertencias de Configuración Históricas
1. **Parámetros reales de la familia X6:** Los valores implementados en el repositorio son $w_a = 0.01$ (`X6`), $w_a = 0.10$ (`X6_wa01`) y $w_a = 0.50$ (`X6_wa05`), y no $0.02$ ni $0.05$.
2. **Descripción textual de `X6.yaml`:** La descripción en [`configs/phase_a/X6.yaml`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/configs/phase_a/X6.yaml) indica erróneamente *"Añade R_vel gaussiana sobre X3"*, cuando en realidad añade penalización de aceleración $w_a = 0.01$ sobre $X5$ (heredando $w_v = 0.10$).
3. **Estado de $R_{near}$ en `x5_vel`:** El reward `x5_vel` mantiene el término positivo $R_{near}$ activo por defecto (`r_near_mode: "historical"`). Únicamente `x5_near` desactiva dicho término (`r_near_mode: "none"`).

---

### 14.6 Trazabilidad y Archivos de la Extensión
* **Reporte Monográfico Completo:** [`reports/phase_b_extension_report.md`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/phase_b_extension_report.md)
* **Veredicto Estructurado:** [`reports/data/phase_b/extension/final_b2_selection_verdict.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_b/extension/final_b2_selection_verdict.json)
* **Figuras a 300 dpi con Metadatos JSON (18 pares):** [`reports/figures/phase_b/extension/`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/figures/phase_b/extension/)
  * `fig_ext_learning_curves_main.png` y `.json`
  * `fig_ext_learning_curves_pilots.png` y `.json`
  * `fig_ext_success_barplot.png` y `.json`
  * `fig_ext_pareto_smoothness.png` y `.json`
  * `fig_ext_pareto_speed.png` y `.json`
  * `fig_ext_canonical_trajectories.png` y `.json`
  * `fig_ext_canonical_actions.png` y `.json`
  * 11 figuras individuales `fig_agent_<agente>_selection.png` y `.json`.
