# 📊 Clasificación Integral del Repositorio para la Transición a Fase D (LQR continuo MDP)

> **Contexto de la Rama:** `phase-d-lqr`  
> **Etiqueta de Respaldo Histórico:** `archive/phase-abc-hold-pomdp`  
> **Fecha de Análisis:** 2026-10-22  
> **Objetivo:** Auditar, clasificar y planificar la limpieza rigurosa del repositorio para consolidar el paradigma de control continuo MDP (`lyapunov_lqr`, `success_mode: none`).

---

## 1. 📈 Resumen Ejecutivo

Se analizó la totalidad de los **760 archivos versionados** en el repositorio (`git ls-files`). Cada archivo fue examinado bajo una única regla determinante: **¿depende del hold, de las zonas, de `success_mode` o de un `reward_type` distinto de `lyapunov_lqr`?**

| Categoría | Cantidad de Archivos | Porcentaje (%) | Descripción y Destino en Fase D |
| :--- | :---: | :---: | :--- |
| **BORRAR** | **722** | 95.0 % | Artefactos exclusivos de Fases A, B y C (checkpoints viejos en `experiments/`, reportes de hold, configs obsoletos, scripts de tandas). Se preservan íntegros en el tag `archive/phase-abc-hold-pomdp`. |
| **CONSERVAR** | **20** | 2.6 % | Archivos agnósticos de recompensa que sirven tal cual (cinemática pura, datasets congelados 202601/202602, configs vigentes `PPO_1`/`SAC_1`, utilidades base). |
| **ADAPTAR** | **18** | 2.4 % | Código central que debe limpiarse para remover vestigios de hold/zonas y alinearse exclusivamente con `lyapunov_lqr` (`arm2d_env.py`, `train.py`, evaluador, visualizador, tests base). |
| **TOTAL** | **760** | 100.0 % | Totalidad del árbol de trabajo auditada. |

### Desglose por Subsistema:
* **`experiments/` (504 archivos):** 100 % clasificados como **BORRAR** (504 checkpoints y metadatos de modelos de Fases A y B).
* **`reports/` (147 archivos):** 141 para **BORRAR** (análisis y figuras de hold A-C) y 6 para **CONSERVAR** (`phase_X`).
* **`configs/` (45 archivos):** 41 para **BORRAR** (fases A, B, C y pruebas viejas), 2 para **CONSERVAR** (`PPO_1.yaml`, `SAC_1.yaml`) y 2 para **ADAPTAR** (`smoke`).
* **`tests/` (11 archivos):** 9 para **BORRAR** (hold, regresión X0, suavidad X6), 1 para **CONSERVAR** (`test_kinematics.py`) y 1 para **ADAPTAR** (`test_normalization.py`).
* **`src/` (17 archivos):** 6 para **BORRAR** (visualizaciones viejas de A-C), 5 para **CONSERVAR** (cinemática, datasets, callbacks de checkpoint, estilos) y 6 para **ADAPTAR** (evaluador, métricas, render, trayectorias).
* **`scripts/` (16 archivos):** 13 para **BORRAR** (tandas viejas) y 3 para **ADAPTAR** (`evaluate_checkpoints`, `generate_figures`, `tune_and_eval_controller`).
* **`data/` (12 archivos):** 8 para **BORRAR** (datasets específicos B y C) y 4 para **CONSERVAR** (`validation_set_200`, `final_eval_set_300`).
* **Archivos Raíz y Docs (8 archivos):** 2 para **CONSERVAR** (`.gitignore`, `requirements.txt`) y 6 para **ADAPTAR** (`arm2d_env.py`, `train.py`, `interactive_render.py`, `README.md`, `docs/manual-de-uso.md`).

---

## 2. 🔬 Análisis de Dependencias del Paradigma Anterior (Paso 3)

### A. Coincidencias de Términos Clave en el Repositorio
Se realizó una búsqueda exhaustiva (`grep` case-insensitive por palabra completa) de los términos: `hold`, `hold_counter`, `hold_steps`, `success_mode`, `zone`, `zona`, `reward_type`, `reward_params`, `shaping`, `bonus`, `terminal`, `success_rate`, `tasa de éxito`. Se detectaron **191 archivos de texto** con dependencias activas del paradigma anterior.

### B. Verificaciones Explícitas Obligatorias:

#### 1. ¿La observación del entorno (`observation_space`, `_get_obs`) incluye algo derivado de `hold_counter` o de las zonas?
* **Respuesta categórica:** **NO.**
* **Evidencia técnica:** En [`arm2d_env.py:L80-L94`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/arm2d_env.py#L80-L94) y [`_get_obs()`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/arm2d_env.py#L142-L158), el vector de observaciones tiene **exactamente 10 dimensiones** puramente cinemáticas:
  $$\mathbf{o}_t = [\cos\theta_1, \sin\theta_1, \cos\theta_2, \sin\theta_2, \dot{\theta}_1, \dot{\theta}_2, x_{\text{tgt}}, y_{\text{tgt}}, \Delta x, \Delta y]^\top$$
* **Impacto en Fase D:** El espacio de observaciones **no requiere cambiar de dimensionalidad**. Se mantiene en 10 dimensiones, lo que garantiza total compatibilidad matemática con las redes de `PPO_1` y `SAC_1` ya parametrizadas.

#### 2. ¿Qué `reward_type` existen en `arm2d_env.py` y cuáles son referenciados?
* **Definidos en código ([`arm2d_env.py:L58-L72`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/arm2d_env.py#L58-L72) y L315-L343):**
  1. `distance_only`: usado en tests iniciales de normalización.
  2. `smooth_precision`: usado en Fase A, B y C (`X1a` a `X4`, `S1`, `S2`, `S3`).
  3. `historical`: usado por el modelo `X0` y tests de regresión.
  4. `x5_near`: usado por la ablación diagnóstica `X5_near`.
  5. `x5_vel`: usado por `X5` (penalización gaussiana $w_v$).
  6. `x6_smooth`: usado por `X6` (penalización de aceleración $w_a$).
  7. `energy_efficient`: variante experimental no consolidada.
  8. `lyapunov_lqr`: **Único reward vigente para Fase D** (MDP continuo).
* **Decisión Fase D:** En `arm2d_env.py` deben eliminarse los tipos 1 al 7, dejando `lyapunov_lqr` como la única función de recompensa del entorno.

#### 3. ¿Qué usa `src/evaluation/evaluator.py` para decidir éxito? ¿Depende del entorno o aplica criterios propios?
* **Respuesta:** `evaluator.py` **fuerza criterios propios hardcodeados** y sobreescribe la configuración del entorno ([`src/evaluation/evaluator.py:L31-L41`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/src/evaluation/evaluator.py#L31-L41)):
  ```python
  merged_cfg['env']['success_mode'] = 'hold'
  merged_cfg['env']['target_radius'] = 0.06
  merged_cfg['env']['velocity_tolerance'] = 0.10
  merged_cfg['env']['hold_steps'] = 20
  ```
* Además lee `configs/phase_a/thresholds.yaml` (que desaparecerá).
* **Decisión Fase D:** `evaluator.py` debe ser **ADAPTADO** para eliminar la dependencia de `thresholds.yaml` y sustituir la condición `terminated and info['is_success']` por la métrica continua que se defina.

#### 4. ¿Qué scripts de figuras o análisis usan 'éxito por hold' como métrica o leen resultados de Fases A–C?
* Todos los scripts de `reports/phase_c_s1_analysis/`, `reports/phase_c_s2_analysis/`, `reports/phase_c_s3_analysis/` y los generadores `src/visualization/phase_a.py`, `phase_b.py`, `phase_c.py` están acoplados al hold y a tablas de Fases A-C.
* **Decisión Fase D:** Se marcan para **BORRAR**, conservando únicamente `phase_custom.py`, `trajectories.py` y `common.py` adaptados para métricas LQR.

#### 5. ¿Qué tests validan hold, zonas o rewards antiguos y cuáles son independientes?
* **Independientes (CONSERVAR):** [`tests/test_kinematics.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/tests/test_kinematics.py) (cinemática analítica pura).
* **Adaptables (ADAPTAR):** [`tests/test_normalization.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/tests/test_normalization.py) (adaptar fixture para instanciar con `lyapunov_lqr`).
* **Dependientes de hold/rewards antiguos (BORRAR):** `test_hold_condition.py`, `test_historical_regression.py`, `test_smoothness.py`, `test_operational_defs.py`, `test_phase_b_validations.py`, `test_phase_b_ext_validations.py`, `test_phase_c_validations.py`, `test_aggregated_pipeline.py`.

#### 6. ¿El README, los informes y la documentación describen Fases A–C o el hold?
* Sí, [`README.md`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/README.md) y [`docs/manual-de-uso.md`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/docs/manual-de-uso.md) documentan extensamente el protocolo de hold estricto de 20 pasos y comandos de fases A-C.
* **Decisión Fase D:** Se clasifican como **ADAPTAR** para reescribir su contenido en torno a la Fase D (LQR continuo MDP).

---

## 3. ⚖️ Decisión Pendiente: Métrica de Éxito para Fase D

Al eliminar el `hold_counter` y fijar `success_mode: none` (episodios estándar de horizonte finito $T=200$ pasos), el evaluador requiere una métrica formal para comparar agentes PPO y SAC frente al controlador cinemático analítico C0. Se presentan las tres opciones para tu decisión:

### Comparativa de Opciones:

| Opción | Formulación Matemática | Significado Físico | Ventajas | Desventajas / Riesgos | Archivos Afectados |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Opción 1: Distancia final** | $d_{T} = \|\mathbf{p}_{\text{ee}}(T) - \mathbf{p}_{\text{tgt}}\| \le 0.06\text{ m}$ | Éxito binario evaluado en el último paso ($t=200$). | Muy simple y directa; coincide con el radio de tolerancia histórico $r=0.06\text{ m}$. | Un agente podría llegar rebotando o con alta velocidad en el paso 200 y contaría como éxito. | `evaluator.py`, `metrics.py`, `evaluate_checkpoints.py` |
| **Opción 2: Distancia media últimos $k$ pasos** | $\bar{d}_{k} = \frac{1}{k} \sum_{t=T-k+1}^{T} d_t \le 0.06\text{ m}$ con $\bar{v}_{k} \le 0.10\text{ rad/s}$ | Estabilización continua sostenida (ej. últimos $k=20$ pasos, $1.0\text{ s}$). | Captura la verdadera estabilidad asintótica sin requerir un contador interno en el entorno. | Requiere definir el hiperparámetro $k$ en la configuración de evaluación. | `evaluator.py`, `metrics.py`, `evaluate_checkpoints.py` |
| **Opción 3: Costo acumulado integral** | $J = \sum_{t=1}^{T} r_t = -\sum_{t=1}^{T} [w_{\text{pos}} d_t^2 + w_{\text{vel}} \|\dot{\boldsymbol{\theta}}\|^2 + w_{\text{ctrl}} \|\mathbf{a}\|^2]$ | Calidad global de la trayectoria en el sentido de Control Óptimo LQR. | Coherencia matemática 100 % con la función de valor $V(s)$ y el MDP; sin umbrales arbitrarios. | Menos intuitiva como porcentaje de éxito para presentaciones no técnicas. | `evaluator.py`, `metrics.py`, `trajectories.py` |

> [!TIP]
> **Recomendación técnica:** La **Opción 2** (distancia media y velocidad media en los últimos 20 pasos) ofrece el puente ideal: valida físicamente que el brazo frenó y se estabilizó en la meta, manteniendo el rigor del objetivo histórico pero calculado externamente sobre la trayectoria sin alterar el MDP.

---

## 4. 📋 Tabla de Clasificación Exhaustiva por Archivo

A continuación se detallan los archivos del repositorio clasificados individualmente, organizados por componentes funcionales:

### Archivos Raíz y Documentación (8 archivos)

| Ruta | Categoría | Justificación | Evidencia | Cambio requerido (solo ADAPTAR) |
| :--- | :---: | :--- | :--- | :--- |
| `.gitignore` | **CONSERVAR** | Reglas de exclusión de git; independiente del entorno. | Agnóstico | N/A |
| `README.md` | **ADAPTAR** | Documentación principal. Describe Fases A-C y protocolo de hold obsoleto. | L10-100+ (Fases A-C, hold) | Reescribir para presentar la Fase D (LQR continuo MDP), agentes PPO_1 y SAC_1, uso y reproducibilidad. |
| `arm2d_env.py` | **ADAPTAR** | Entorno central. Contiene código de hold, zonas y rewards obsoletos a eliminar. | L50-72 (hold/zonas), L315-343 (rewards obsoletos) | Eliminar success_mode hold/touch, r_near, rewards obsoletos (smooth_precision, x5_vel, x6_smooth, etc.); dejar lyapunov_lqr como único reward. |
| `docs/manual-de-uso.md` | **ADAPTAR** | Manual de usuario. Contiene comandos de Fases A-C que deben enfocarse en Fase D (LQR). | Comandos de train.py con phase_a, phase_b, phase_c | Actualizar secciones 2 y 3 para explicar el paradigma LQR, agentes PPO_1/SAC_1 y métricas continuas. |
| `docs/metodologia.md` | **ADAPTAR** | Documento en docs/. | L43: hold | Revisar y actualizar según Fase D. |
| `interactive_render.py` | **ADAPTAR** | Render interactivo en vivo. Debe adaptarse a la métrica continua y limpiar telemetría de hold. | L25+ (interfaz con entorno) | Eliminar visualización de hold_counter del HUD y mostrar error cartesiano/costo LQR en vivo. |
| `requirements.txt` | **CONSERVAR** | Dependencias del proyecto (gymnasium, stable-baselines3, torch, etc.). | Agnóstico | N/A |
| `train.py` | **ADAPTAR** | Script de entrenamiento unificado. Debe fijar y validar configs LQR y limpiar callbacks antiguos. | L106-124 (resolve_phase), L140+ (callbacks Fase A) | Limpiar callbacks obsoletos de hold (PhaseATrainingCallback); registrar métricas continuas de retorno LQR. |

### Configuraciones (`configs/`) (45 archivos)

| Ruta | Categoría | Justificación | Evidencia | Cambio requerido (solo ADAPTAR) |
| :--- | :---: | :--- | :--- | :--- |
| `configs/phase_X/prueba/PPO_1.yaml` | **CONSERVAR** | Configuraciones vigentes del nuevo paradigma LQR (PPO_1.yaml y SAC_1.yaml). | reward_type: lyapunov_lqr, success_mode: none | N/A |
| `configs/phase_X/prueba/SAC_1.yaml` | **CONSERVAR** | Configuraciones vigentes del nuevo paradigma LQR (PPO_1.yaml y SAC_1.yaml). | reward_type: lyapunov_lqr, success_mode: none | N/A |
| `configs/phase_X/smoke/s1_smoke.yaml` | **ADAPTAR** | Smoke tests rápidos para CI/CD. Actualmente apuntan a configuraciones antiguas (s1 y x3). | reward_type: smooth_precision | Actualizar s1_smoke.yaml y x3_smoke.yaml a s1_lqr_smoke y ppo_lqr_smoke usando lyapunov_lqr. |
| `configs/phase_X/smoke/x3_smoke.yaml` | **ADAPTAR** | Smoke tests rápidos para CI/CD. Actualmente apuntan a configuraciones antiguas (s1 y x3). | reward_type: smooth_precision | Actualizar s1_smoke.yaml y x3_smoke.yaml a s1_lqr_smoke y ppo_lqr_smoke usando lyapunov_lqr. |
| `configs/phase_X/tomi/ppo_x3_v1.yaml` | **BORRAR** | Experimentos exploratorios personales basados en smooth_precision y x3. | reward_type: smooth_precision | N/A |
| `configs/phase_X/tomi/sac_smooth_v1.yaml` | **BORRAR** | Experimentos exploratorios personales basados en smooth_precision y x3. | reward_type: smooth_precision | N/A |
| `configs/phase_a/X1a.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L4: hold | N/A |
| `configs/phase_a/X1b.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L4: hold | N/A |
| `configs/phase_a/X1c.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L4: hold | N/A |
| `configs/phase_a/X1s.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L4: hold | N/A |
| `configs/phase_a/X2.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_a/X2b.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_a/X3.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_a/X4.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_a/X5.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_a/X5_near.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_a/X5_wv01.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_a/X5_wv10.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_a/X5_wv50.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_a/X6.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_a/X6_wa001.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_a/X6_wa01.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_a/X6_wa05.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_a/thresholds.yaml` | **BORRAR** | Configuración de phase_a basada estrictamente en hold, zonas y rewards obsoletos. | L15: hold_steps | N/A |
| `configs/phase_b/X3.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L15: hold | N/A |
| `configs/phase_b/X5_near.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_b/extension/X1a.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L4: hold | N/A |
| `configs/phase_b/extension/X1b.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L4: hold | N/A |
| `configs/phase_b/extension/X1c.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L4: hold | N/A |
| `configs/phase_b/extension/X1s.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L4: hold | N/A |
| `configs/phase_b/extension/X2.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_b/extension/X2b.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_b/extension/X3.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_b/extension/X4.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_b/extension/X5.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_b/extension/X5_near.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_b/extension/X6.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_b/extension/selection_rule.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L24: terminal | N/A |
| `configs/phase_b/selection_rule.yaml` | **BORRAR** | Configuración de phase_b basada estrictamente en hold, zonas y rewards obsoletos. | L20: hold_steps | N/A |
| `configs/phase_c/S1.yaml` | **BORRAR** | Configuración de phase_c basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_c/S1_smooth_200.yaml` | **BORRAR** | Configuración de phase_c basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_c/S2_freq.yaml` | **BORRAR** | Configuración de phase_c basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_c/S2_wa.yaml` | **BORRAR** | Configuración de phase_c basada estrictamente en hold, zonas y rewards obsoletos. | L14: hold | N/A |
| `configs/phase_c/S3_combined.yaml` | **BORRAR** | Configuración de phase_c basada estrictamente en hold, zonas y rewards obsoletos. | L15: hold | N/A |
| `configs/phase_c/selection_rule_s3_combined.yaml` | **BORRAR** | Configuración de phase_c basada estrictamente en hold, zonas y rewards obsoletos. | L13: hold_steps | N/A |

### Código Fuente (`src/`) (17 archivos)

| Ruta | Categoría | Justificación | Evidencia | Cambio requerido (solo ADAPTAR) |
| :--- | :---: | :--- | :--- | :--- |
| `src/controllers/jacobian_controller.py` | **CONSERVAR** | Controlador analítico Jacobiano (C0). Baseline clásico sin aprendizaje independiente del reward. | Cinemática analítica DLS | N/A |
| `src/evaluation/checkpoints.py` | **ADAPTAR** | Módulo de evaluación y métricas. Actualmente fuerza hold_steps=20 y thresholds de Fase A. | src/evaluation/checkpoints.py: hardcodea success_mode='hold' y thresholds.yaml | Reemplazar el criterio de éxito estricto por la métrica de convergencia continua LQR que se defina. |
| `src/evaluation/datasets.py` | **CONSERVAR** | Generador de episodios con distribución uniforme. Cinemática e inicialización independientes de recompensa. | theta1, theta2, target_r ~ U(0.15, 1.90) | N/A |
| `src/evaluation/evaluator.py` | **ADAPTAR** | Módulo de evaluación y métricas. Actualmente fuerza hold_steps=20 y thresholds de Fase A. | src/evaluation/evaluator.py: hardcodea success_mode='hold' y thresholds.yaml | Reemplazar el criterio de éxito estricto por la métrica de convergencia continua LQR que se defina. |
| `src/evaluation/metrics.py` | **ADAPTAR** | Módulo de evaluación y métricas. Actualmente fuerza hold_steps=20 y thresholds de Fase A. | src/evaluation/metrics.py: hardcodea success_mode='hold' y thresholds.yaml | Reemplazar el criterio de éxito estricto por la métrica de convergencia continua LQR que se defina. |
| `src/training/callbacks.py` | **ADAPTAR** | Callbacks de entrenamiento y evaluación en línea con telemetría de hold_counter_max. | L16+: hold_counter_max, is_success | Limpiar métricas de hold; registrar retorno acumulado continuo y error cuadrático en TensorBoard. |
| `src/training/phase_b_callbacks.py` | **CONSERVAR** | Guarda atómicamente checkpoints (model.zip + vecnormalize_stats.pkl); lógica independiente del reward. | Guardado sincronizado agnóstico de reward | N/A |
| `src/visualization/__init__.py` | **CONSERVAR** | Utilidades comunes de estilo y guardado de figuras requeridas por los scripts de visualización. | Importado por phase_custom.py | N/A |
| `src/visualization/common.py` | **CONSERVAR** | Utilidades comunes de estilo y guardado de figuras requeridas por los scripts de visualización. | Importado por phase_custom.py | N/A |
| `src/visualization/phase_a.py` | **BORRAR** | Módulo de figuras exclusivo de Fase A, B o C (phase_a). | src/visualization/phase_... | N/A |
| `src/visualization/phase_a_agg.py` | **BORRAR** | Módulo de figuras exclusivo de Fase A, B o C (phase_a_agg). | src/visualization/phase_... | N/A |
| `src/visualization/phase_b.py` | **BORRAR** | Módulo de figuras exclusivo de Fase A, B o C (phase_b). | src/visualization/phase_... | N/A |
| `src/visualization/phase_b_ext.py` | **BORRAR** | Módulo de figuras exclusivo de Fase A, B o C (phase_b_ext). | src/visualization/phase_... | N/A |
| `src/visualization/phase_c.py` | **BORRAR** | Módulo de figuras exclusivo de Fase A, B o C (phase_c). | src/visualization/phase_... | N/A |
| `src/visualization/phase_c_s3.py` | **BORRAR** | Módulo de figuras exclusivo de Fase A, B o C (phase_c_s3). | src/visualization/phase_... | N/A |
| `src/visualization/phase_custom.py` | **ADAPTAR** | Graficado de trayectorias cinemáticas y figuras personalizadas; útil para Fase D. | src/visualization/phase_custom.py: graficado cinemático | Eliminar anotaciones de hold_steps y umbrales de zona near; enfocar en convergencia LQR. |
| `src/visualization/trajectories.py` | **ADAPTAR** | Graficado de trayectorias cinemáticas y figuras personalizadas; útil para Fase D. | src/visualization/trajectories.py: graficado cinemático | Eliminar anotaciones de hold_steps y umbrales de zona near; enfocar en convergencia LQR. |

### Tests Automatizados (`tests/`) (11 archivos)

| Ruta | Categoría | Justificación | Evidencia | Cambio requerido (solo ADAPTAR) |
| :--- | :---: | :--- | :--- | :--- |
| `tests/data/historical_regression_reference.json` | **BORRAR** | Test que valida exclusivamente comportamiento de Fases A-C (hold, zonas, regresión X0, suavidad X6). | Sin mención de términos | N/A |
| `tests/record_historical_baseline.py` | **BORRAR** | Test que valida exclusivamente comportamiento de Fases A-C (hold, zonas, regresión X0, suavidad X6). | Sin mención de términos | N/A |
| `tests/test_aggregated_pipeline.py` | **BORRAR** | Test que valida exclusivamente comportamiento de Fases A-C (hold, zonas, regresión X0, suavidad X6). | L9: tasa de éxito | N/A |
| `tests/test_hold_condition.py` | **BORRAR** | Test que valida exclusivamente comportamiento de Fases A-C (hold, zonas, regresión X0, suavidad X6). | L2: hold | N/A |
| `tests/test_kinematics.py` | **CONSERVAR** | Valida cinemática directa y límites geométricos (L1=1, L2=1, r in [0, 2]); matemática pura. | Sin dependencias de reward ni hold | N/A |
| `tests/test_normalization.py` | **ADAPTAR** | Valida VecNormalize(obs), pero usa configs antiguos (distance_only) y hold. | L20: reward_type: distance_only | Adaptar fixtures para validar normalización con lyapunov_lqr y horizonte fijo. |
| `tests/test_operational_defs.py` | **BORRAR** | Test que valida exclusivamente comportamiento de Fases A-C (hold, zonas, regresión X0, suavidad X6). | Sin mención de términos | N/A |
| `tests/test_phase_b_ext_validations.py` | **BORRAR** | Test que valida exclusivamente comportamiento de Fases A-C (hold, zonas, regresión X0, suavidad X6). | Sin mención de términos | N/A |
| `tests/test_phase_b_validations.py` | **BORRAR** | Test que valida exclusivamente comportamiento de Fases A-C (hold, zonas, regresión X0, suavidad X6). | L70: hold | N/A |
| `tests/test_phase_c_validations.py` | **BORRAR** | Test que valida exclusivamente comportamiento de Fases A-C (hold, zonas, regresión X0, suavidad X6). | L153: hold | N/A |
| `tests/test_smoothness.py` | **BORRAR** | Test que valida exclusivamente comportamiento de Fases A-C (hold, zonas, regresión X0, suavidad X6). | L16: reward_type | N/A |

### Scripts Auxiliares y Ejecutables (`scripts/`) (16 archivos)

| Ruta | Categoría | Justificación | Evidencia | Cambio requerido (solo ADAPTAR) |
| :--- | :---: | :--- | :--- | :--- |
| `scripts/evaluate_checkpoints.py` | **ADAPTAR** | Evaluador batch de checkpoints. Debe evaluar modelos LQR contra la métrica continua. | L1+: dependencias de evaluación | Actualizar para cargar configs LQR y registrar métricas continuas sin hold. |
| `scripts/generate_figures.py` | **ADAPTAR** | Orquestador de generación de figuras. Actualmente ejecuta scripts de Fases A-C. | L20+: llamadas a phase_a, phase_b, phase_c | Limpiar llamadas a Fases A-C y dejar únicamente generación de figuras de Fase D (custom/LQR). |
| `scripts/phase_b_batch_eval_validation.py` | **BORRAR** | Script de ejecución o preparación de datos exclusivo de Fase B o C (phase_b_batch_eval_validation). | Sin mención de términos | N/A |
| `scripts/phase_b_ext_apply_selection.py` | **BORRAR** | Script de ejecución o preparación de datos exclusivo de Fase B o C (phase_b_ext_apply_selection). | Sin mención de términos | N/A |
| `scripts/phase_b_ext_batch_eval_validation.py` | **BORRAR** | Script de ejecución o preparación de datos exclusivo de Fase B o C (phase_b_ext_batch_eval_validation). | Sin mención de términos | N/A |
| `scripts/phase_b_ext_prepare_datasets.py` | **BORRAR** | Script de ejecución o preparación de datos exclusivo de Fase B o C (phase_b_ext_prepare_datasets). | Sin mención de términos | N/A |
| `scripts/phase_b_ext_run_b12_final_eval.py` | **BORRAR** | Script de ejecución o preparación de datos exclusivo de Fase B o C (phase_b_ext_run_b12_final_eval). | L2: terminal | N/A |
| `scripts/phase_b_ext_step_data.py` | **BORRAR** | Script de ejecución o preparación de datos exclusivo de Fase B o C (phase_b_ext_step_data). | L34: hold | N/A |
| `scripts/phase_b_final_evaluation.py` | **BORRAR** | Script de ejecución o preparación de datos exclusivo de Fase B o C (phase_b_final_evaluation). | Sin mención de términos | N/A |
| `scripts/phase_b_prepare_eval_and_telemetry.py` | **BORRAR** | Script de ejecución o preparación de datos exclusivo de Fase B o C (phase_b_prepare_eval_and_telemetry). | Sin mención de términos | N/A |
| `scripts/phase_b_step_data.py` | **BORRAR** | Script de ejecución o preparación de datos exclusivo de Fase B o C (phase_b_step_data). | L4: hold | N/A |
| `scripts/phase_c_eval_stochastic_diagnostic.py` | **BORRAR** | Script de ejecución o preparación de datos exclusivo de Fase B o C (phase_c_eval_stochastic_diagnostic). | L22: hold | N/A |
| `scripts/phase_c_eval_tanda1_s2_freq.py` | **BORRAR** | Script de ejecución o preparación de datos exclusivo de Fase B o C (phase_c_eval_tanda1_s2_freq). | L5: hold_steps | N/A |
| `scripts/phase_c_prepare_datasets.py` | **BORRAR** | Script de ejecución o preparación de datos exclusivo de Fase B o C (phase_c_prepare_datasets). | Sin mención de términos | N/A |
| `scripts/phase_c_step_data.py` | **BORRAR** | Script de ejecución o preparación de datos exclusivo de Fase B o C (phase_c_step_data). | L30: hold | N/A |
| `scripts/tune_and_eval_controller.py` | **ADAPTAR** | Sintonización y evaluación del controlador cinemático C0. | Evaluación de baseline C0 | Adaptar función de evaluación a métricas continuas LQR para benchmarking. |

### Datasets y Datos (`data/`) (12 archivos)

| Ruta | Categoría | Justificación | Evidencia | Cambio requerido (solo ADAPTAR) |
| :--- | :---: | :--- | :--- | :--- |
| `data/datasets/final_eval_set_300.csv` | **CONSERVAR** | Datasets canónicos de evaluación (semillas 202601 y 202602). Cinemática pura independiente del reward. | Cinemática pura (r_tgt, phi, theta1, theta2) | N/A |
| `data/datasets/final_eval_set_300.json` | **CONSERVAR** | Datasets canónicos de evaluación (semillas 202601 y 202602). Cinemática pura independiente del reward. | Cinemática pura (r_tgt, phi, theta1, theta2) | N/A |
| `data/datasets/phase_b_ext/final_b2_500.csv` | **BORRAR** | Datasets y splits específicos de Fases B o C. | Ruta data/datasets/phase_b_ext/final_b2_500.csv | N/A |
| `data/datasets/phase_b_ext/final_b2_500.json` | **BORRAR** | Datasets y splits específicos de Fases B o C. | Ruta data/datasets/phase_b_ext/final_b2_500.json | N/A |
| `data/datasets/phase_b_ext/validation_b2_500.csv` | **BORRAR** | Datasets y splits específicos de Fases B o C. | Ruta data/datasets/phase_b_ext/validation_b2_500.csv | N/A |
| `data/datasets/phase_b_ext/validation_b2_500.json` | **BORRAR** | Datasets y splits específicos de Fases B o C. | Ruta data/datasets/phase_b_ext/validation_b2_500.json | N/A |
| `data/datasets/phase_c/final_c_500.csv` | **BORRAR** | Datasets y splits específicos de Fases B o C. | Ruta data/datasets/phase_c/final_c_500.csv | N/A |
| `data/datasets/phase_c/final_c_500.json` | **BORRAR** | Datasets y splits específicos de Fases B o C. | Ruta data/datasets/phase_c/final_c_500.json | N/A |
| `data/datasets/phase_c/validation_c_500.csv` | **BORRAR** | Datasets y splits específicos de Fases B o C. | Ruta data/datasets/phase_c/validation_c_500.csv | N/A |
| `data/datasets/phase_c/validation_c_500.json` | **BORRAR** | Datasets y splits específicos de Fases B o C. | Ruta data/datasets/phase_c/validation_c_500.json | N/A |
| `data/datasets/validation_set_200.csv` | **CONSERVAR** | Datasets canónicos de evaluación (semillas 202601 y 202602). Cinemática pura independiente del reward. | Cinemática pura (r_tgt, phi, theta1, theta2) | N/A |
| `data/datasets/validation_set_200.json` | **CONSERVAR** | Datasets canónicos de evaluación (semillas 202601 y 202602). Cinemática pura independiente del reward. | Cinemática pura (r_tgt, phi, theta1, theta2) | N/A |

### Reportes, Figuras y Análisis (`reports/`) (147 archivos)

| Ruta | Categoría | Justificación | Evidencia | Cambio requerido (solo ADAPTAR) |
| :--- | :---: | :--- | :--- | :--- |
| `reports/EXPERIMENTS_SUMMARY.md` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L5: hold | N/A |
| `reports/data/aggregated/acceptance_criteria_compliance.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/aggregated/episode_outcomes_summary.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/aggregated/master_comparison_table_aggregated.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/aggregated/master_comparison_table_aggregated_per_seed.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: success_rate | N/A |
| `reports/data/all_final_deterministic_episodes.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/all_final_stochastic_episodes.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/confirmation_all_seeds_results.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/master_comparison_table.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_X/checkpoint_eval.csv` | **CONSERVAR** | Metadatos y figuras preliminares de Fase X (paradigma LQR). | Fase X | N/A |
| `reports/data/phase_X/comparativa.csv` | **CONSERVAR** | Metadatos y figuras preliminares de Fase X (paradigma LQR). | Fase X | N/A |
| `reports/data/phase_b/checkpoints_eval_validation.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_b/checkpoints_eval_validation_aggregated.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_b/eval_val_c0_episodes.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_b/eval_val_x3_phase_b_selected.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_b/eval_val_x5_near_phase_a_det.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_b/eval_val_x5_near_phase_b_selected.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_b/extension/checkpoints_eval_validation.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_b/extension/checkpoints_eval_validation_aggregated.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_b/extension/final_b2_eval_aggregated.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_b/extension/final_b2_eval_details.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_b/extension/phase_b_report.ORIGINAL.md` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L14: hold | N/A |
| `reports/data/phase_b/extension/selected_checkpoints.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_b/extension/step_data_val_geometry.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: hold_counter | N/A |
| `reports/data/phase_b/final_eval_all_episodes.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_b/final_eval_summary.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_b/phase_a_models_val_comparison.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: success_rate | N/A |
| `reports/data/phase_b/step_data_final_geometry.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: hold_counter | N/A |
| `reports/data/phase_b/step_data_val_geometry.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: hold_counter | N/A |
| `reports/data/phase_c/checkpoints_eval_validation_aggregated.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/data/phase_c/checkpoints_eval_validation_details.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: success_rate | N/A |
| `reports/data/phase_c/step_data_val_geometry.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: hold_counter | N/A |
| `reports/data/phase_c/stochastic_diagnostic_final_ckpt.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: success_rate | N/A |
| `reports/figures/phase_X/figX_01_training_reward_curves.json` | **CONSERVAR** | Metadatos y figuras preliminares de Fase X (paradigma LQR). | Fase X | N/A |
| `reports/figures/phase_X/figX_02_periodic_evaluation_curve.json` | **CONSERVAR** | Metadatos y figuras preliminares de Fase X (paradigma LQR). | Fase X | N/A |
| `reports/figures/phase_X/figX_03_episode_rewards_scatter.json` | **CONSERVAR** | Metadatos y figuras preliminares de Fase X (paradigma LQR). | Fase X | N/A |
| `reports/figures/phase_X/figX_05_checkpoint_success_curve.json` | **CONSERVAR** | Metadatos y figuras preliminares de Fase X (paradigma LQR). | Fase X | N/A |
| `reports/phase_a_report.md` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L15: hold | N/A |
| `reports/phase_b_report.md` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L14: hold | N/A |
| `reports/phase_c_report.md` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L14: hold | N/A |
| `reports/phase_c_s1_analysis/README.md` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L41: hold | N/A |
| `reports/phase_c_s1_analysis/alpha_tensorboard_por_semilla.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/phase_c_s1_analysis/analisis_deterioro.md` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/phase_c_s1_analysis/analisis_espacial.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: success_rate | N/A |
| `reports/phase_c_s1_analysis/audit_table.py` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L31: success_rate | N/A |
| `reports/phase_c_s1_analysis/auditoria_implementacion.md` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L67: zona | N/A |
| `reports/phase_c_s1_analysis/comparacion_x3_vs_s1_identicos.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: success_rate | N/A |
| `reports/phase_c_s1_analysis/diagnostico_s1.md` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L19: tasa de éxito | N/A |
| `reports/phase_c_s1_analysis/execute_all_pending.py` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L141: hold_counter | N/A |
| `reports/phase_c_s1_analysis/exito_criterio_entrenamiento_vs_estricto.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/phase_c_s1_analysis/format_table.py` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L7: hold | N/A |
| `reports/phase_c_s1_analysis/metricas_checkpoints.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: success_rate | N/A |
| `reports/phase_c_s1_analysis/metricas_trayectorias.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/phase_c_s1_analysis/oscilacion_borde.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/phase_c_s1_analysis/perfiles_fallos_seleccionados.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: hold_counter | N/A |
| `reports/phase_c_s1_analysis/reproduce_diagnosis.py` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L119: hold_counter | N/A |
| `reports/phase_c_s1_analysis/sigma_efectiva_por_semilla.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/phase_c_s1_analysis/tabla_4_1_corregida.md` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: hold | N/A |
| `reports/phase_c_s2_analysis/metricas_checkpoints_s2_freq.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: success_rate | N/A |
| `reports/phase_c_s2_analysis/metricas_checkpoints_s2_freq_agg.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/phase_c_s2_analysis/metricas_checkpoints_s2_wa.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: success_rate | N/A |
| `reports/phase_c_s2_analysis/metricas_checkpoints_s2_wa_agg.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/phase_c_s2_analysis/tabla_comparativa_245k_final.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: success_rate | N/A |
| `reports/phase_c_s3_analysis/informe_cierre_fase_c.md` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L10: tasa de éxito | N/A |
| `reports/phase_c_s3_analysis/informe_piloto_s3_combined.md` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L11: reward_params | N/A |
| `reports/phase_c_s3_analysis/metricas_checkpoints_s3_combined_agg.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/phase_c_s3_analysis/metricas_checkpoints_s3_combined_all_seeds.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: success_rate | N/A |
| `reports/phase_c_s3_analysis/metricas_checkpoints_s3_combined_seed42.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: success_rate | N/A |
| `reports/phase_c_s3_analysis/tabla_comparativa_3seeds_245k.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | Sin mención de términos | N/A |
| `reports/phase_c_s3_analysis/tabla_comparativa_seed42.csv` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L1: success_rate | N/A |
| `reports/phase_c_s3_analysis/tabla_palancas.md` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L13: tasa de éxito | N/A |
| `reports/phase_c_s3_analysis/verificaciones_etapa1.md` | **BORRAR** | Informe, análisis detallado o figura de Fases A-C basados en éxito por hold. | L38: hold | N/A |
| `reports/figures/phase_b,c,agg/*.json` (75 archivos) | **BORRAR** | Metadatos y datos de figuras de Fases A-C. | Figuras de hold | N/A |

### Modelos y Checkpoints (`experiments/`) (504 archivos)

| Ruta | Categoría | Justificación | Evidencia | Cambio requerido (solo ADAPTAR) |
| :--- | :---: | :--- | :--- | :--- |
| `experiments/phase_a/*` (101 archivos) | **BORRAR** | Checkpoints, pesos y metadatos de semillas de Fase A. | Modelos hold POMDP | N/A (Preservado en tag) |
| `experiments/phase_b/*` (403 archivos) | **BORRAR** | Checkpoints, pesos y metadatos de semillas de Fase B y extensión. | Modelos hold POMDP | N/A (Preservado en tag) |

---

## 5. 🛠️ Orden de Ejecución Propuesto (Un Commit por Paso)

Para garantizar que la rama `phase-d-lqr` se mantenga siempre funcional, comprobable y sin rupturas intermedias, se propone la siguiente secuencia de commits:

1. **Commit 1: Entorno LQR Puro (`arm2d_env.py`)**
   * Depurar `arm2d_env.py`: retirar `success_mode` hold/touch, contadores de permanencia, $r_{\text{near}}$ no lineal y rewards antiguos (`smooth_precision`, `x5_near`, `x5_vel`, `x6_smooth`).
   * Establecer `lyapunov_lqr` como la formulación de recompensa central del entorno.

2. **Commit 2: Pipeline de Entrenamiento y Smoke Tests (`train.py`, `configs/phase_X/smoke/`)**
   * Depurar `train.py`: eliminar callbacks de Fase A vinculados a hold.
   * Actualizar smoke tests (`s1_smoke.yaml` y `x3_smoke.yaml`) para verificar el entrenamiento de PPO y SAC con `lyapunov_lqr` en 1 000 timesteps.
   * Ejecutar smoke tests de validación.

3. **Commit 3: Evaluador y Métricas Continuas (`src/evaluation/`)**
   * Implementar la métrica de éxito seleccionada en `src/evaluation/evaluator.py` y `metrics.py`.
   * Eliminar la carga forzada de `configs/phase_a/thresholds.yaml`.
   * Validar evaluación determinista con `datasets.py`.

4. **Commit 4: Visualización y Figuras (`src/visualization/`, `scripts/`)**
   * Adaptar `phase_custom.py` y `trajectories.py` para graficar curvas de convergencia LQR, perfiles cartesianos y acción.
   * Ajustar `generate_figures.py` para orquestar únicamente las figuras de Fase D.

5. **Commit 5: Borrado Masivo de Fases A–C**
   * Eliminar los 722 archivos obsoletos identificados (`experiments/phase_a`, `experiments/phase_b`, `reports/phase_a,b,c`, `configs/phase_a,b,c`, scripts viejos).
   * Verificar que el repositorio quede limpio y libre de código muerto.

6. **Commit 6: Tests Automatizados y Documentación (`tests/`, `README.md`, `manual-de-uso.md`)**
   * Adaptar `test_normalization.py` y `test_kinematics.py`.
   * Eliminar tests viejos y añadir tests unitarios específicos para `reward_type: lyapunov_lqr`.
   * Reescribir `README.md` y `docs/manual-de-uso.md` reflejando el estado final de Fase D.
   * Ejecución exitosa de `python -m unittest discover tests`.

---

## 6. ⚠️ Análisis de Riesgos y Mitigaciones

| Riesgo Identificado | Causa Potencial | Impacto | Estrategia de Mitigación |
| :--- | :--- | :---: | :--- |
| **Import cruzado roto en visualización** | Borrar `src/visualization/common.py` pensando que era de Fase A. | Alto | Clasificado explícitamente como **CONSERVAR**; `phase_custom.py` depende de sus utilidades de estilo. |
| **Fallo de importación por rutas hardcodeadas** | `evaluator.py` busca `configs/phase_a/thresholds.yaml` al evaluar un modelo. | Crítico | El Commit 3 desacopla `evaluator.py` de `thresholds.yaml` antes de borrar la carpeta `configs/phase_a/` en el Commit 5. |
| **Tests que intentan instanciar rewards obsoletos** | `test_normalization.py` usa `distance_only` por defecto. | Medio | El Commit 6 actualiza los fixtures de prueba a `lyapunov_lqr` simultáneamente con la limpieza de tests. |
| **Pérdida accidental de resultados históricos** | Borrado de 722 archivos. | Nulo | Garantizado mediante la etiqueta inmutable `archive/phase-abc-hold-pomdp` creada en el commit `5772751`. |
