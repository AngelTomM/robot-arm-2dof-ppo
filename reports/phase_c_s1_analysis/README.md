# Diagnóstico Integral de SAC S1 (Fase C) — Guía y Reproducibilidad

**Directorio:** `reports/phase_c_s1_analysis/`  
**Proyecto:** `robot-arm-2dof-ppo`  
**Fecha:** Octubre 2026  
**Estado:** Diagnóstico completo, de solo lectura, reproducible y aislado.

---

## 1. Objetivo del Diagnóstico

El objetivo de este diagnóstico técnico fue determinar mediante evidencia empírica verificable las causas del rendimiento insatisfactorio y del deterioro final del agente Soft Actor-Critic (SAC) variante S1 en la Fase C, evaluar el comportamiento espacial y temporal de sus políticas, y formular recomendaciones experimentales fundamentadas para la siguiente fase de desarrollo.

---

## 2. Artefactos Originales Utilizados (Modo Solo Lectura)

- **Entorno y Dinámica:** [arm2d_env.py](../../arm2d_env.py)
- **Evaluador Histórico:** [src/evaluation/evaluator.py](../../src/evaluation/evaluator.py)
- **Configuración de S1:** [configs/phase_c/S1.yaml](../../configs/phase_c/S1.yaml)
- **Criterios de Evaluación Estricta:** [configs/phase_a/thresholds.yaml](../../configs/phase_a/thresholds.yaml)
- **Dataset de Validación Empleado:** `data/datasets/phase_c/validation_c_500.json` (500 episodios deterministas).
- **Checkpoints y Estadísticas:**
  - `experiments/phase_c/S1/seed_42/checkpoints/ckpt_*/`
  - `experiments/phase_c/S1/seed_100/checkpoints/ckpt_*/`
  - `experiments/phase_c/S1/seed_2026/checkpoints/ckpt_*/`
- **Logs de Entrenamiento:**
  - `experiments/phase_c/S1/seed_*/logs/SAC_1/events.out.tfevents.*`

> [!IMPORTANT]
> **Prohibición Expresa Cumplida:** El dataset `data/datasets/phase_c/final_c_500.json` **no fue abierto, leído ni utilizado** en ningún momento durante este análisis.

---

## 3. Criterio de Éxito y Protocolo de Evaluación

La evaluación se realizó estrictamente bajo el protocolo de Fase A y Fase C:
- **Modo:** Determinista ($\mathbf{a}_t = \tanh(\boldsymbol{\mu}(\mathbf{s}_t))$).
- **Radio Objetivo ($r_{\text{tgt}}$):** $0{,}06\text{ m}$.
- **Tolerancia de Velocidad Articular ($v_{\text{tol}}$):** $0{,}10\text{ rad/s}$ ($v_{\text{joint}} = \sqrt{\dot{\theta}_1^2 + \dot{\theta}_2^2}$).
- **Permanencia Requerida ($N_{\text{hold}}$):** $20\text{ pasos consecutivos}$ ($1{,}00\text{ s}$ con $\Delta t = 0{,}05\text{ s}$).
- **Límite de Episodio:** $200\text{ pasos}$ ($10{,}0\text{ s}$).
- **Condición de Reinicio:** Si el efector sale de $r_{\text{tgt}}$ o la velocidad supera $v_{\text{tol}}$, el contador de hold se reinicia a 0.

---

## 4. Diccionario de Métricas

1. **`success_rate`:** Fracción de episodios que completan $N_{\text{hold}} \ge 20$ dentro de los 200 pasos.
2. **`truncated_rate`:** Fracción de episodios que alcanzan 200 pasos sin completar el éxito ($1 - \text{success\_rate}$).
3. **`never_entered_rate`:** Fracción de episodios donde la distancia mínima al objetivo es estrictamente mayor o igual a $0{,}06\text{ m}$ ($d_{\min} \ge 0{,}06\text{ m}$).
4. **`entered_no_hold_rate`:** Fracción de episodios que lograron entrar al radio ($d_{\min} < 0{,}06\text{ m}$) pero nunca acumularon 20 pasos consecutivos de hold con $v_{\text{joint}} < 0{,}10\text{ rad/s}$.
5. **`min_distance`:** Menor distancia euclidiana entre el efector final y el objetivo a lo largo del episodio ($m$).
6. **`final_distance`:** Distancia en el último paso ejecutado del episodio ($m$).
7. **`steps_to_first_entry`:** Paso temporal en que el efector entra por primera vez a $d < 0{,}06\text{ m}$ (definido solo para episodios que entran; `NaN` en caso contrario).
8. **`steps_to_success`:** Paso temporal en el que se completa el paso 20 de hold consecutivo (`NaN` si no hubo éxito).
9. **`hold_counter_max`:** Máxima racha consecutiva de pasos satisfaciendo simultáneamente distancia y velocidad dentro del episodio.
10. **`joint_speed_at_first_entry`:** Magnitud de la velocidad articular instantánea al entrar por primera vez al radio objetivo.
11. **`action_delta_mean` ($\overline{\|\Delta \mathbf{a}\|}$):** Promedio de la norma euclidiana de la variación entre acciones consecutivas:
    $$\overline{\|\Delta \mathbf{a}\|} = \frac{1}{T-1} \sum_{t=1}^{T-1} \|\mathbf{a}_{t+1} - \mathbf{a}_t\|_2$$
12. **`ballistic_rate`:** Fracción de episodios donde $v_{\text{joint\_entry}} > 0{,}40\text{ rad/s}$.
13. **`oscillatory_rate`:** Fracción de episodios con $\ge 3$ salidas del radio objetivo o $\ge 3$ cruces de signo de vector de error en zona cercana ($d < 0{,}30\text{ m}$).

---

## 5. Convenciones Estadísticas

- **Unidad de Agregación:** La semilla ($N=3$). Primero se calculan las métricas sobre los 500 episodios de cada semilla y checkpoint; posteriormente se computa la media y la desviación estándar muestral con corrección de Bessel (`ddof=1`):
  $$s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}$$
- No se agrupan los 1500 episodios como si fueran independientes.

---

## 6. Inventario de Archivos Generados

Todos los archivos generados están confinados estrictamente en este directorio:

```text
reports/phase_c_s1_analysis/
├── README.md                          <- Esta guía técnica y diccionario
├── auditoria_implementacion.md        <- Auditoría exhaustiva de código y discrepancias
├── diagnostico_s1.md                  <- Informe técnico integral con respuestas a preguntas clave
├── metricas_checkpoints.csv           <- 48 filas: métricas agregadas por checkpoint y semilla
├── analisis_espacial.csv              <- 144 filas: métricas desglosadas por región (near, mid, far)
├── metricas_trayectorias.csv          <- 21 filas: episodios representativos clasificados
├── reproduce_diagnosis.py             <- Script ejecutable autónomo y reproducible de evaluación
└── figures/                           <- 11 figuras de diagnóstico técnico
    ├── evolucion_exito.png            <- Figura 1: Éxito estricto vs pasos por semilla y media
    ├── evolucion_truncamiento.png     <- Figura 2: Truncamiento vs pasos
    ├── comparacion_semillas.png       <- Figura 3: Panel 4 métricas comparativas entre semillas
    ├── comparacion_mejor_final.png    <- Figura 4: Barras comparativas Pico vs Checkpoint Final
    ├── rendimiento_espacial.png       <- Figura 5: Éxito por región radial a lo largo del tiempo
    ├── mapa_exitos_fallos.png         <- Figura 6: Mapa 2D de objetivos (Pico vs Final)
    ├── trayectorias_representativas.png <- Figura 7: Trazas cartesianas 2D del efector final
    ├── distancia_temporal.png         <- Figura 8: Evolución d(t) en episodios característicos
    ├── velocidad_articular.png        <- Figura 9: Perfiles temporales de velocidad articular
    ├── acciones_temporales.png        <- Figura 10: Trazado de acciones y chattering
    └── indicadores_entrenamiento.png  <- Figura 11: Dinámica de TensorBoard (alpha, pérdidas)
```

---

## 7. Instrucciones para Reproducir el Diagnóstico

El análisis completo puede reproducirse en cualquier momento ejecutando el script auxiliar autónomo:

```powershell
# Desde la raíz del repositorio
.venv/Scripts/python.exe reports/phase_c_s1_analysis/reproduce_diagnosis.py
```

El script:
1. No escribe fuera de `reports/phase_c_s1_analysis/`.
2. No modifica ningún modelo, log ni dataset original.
3. Se ejecuta de manera determinista y vectorizada en aproximadamente 20 segundos.

