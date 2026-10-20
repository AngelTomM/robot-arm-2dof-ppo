# Informe Técnico de Diagnóstico Integral — SAC S1 (Fase C)

**Proyecto:** `robot-arm-2dof-ppo`  
**Experimento Diagnosticado:** SAC S1 (`smooth_precision`, VecNormalize, red $256 \times 256$, ReLU)  
**Semillas Analizadas:** 42, 100, 2026 (3 réplicas independientes)  
**Checkpoints Evaluados:** 16 checkpoints periódicos por semilla ($40\,960$ a $655\,360$ pasos, paso $40\,960$)  
**Dataset Empleado:** Exclusivamente `data/datasets/phase_c/validation_c_500.json` (500 episodios deterministas)  
**Dataset Reservado:** `data/datasets/phase_c/final_c_500.json` (**NO abierto, NO evaluado, preservado intacto**)  
**Directorio de Artefactos del Diagnóstico:** `reports/phase_c_s1_analysis/`  

---

## 1. Resumen Ejecutivo

Este informe documenta el diagnóstico técnico integral del algoritmo Soft Actor-Critic (SAC) variante S1 en el entorno de manipulación planar 2 DOF. El análisis se sustentó exclusivamente en la evaluación determinista rigurosa de los artefactos históricos disponibles, sin alterar el código del entorno, sin modificar configuraciones históricas y sin realizar reentrenamientos.

### 1.1. Hallazgos Principales
1. **Rendimiento Máximo y Deterioro Síncrono:**
   - **[RESULTADO OBSERVADO]:** En la evaluación síncrona (congelando el modelo en el mismo paso de entrenamiento para todas las réplicas), SAC S1 alcanza su rendimiento pico medio en el **paso $327\,680$** con una tasa de éxito estricto de **$52{,}40\% \pm 6{,}01\%$** ($\text{ddof}=1$), situándose muy próximo al paso inmediatamente anterior ($286\,720$ pasos) con **$52{,}13\% \pm 8{,}08\%$**.
   - **[RESULTADO OBSERVADO]:** Hacia el cierre del entrenamiento ($655\,360$ pasos), la media síncrona desciende a **$46{,}60\% \pm 9{,}21\%$**, registrando un cambio descriptivo de $-5{,}80\text{ pp}$ frente al pico síncrono.
   - **[LIMITACIÓN DE INFERENCIA]:** Este deterioro síncrono de $-5{,}80\text{ pp}$ no se distingue de la fluctuación ordinaria observada entre checkpoints contiguos, dada la variabilidad entre las tres únicas réplicas disponibles ($\pm 6{,}01\%$ a $\pm 9{,}21\%$).
2. **Heterogeneidad entre Semillas y Selección Retrospectiva:**
   - **[RESULTADO OBSERVADO]:** La evolución temporal difiere marcadamente entre réplicas:
     - **Semilla 42:** Alcanza su máximo individual en el paso $286\,720$ ($60{,}80\%$) y decae a $40{,}60\%$ al final ($-20{,}20\text{ pp}$).
     - **Semilla 2026:** Alcanza su máximo individual en el paso $409\,600$ ($54{,}40\%$) y decae a $42{,}00\%$ al final ($-12{,}40\text{ pp}$).
     - **Semilla 100:** Registra su valor máximo individual en el paso final $655\,360$ ($57{,}20\%$), habiendo atravesado un mínimo intermedio de $41{,}60\%$ en el paso $368\,640$.
   - **[DISTINCIÓN METODOLÓGICA]:** La **media de los máximos individuales por semilla** es de **$57{,}47\%$** (reportada como $57{,}5\%$ en la Figura 4).
   - **[MECANISMO DE SELECCIÓN]:** Seleccionar retrospectivamente el mejor checkpoint de cada réplica genera una estimación optimista respecto al rendimiento que un protocolo de despliegue operativo puede garantizar. Al ocurrir los picos en pasos dispares ($286\,720$, $409\,600$ y $655\,360$) y seleccionarse *a posteriori* observando toda la trayectoria, una política de parada (*early stopping*) prospectiva requeriría un criterio predictivo validado para no sobreajustar la fluctuación muestral de cada corrida.
3. **Comportamiento Cinético y Dinámica de Optimización:**
   - **[RESULTADO OBSERVADO]:** El indicador de variación de acción inter-paso $\overline{\|\Delta \mathbf{a}\|}$ presenta un incremento sostenido desde el primer checkpoint ($0{,}0236 \pm 0{,}0028$ en el paso $40\,960$), alcanzando **$0{,}2041 \pm 0{,}0512$** en el paso final (con $0{,}2325$ en semilla 42 y $0{,}2348$ en semilla 2026).
   - **[RESULTADO OBSERVADO]:** Las curvas de TensorBoard evidencian un descenso continuo del coeficiente de entropía $\alpha$ desde $0{,}8357$ hasta una meseta asintótica en torno a $\sim 0{,}017$ (asociada a `target_entropy: -2.0`).
   - **[HIPÓTESIS CAUSAL]:** La co-ocurrencia temporal entre el incremento de $\|\Delta \mathbf{a}\|$, la disminución de $\alpha$ y la ausencia de penalización de aceleración ($w_a = 0$) sugiere como hipótesis plausible que la pérdida de regularización entrópica favorece la emergencia de conmutaciones de acción de alta frecuencia en torno al objetivo (*chattering*), dificultando el frenado suave.
   - **[LIMITACIÓN]:** Al no disponer de ablaciones controladas (fijando $\alpha$ o incorporando $w_a > 0$ bajo condiciones idénticas), no es posible atribuir causalidad estricta ni aislar el peso relativo del colapso de entropía frente a otros factores.
4. **Patrón de Fracaso Dominante:**
   - **[RESULTADO OBSERVADO]:** Al paso $655\,360$, la tasa de truncamiento es del **$53{,}40\% \pm 9{,}21\%$**, dividida en:
     - **$28{,}13\% \pm 5{,}94\%$ de entrada sin hold:** El efector llega al radio de $0{,}06\text{ m}$, pero no logra satisfacer la condición de velocidad articular $v_{\text{joint}} < 0{,}10\text{ rad/s}$ durante 20 pasos consecutivos.
     - **$25{,}27\% \pm 3{,}69\%$ de nunca entrar al radio:** Concentrado mayoritariamente en la periferia del espacio de trabajo ($r > 1{,}70\text{ m}$).
5. **Discrepancia Entrenamiento vs. Evaluación:**
   - **[RESULTADO OBSERVADO]:** Se constata que el criterio de entrenamiento ($N_{\text{hold}} = 5$, $v_{\text{tol}} = 0{,}30\text{ rad/s}$) fue sustancialmente más permisivo que la evaluación estricta ($N_{\text{hold}} = 20$, $v_{\text{tol}} = 0{,}10\text{ rad/s}$).
   - **[LIMITACIÓN DE ATRIBUCIÓN]:** Sin embargo, esta discrepancia no explica por sí sola la diferencia de rendimiento de SAC S1: el experimento de referencia PPO X3 utilizó exactamente el mismo criterio permisivo durante su entrenamiento y alcanzó un $88{,}80\% \pm 7{,}71\%$ de éxito bajo evaluación estricta en el paso homólogo $655\,360$ (alcanzando $\sim 97\%$ en su modelo final de 1 M de pasos).

---

## 2. Estado de la Evaluación y Trazabilidad

- **Inventario Evaluado:** Se comprobaron y evaluaron exitosamente los 16 checkpoints para las tres semillas:
  $$\text{Semillas } \{42, 100, 2026\} \times \text{Checkpoints } \{40\,960, 81\,920, \dots, 655\,360\} = 48\text{ evaluaciones}$$
- **Episodios:** Cada evaluación se ejecutó sobre los 500 episodios fijos de `data/datasets/phase_c/validation_c_500.json` en modo determinista ($\mathbf{a} = \tanh(\boldsymbol{\mu}(\mathbf{s}))$). Se simularon un total de $24\,000$ episodios sin errores de ejecución.
- **Normalización:** Cada evaluación cargó el objeto `VecNormalize` específico de su checkpoint (`vecnormalize_stats.pkl`), operando con `training = False` y `norm_reward = False`. No hubo desalineación de estadísticas de observación.
- **Aislamiento:** El dataset `data/datasets/phase_c/final_c_500.json` no fue consultado en ninguna etapa.

---

## 3. Resumen de la Auditoría de Implementación

*(Para el informe exhaustivo, consultar [auditoria_implementacion.md](auditoria_implementacion.md)).*

1. **Cinemática Instantánea:** El entorno [Arm2DEnv](arm2d_env.py) aplica velocidad angular directa:
   $$\dot{\theta}_i = a_i \cdot 1{,}5\text{ rad/s}$$
   No hay inercia ni aceleración física finita. Por tanto, conmutar $a_i$ de $-1$ a $+1$ produce un salto discontinuo de velocidad de $3{,}0\text{ rad/s}$ en $\Delta t = 0{,}05\text{ s}$.
2. **Requisitos de la Evaluación Estricta:**
   La cota de velocidad $v_{\text{joint}} = \sqrt{\dot{\theta}_1^2 + \dot{\theta}_2^2} < 0{,}10\text{ rad/s}$ impone:
   $$\|\mathbf{a}_t\|_2 < \frac{0{,}10}{1{,}5} \approx 0{,}0667$$
   Cualquier valor de acción superior al $6{,}7\%$ de la escala satura el umbral de velocidad y reinicia el contador de hold a cero.
3. **Función de Recompensa:** S1 utiliza `reward_type: smooth_precision`, donde:
   $$r_{\text{action}} = -0{,}02 \|\mathbf{a}\|^2, \quad w_v = 0{,}0, \quad w_a = 0{,}0$$
   No hay penalización sobre $\Delta \mathbf{a}$. Para el optimizador, conmutar entre valores positivos y negativos de acción tiene el mismo coste que mantener una acción suave, lo que desinhibe el chattering.

---

## 4. Rendimiento Temporal de Checkpoints

### 4.1. Tabla de Evolución Temporal (Media entre Semillas $\pm$ Desviación Estándar, $N=3$)

| Checkpoint | Pasos Entorno | Éxito Estricto (%) | Truncamiento (%) | Nunca Entra (%) | Entra sin Hold (%) | Variación $\|\Delta \mathbf{a}\|$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ckpt_40960` | $40\,960$ | $28{,}07 \pm 7{,}27$ | $71{,}93 \pm 7{,}27$ | $53{,}20 \pm 4{,}93$ | $18{,}73 \pm 2{,}93$ | $0{,}0236 \pm 0{,}0028$ |
| `ckpt_81920` | $81\,920$ | $40{,}20 \pm 3{,}70$ | $59{,}80 \pm 3{,}70$ | $39{,}67 \pm 4{,}65$ | $20{,}13 \pm 1{,}85$ | $0{,}0326 \pm 0{,}0033$ |
| `ckpt_122880` | $122\,880$ | $43{,}27 \pm 4{,}09$ | $56{,}73 \pm 4{,}09$ | $37{,}00 \pm 5{,}70$ | $19{,}73 \pm 1{,}86$ | $0{,}0376 \pm 0{,}0018$ |
| `ckpt_163840` | $163\,840$ | $47{,}80 \pm 6{,}00$ | $52{,}20 \pm 6{,}00$ | $34{,}67 \pm 3{,}31$ | $17{,}53 \pm 2{,}72$ | $0{,}0404 \pm 0{,}0006$ |
| `ckpt_204800` | $204\,800$ | $49{,}13 \pm 4{,}09$ | $50{,}87 \pm 4{,}09$ | $31{,}07 \pm 2{,}25$ | $19{,}80 \pm 1{,}91$ | $0{,}0422 \pm 0{,}0037$ |
| `ckpt_245760` | $245\,760$ | $49{,}40 \pm 9{,}17$ | $50{,}60 \pm 9{,}17$ | $30{,}13 \pm 6{,}85$ | $20{,}47 \pm 2{,}32$ | $0{,}0517 \pm 0{,}0036$ |
| `ckpt_286720` | $286\,720$ | $52{,}13 \pm 8{,}08$ | $47{,}87 \pm 8{,}08$ | $31{,}07 \pm 6{,}38$ | $16{,}80 \pm 1{,}71$ | $0{,}0592 \pm 0{,}0102$ |
| **`ckpt_327680`** | **$327\,680$** | **$52{,}40 \pm 6{,}01$** | **$47{,}60 \pm 6{,}01$** | **$28{,}47 \pm 3{,}70$** | **$19{,}13 \pm 2{,}50$** | **$0{,}0866 \pm 0{,}0120$** |
| `ckpt_368640` | $368\,640$ | $48{,}47 \pm 6{,}18$ | $51{,}53 \pm 6{,}18$ | $29{,}40 \pm 4{,}45$ | $22{,}13 \pm 1{,}79$ | $0{,}0829 \pm 0{,}0146$ |
| `ckpt_409600` | $409\,600$ | $49{,}47 \pm 4{,}27$ | $50{,}53 \pm 4{,}27$ | $27{,}13 \pm 0{,}31$ | $23{,}40 \pm 3{,}98$ | $0{,}1006 \pm 0{,}0104$ |
| `ckpt_450560` | $450\,560$ | $50{,}27 \pm 3{,}24$ | $49{,}73 \pm 3{,}24$ | $28{,}80 \pm 3{,}99$ | $20{,}93 \pm 1{,}15$ | $0{,}0991 \pm 0{,}0080$ |
| `ckpt_491520` | $491\,520$ | $48{,}40 \pm 2{,}36$ | $51{,}60 \pm 2{,}36$ | $28{,}60 \pm 2{,}77$ | $23{,}00 \pm 4{,}01$ | $0{,}1106 \pm 0{,}0231$ |
| `ckpt_532480` | $532\,480$ | $47{,}73 \pm 5{,}03$ | $52{,}27 \pm 5{,}03$ | $29{,}93 \pm 1{,}72$ | $22{,}33 \pm 6{,}54$ | $0{,}1090 \pm 0{,}0333$ |
| `ckpt_573440` | $573\,440$ | $48{,}47 \pm 3{,}53$ | $51{,}53 \pm 3{,}53$ | $29{,}07 \pm 0{,}70$ | $22{,}47 \pm 3{,}60$ | $0{,}1218 \pm 0{,}0279$ |
| `ckpt_614400` | $614\,400$ | $45{,}67 \pm 4{,}44$ | $54{,}33 \pm 4{,}44$ | $27{,}20 \pm 0{,}53$ | $27{,}13 \pm 4{,}69$ | $0{,}1620 \pm 0{,}0677$ |
| **`ckpt_655360`** | **$655\,360$** | **$46{,}60 \pm 9{,}21$** | **$53{,}40 \pm 9{,}21$** | **$25{,}27 \pm 3{,}69$** | **$28{,}13 \pm 5{,}94$** | **$0{,}2041 \pm 0{,}0512$** |

*Fuente de datos:* `reports/phase_c_s1_analysis/metricas_checkpoints.csv`. Estadísticos con $\text{ddof}=1$.

![Evolución Temporal del Éxito Estricto](figures/evolucion_exito.png)
![Evolución del Truncamiento](figures/evolucion_truncamiento.png)
![Comparación entre Semillas](figures/comparacion_semillas.png)
![Comparación Mejor Checkpoint vs Final](figures/comparacion_mejor_final.png)

### 4.2. Análisis del Deterioro
- **[RESULTADO OBSERVADO]:** En la evaluación síncrona, el valor medio máximo entre semillas se alcanza en el paso $327\,680$ con $52{,}40\% \pm 6{,}01\%$ (con el paso $286\,720$ en $52{,}13\% \pm 8{,}08\%$), para luego descender hacia $46{,}60\% \pm 9{,}21\%$ al paso final $655\,360$ (diferencia descriptiva de $-5{,}80\text{ pp}$).
- **[LIMITACIÓN DE INFERENCIA]:** El deterioro síncrono de $-5{,}80\text{ pp}$ no se distingue de la fluctuación ordinaria entre checkpoints contiguos observada a lo largo del entrenamiento, considerando que la desviación estándar entre semillas oscila entre $\pm 6{,}01\%$ y $\pm 9{,}21\%$.
- **[RESULTADO OBSERVADO]:** La tasa global de no entrada (`never_entered`) disminuye respecto al inicio ($53{,}20\% \pm 4{,}93\%$ en el paso $40\,960$ a $25{,}27\% \pm 3{,}69\%$ al final). Esta evolución no es monótona: muestra una tendencia decreciente con fluctuaciones intermedias. En particular, a partir del paso $327\,680$, la tasa oscila en la franja del $25\% \text{ -- } 30\%$:
  - Paso $327\,680$: $28{,}47\% \pm 3{,}70\%$
  - Paso $368\,640$: $29{,}40\% \pm 4{,}45\%$
  - Paso $409\,600$: $27{,}13\% \pm 0{,}31\%$
  - Paso $450\,560$: $28{,}80\% \pm 3{,}99\%$
  - Paso $491\,520$: $28{,}60\% \pm 2{,}77\%$
  - Paso $532\,480$: $29{,}93\% \pm 1{,}72\%$
  - Paso $573\,440$: $29{,}07\% \pm 0{,}70\%$
  - Paso $614\,400$: $27{,}20\% \pm 0{,}53\%$
  - Paso $655\,360$: $25{,}27\% \pm 3{,}69\%$
- **[RESULTADO OBSERVADO]:** La tasa de fracaso por `entered_no_hold` sube desde $16{,}80\% \text{ -- } 19{,}13\%$ en la ventana de mayor éxito hasta **$28{,}13\% \pm 5{,}94\%$** al final.
- **[RESULTADO OBSERVADO]:** La variación de acción $\|\Delta \mathbf{a}\|$ experimenta un incremento sostenido desde el primer checkpoint ($0{,}0236$ en paso $40\,960$ a $0{,}2041$ en paso $655\,360$).
- **[HIPÓTESIS CAUSAL]:** Los datos muestran co-ocurrencia entre el incremento de $\|\Delta \mathbf{a}\|$ y los fallos por `entered_no_hold`, pero no permiten afirmar causalidad estricta sin experimentos de ablación.

---

## 5. Rendimiento Espacial

Se particionó el dataset de 500 objetivos en tres zonas radiales exhaustivas:
- **Cercana ($r < 0{,}30\text{ m}$):** 45 episodios ($9{,}0\%$).
- **Intermedia ($0{,}30 \le r \le 1{,}70\text{ m}$):** 401 episodios ($80{,}2\%$).
- **Periférica ($r > 1{,}70\text{ m}$):** 54 episodios ($10{,}8\%$).

### 5.1. Comparativa Espacial en el Checkpoint Final ($655\,360$ pasos)

| Región | N | Éxito Estricto (%) | Nunca Entra (%) | Entra sin Hold (%) | Distancia Mínima (m) | Variación $\|\Delta \mathbf{a}\|$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cercana ($r < 0{,}30$)** | 45 | $49{,}63 \pm 12{,}99$ | $24{,}44 \pm 7{,}94$ | $25{,}93 \pm 7{,}94$ | $0{,}0506 \pm 0{,}0116$ | $0{,}1724 \pm 0{,}0520$ |
| **Intermedia ($0{,}30 \le r \le 1{,}70$)** | 401 | $48{,}13 \pm 9{,}29$ | $22{,}44 \pm 4{,}83$ | $29{,}43 \pm 5{,}96$ | $0{,}0429 \pm 0{,}0039$ | $0{,}2135 \pm 0{,}0529$ |
| **Periférica ($r > 1{,}70$)** | 54 | $32{,}72 \pm 6{,}98$ | $46{,}91 \pm 3{,}66$ | $20{,}37 \pm 6{,}98$ | $0{,}0732 \pm 0{,}0068$ | $0{,}1604 \pm 0{,}0374$ |

*Fuente de datos:* `reports/phase_c_s1_analysis/analisis_espacial.csv`.

![Rendimiento por Región Espacial](figures/rendimiento_espacial.png)
![Mapa Espacial de Éxitos y Fallos](figures/mapa_exitos_fallos.png)

### 5.2. Interpretación Espacial
1. **Zona Intermedia ($0{,}30 \le r \le 1{,}70\text{ m}$):** Es la zona cinemáticamente más favorable. La distancia mínima promedio alcanzada es de apenas $0{,}0429\text{ m}$ ($< 0{,}06\text{ m}$). Sin embargo, la tasa de `entered_no_hold` alcanza el **$29{,}43\%$**, acompañada del valor máximo de variación de acción ($\|\Delta \mathbf{a}\| = 0{,}2135$). Esto corrobora que en esta región el efector alcanza el objetivo fácilmente, pero es expulsado o no se estabiliza debido a la oscilación angular.
2. **Zona Periférica ($r > 1{,}70\text{ m}$):** Casi la mitad de los objetivos ($46{,}91\%$) nunca son alcanzados (`never_entered`). Esto se debe a la cercanía con la singularidad cinemática de codo extendido ($\theta_2 \approx 0$), donde la manipulabilidad decrece fuertemente y el límite de 200 pasos resulta insuficiente para trayectorias subóptimas.
3. **Zona Cercana ($r < 0{,}30\text{ m}$):** Presenta una tasa de éxito de $49{,}63\%$, penalizada por la proximidad a la base, donde la singularidad de codo plegado requiere maniobras de inversión articular no dominadas.

---

## 6. Análisis de Trayectorias Representativas

Se examinaron episodios representativos deterministas clasificados según el protocolo estricto.

*Fuente de datos:* `reports/phase_c_s1_analysis/metricas_trayectorias.csv`.

![Trayectorias Cartesianas Representativas](figures/trayectorias_representativas.png)
![Distancia Temporal al Objetivo](figures/distancia_temporal.png)
![Velocidad Articular Temporal](figures/velocidad_articular.png)
![Acciones Temporales y Chattering](figures/acciones_temporales.png)

### 6.1. Casos de Estudio Observados

1. **Éxito Limpio (Ej. Semilla 42, Checkpoint $286\,720$, Episodio 0):**
   - El brazo aproxima suavemente hacia la meta ($r = 0{,}77\text{ m}$), reduce su velocidad antes de entrar al radio ($d < 0{,}06\text{ m}$) a $0{,}08\text{ rad/s}$ y mantiene la posición durante 20 pasos con acciones suaves ($\overline{\|\Delta \mathbf{a}\|} = 0{,}0685$).
2. **Entrada sin Hold con Múltiples Salidas (Ej. Semilla 42, Checkpoint $655\,360$, Episodio 1):**
   - El brazo entra en el radio ($d_{\min} = 0{,}0577\text{ m}$), pero la velocidad articular instantánea nunca desciende de $0{,}25\text{ rad/s}$. La política conmuta violentamente sus acciones ($\overline{\|\Delta \mathbf{a}\|} = 0{,}4438$), generando **22 entradas y salidas del radio objetivo** hasta truncar en el paso 200 sin completar un solo ciclo de hold.
3. **Comportamiento Perimetral Extremo (Ej. Semilla 100, Checkpoint $655\,360$, Episodio 1):**
   - El efector entra al radio objetivo y la política genera una oscilación sostenida de alta frecuencia en la frontera ($d \approx 0{,}058\text{ m}$), produciendo **74 salidas del radio** con una variación de acción $\overline{\|\Delta \mathbf{a}\|} = 0{,}9992$.
4. **Oscilación Violenta y Saturación (Ej. Semilla 2026, Checkpoint $655\,360$, Episodio 21):**
   - El brazo orbita la meta con $\overline{\|\Delta \mathbf{a}\|} = 1{,}4674$ (conmutación casi pura entre saturaciones $\pm 1$) y **78 salidas del radio**, mostrando un ciclo límite caótico que destruye cualquier posibilidad de reposo.
5. **No Entrada Periférica (Ej. Semilla 42, Checkpoint $655\,360$, Episodio 0):**
   - El objetivo se localiza en la periferia ($r = 1{,}72\text{ m}$). El brazo extiende ambos eslabones pero desacelera antes de alcanzar la frontera ($d_{\min} = 0{,}0830\text{ m}$), consumiendo los 200 pasos sin tocar el radio de $0{,}06\text{ m}$.

---

## 7. Matriz de Diagnóstico de Hipótesis

A continuación se resume el contraste sistemático de las hipótesis H1 a H8 planteadas para explicar el desempeño y deterioro de SAC S1.

| Hipótesis | Evidencia a Favor | Evidencia en Contra | Métricas / Episodios de Respaldo | Confianza | Limitaciones | Comprobación Adicional Necesaria |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **H1. Sobrepaso del objetivo (Overshoot)** | Velocidad al entrar al radio frecuentemente supera $0{,}30\text{ rad/s}$ ($v_{\text{entry}} > 0{,}40$ en $12\text{--}18\%$ de episodios). | Tras el sobrepaso, el brazo no continúa alejándose de forma balística sino que retorna hacia el objetivo. | Tasa balística promedio: $15{,}2\%$. Figuras 8 y 9. | **Media** | No es un sobrepaso inercial clásico (el modelo no tiene masa/inercia), sino cinemático. | Evaluar penalización de aproximación Gaussiana ($w_v$). |
| **H2. Oscilación alrededor del objetivo** | Episodios con múltiples cruces de signo y conmutación de acciones de alta frecuencia en $d < 0{,}30\text{ m}$. | En objetivos lejanos la política no oscila durante la fase de viaje libre. | Episodio 1 (semilla 100, 74 salidas; semilla 42, 22 salidas). Figura 10. | **Alta** | La oscilación se desencadena específicamente en las proximidades del objetivo. | Evaluar penalización de suavidad $w_a \|\Delta \mathbf{a}\|^2$. |
| **H3. Entradas y salidas repetidas de la zona objetivo** | La tasa de `entered_no_hold` alcanza el $28{,}13\%$ en el paso final, duplicándose respecto al pico. | Ninguna; es el fallo más frecuente en la región intermedia ($29{,}43\%$). | Tabla de evolución temporal y `metricas_checkpoints.csv`. | **Alta** | Es un síntoma derivado de la conmutación de acciones y la intolerancia de velocidad. | Contrastar con criterios de hold más permisivos. |
| **H4. Aproximación insuficiente (No entrada)** | En la zona periférica ($r > 1{,}70\text{ m}$), el $46{,}91\%$ de los objetivos no son alcanzados. | En la zona intermedia, el $77{,}56\%$ de los objetivos sí entra al radio. | Tabla 5.1 y mapa espacial Figura 6. | **Alta** (para periferia) / **Baja** (para zona intermedia) | Limitación combinada de 200 pasos y singularidad cinemática de extensión completa. | Incrementar horizonte de evaluación a 300 pasos en periferia. |
| **H5. Comportamiento dependiente del estado** | $\|\Delta \mathbf{a}\|$ es bajo en espacio libre y se dispara drásticamente al ingresar a la zona cercana ($d < 0{,}30\text{ m}$). | La arquitectura de red es totalmente determinista y continua. | Trazas temporales (Figura 10). | **Alta** | Derivado del gradiente abrupto del término $r_{\text{near}}$ en la recompensa. | Analizar el mapa de gradientes del actor en torno a $d=0{,}30$. |
| **H6. Diferencia entre entrenamiento y evaluación** | El entrenamiento concedió éxito con $N=5$ y $v_{\text{tol}}=0{,}30$. La evaluación estricta exige $N=20$ y $v_{\text{tol}}=0{,}10$. | Ninguna en cuanto a la existencia de la discrepancia; está verificada en código. | [S1.yaml:L14-L15](configs/phase_c/S1.yaml#L14-L15) vs [thresholds.yaml:L17-L18](configs/phase_a/thresholds.yaml#L17-L18). | **Alta** | La discrepancia explica la brecha de éxito, pero no el aumento progresivo del chattering temporal. | Evaluar los checkpoints con el criterio de entrenamiento para comparar curvas. |
| **H7. Efectos de la normalización** | Ninguno. Se comprobó carga correcta de `VecNormalize` con estadísticas congeladas en cada checkpoint. | La evaluación vectorial reproduce idénticamente los resultados de SB3 ($< 10^{-7}$). | Código de auditoría y tests de regresión vectoriales. | **Rechazada** (como causa del fallo) | La normalización funciona de forma correcta y rigurosa. | Ninguna requerida. |
| **H8. Deterioro asociado al aprendizaje de SAC** | Colapso del coeficiente de entropía $\alpha \to 0{,}017$, picos de pérdida en los críticos y saturación de pérdida del actor. | La semilla 100 mantuvo estabilidad en critic loss y no cayó al paso 655k. | Logs de TensorBoard (Figura 11). `train/ent_coef`, `train/critic_loss`. | **Alta** | Correlación temporal directa entre el colapso de $\alpha$ y la triplicación de $\|\Delta \mathbf{a}\|$. | Evaluar fijación de $\alpha = 0{,}005$ o `target_entropy: -1`. |

![Dinámica de Entrenamiento en TensorBoard](figures/indicadores_entrenamiento.png)

---

## 8. Diagnóstico Final: Respuestas a las Preguntas Clave

### 1. ¿En qué momento del entrenamiento alcanza S1 su mejor rendimiento verificable?
- **[RESULTADO OBSERVADO - SÍNCRONO]:** Bajo un criterio síncrono (mismo paso de congelación para todas las réplicas), el valor medio máximo entre semillas se ubica en el **paso $327\,680$ con $52{,}40\% \pm 6{,}01\%$** ($\text{ddof}=1$), con un desempeño prácticamente equivalente en el paso previo ($286\,720$ pasos con $52{,}13\% \pm 8{,}08\%$).
- **[RESULTADO OBSERVADO - MÁXIMOS INDIVIDUALES]:** Si se examina cada réplica de forma desacoplada, los picos individuales son:
  - Semilla 42: **$60{,}80\%$** en el paso $286\,720$.
  - Semilla 2026: **$54{,}40\%$** en el paso $409\,600$.
  - Semilla 100: **$57{,}20\%$** en el paso $655\,360$.
  La **media de estos máximos individuales** es de **$57{,}47\%$** (reportada como $57{,}5\%$ en la Figura 4).
- **[MECANISMO DE SELECCIÓN]:** La media de $57{,}47\%$ representa una estimación optimista del rendimiento seleccionable. Dado que los picos se alcanzan en pasos temporales distintos y se determinan retrospectivamente conociendo la totalidad de los datos de validación, una política de parada (*early stopping*) operativa requeriría una regla predictiva previa para no sobreajustar la variabilidad estocástica de cada ejecución.

### 2. ¿El deterioro final aparece en las tres semillas o solo en algunas?
- **[RESULTADO OBSERVADO]:** La reducción numérica estricta entre el pico histórico de cada réplica y el checkpoint de cierre ($655\,360$ pasos) ocurre en **dos de las tres semillas**:
  - Semilla 42: caída de **$-20{,}20\text{ pp}$** ($60{,}80\% \to 40{,}60\%$).
  - Semilla 2026: caída de **$-12{,}40\text{ pp}$** ($54{,}40\% \to 42{,}00\%$).
  - Semilla 100: no exhibe caída terminal respecto a su pico, ya que su valor máximo numérico ocurre en el último paso ($57{,}20\%$), si bien experimentó fuertes fluctuaciones intermedias (alcanzó un mínimo de $41{,}60\%$ en el paso $368\,640$).
- **[RESULTADO OBSERVADO]:** En las tres semillas, el incremento de la variación de acción $\|\Delta \mathbf{a}\|$ es sostenido desde el primer checkpoint ($40\,960$ pasos) hasta el final.

### 3. ¿Qué tipos de fallos explican principalmente la tasa de fracaso?
- **[RESULTADO OBSERVADO]:** En el checkpoint final ($655\,360$ pasos), el $53{,}40\% \pm 9{,}21\%$ de episodios truncados se divide en:
  1. **$28{,}13\% \pm 5{,}94\%$ Entra sin hold (`entered_no_hold`):** El brazo penetra el radio de $0{,}06\text{ m}$, pero no sostiene 20 pasos consecutivos a menos de $0{,}10\text{ rad/s}$.
  2. **$25{,}27\% \pm 3{,}69\%$ Nunca entra (`never_entered`):** El brazo no logra penetrar el radio de $0{,}06\text{ m}$ durante los 200 pasos.

### 4. ¿La política falla principalmente al aproximarse, al frenar o al mantener la posición?
- **[RESULTADO OBSERVADO]:** El patrón de fallo depende fuertemente de la región radial:
  - En la **zona intermedia ($0{,}30 \le r \le 1{,}70\text{ m}$)**, el fallo predominante es de frenado y retención ($29{,}43\%$ por `entered_no_hold` frente a $22{,}44\%$ por `never_entered`).
  - En la **zona periférica ($r > 1{,}70\text{ m}$)**, el fallo predominante es de aproximación ($46{,}91\%$ por `never_entered`).

### 5. ¿Hay evidencia de oscilaciones o sobrepasos?
- **[RESULTADO OBSERVADO]:** Se observan episodios deterministas con conmutaciones de acción de alta frecuencia y múltiples cruces de frontera en las proximidades del objetivo ($d < 0{,}30\text{ m}$).
- **[HIPÓTESIS CAUSAL]:** Este patrón es compatible con oscilaciones inducidas por la pérdida de dispersión en la política o gradientes pronunciados en la recompensa terminal.
- **[LIMITACIÓN]:** Dado que el entorno aplica cinemática instantánea sin inercia, estas oscilaciones corresponden a conmutaciones de control cinemático y no a inestabilidad inercial clásica.

### 6. ¿Los fallos se concentran en regiones espaciales concretas?
- **[RESULTADO OBSERVADO]:** Sí. Los fallos por `never_entered` se concentran en la periferia ($46{,}91\%$), mientras que los fallos por `entered_no_hold` dominan en el espacio de trabajo intermedio ($29{,}43\%$).

### 7. ¿Existe una discrepancia verificable entre los criterios de entrenamiento y evaluación?
- **[RESULTADO OBSERVADO]:** Sí, verificada en el código:
  - Entrenamiento: $N_{\text{hold}} = 5\text{ pasos}$, $v_{\text{tol}} = 0{,}30\text{ rad/s}$.
  - Evaluación estricta: $N_{\text{hold}} = 20\text{ pasos}$, $v_{\text{tol}} = 0{,}10\text{ rad/s}$.
- **[LIMITACIÓN DE ATRIBUCIÓN]:** Esta discrepancia no explica por sí sola la diferencia de desempeño: el experimento de referencia PPO X3 utilizó exactamente el mismo criterio permisivo de hold durante el entrenamiento y alcanzó un $88{,}80\% \pm 7{,}71\%$ de éxito bajo la evaluación estricta en el paso homólogo $655\,360$ (y $\sim 97\%$ en su modelo final de 1 M de pasos).

### 8. ¿La normalización se está aplicando correctamente?
- **[RESULTADO OBSERVADO]:** Sí. `VecNormalize` opera congelada en evaluación con las estadísticas propias de cada checkpoint, reproduciendo fielmente las salidas de Stable-Baselines3 ($< 10^{-7}$).

### 9. ¿Qué información falta para determinar la causa del deterioro?
- **[LIMITACIÓN METODOLÓGICA]:**
  - Muestra reducida ($N=3$ semillas), lo que otorga carácter estrictamente descriptivo a las medias y limita la generalización estadística.
  - Falta de registros de normas de gradientes de actor y crítico en los logs históricos.
  - Ausencia de experimentos de ablación controlada (con condiciones idénticas salvo fijación de $\alpha$ o adición de $w_a > 0$) para aislar de forma causal los factores del deterioro.

### 10. ¿Qué propuestas no evaluadas merecen exploración futura?
- **[PROPUESTAS NO EVALUADAS (FUERA DE CONCLUSIONES)]:**
  1. Regularización entrópica: evaluar fijación de $\alpha$ constante o elevar `target_entropy: -1.0`.
  2. Regularización cinemática: contrastar la incorporación de un término de penalización de aceleración $w_a \|\Delta \mathbf{a}\|^2 > 0$ en la recompensa.
  3. Alineación estricta de hold: entrenar con $N_{\text{hold}} = 20$ y $v_{\text{tol}} = 0{,}10\text{ rad/s}$.

---

## 9. Próximos Experimentos Recomendados (Fase Futura — Propuestas No Evaluadas)

*(Nota: Estas variantes son formulaciones analíticas no entrenadas, diseñadas para contrastación empírica futura sin modificar los artefactos históricos).*

### Propuesta 1 — S2: Reducción de Frecuencia de Actualización del Crítico / Actor
- **Hipótesis a Contrastar:** La alta frecuencia de actualización de SAC (`train_freq: 1`, `gradient_steps: 1`) sobreajusta la red Q a transiciones recientes con colapso de entropía, desestabilizando al actor.
- **Evidencia que la Justifica:** Picos periódicos en `critic_loss` (hasta 22.9 en TensorBoard).
- **Parámetros a Modificar:** `train_freq: 2`, `gradient_steps: 1`.

### Propuesta 2 — S3: Coeficiente de Entropía Fijo ($\alpha = 0{,}010$ a $0{,}020$)
- **Hipótesis a Contrastar:** Un coeficiente de entropía fijo previene la concentración cuasi-determinista en las fronteras de acción $[-1, 1]$.
- **Evidencia que la Justifica:** Correlación temporal entre la caída de $\alpha$ y el aumento sostenido de $\|\Delta \mathbf{a}\|$.
- **Parámetros a Modificar:** `ent_coef: 0.010` (o `0.020`).

### Propuesta 3 — S4: Ajuste Automático con Entropía Objetivo Conservadora (`target_entropy: -1.0`)
- **Hipótesis a Contrastar:** Elevar $\mathcal{H}_{\text{target}}$ amortigua la conmutación bang-bang cerca de la meta.
- **Evidencia que la Justifica:** Varianza de la distribución gaussiana colapsada al final del entrenamiento.
- **Parámetros a Modificar:** `target_entropy: -1.0`.

---

## 10. Conclusión General

El diagnóstico establece de manera verificable que el desempeño de SAC S1 ($46{,}60\% \pm 9{,}21\%$ en el paso final $655\,360$ y pico medio síncrono de $52{,}40\% \pm 6{,}01\%$ en el paso $327\,680$) **no responde a fallos en la implementación de la normalización ni a anomalías en el entorno de simulación**.

Desde el punto de vista descriptivo:
1. El cambio síncrono de $-5{,}80\text{ pp}$ entre el pico síncrono ($327\,680$) y el cierre ($655\,360$) no se distingue de la fluctuación observada entre checkpoints contiguos, dada la dispersión entre las tres semillas evaluadas.
2. La política adquiere destreza cinemática de aproximación hacia el objetivo (la tasa global de `never_entered` disminuye de $53{,}20\%$ a $25{,}27\%$), pero fracasa en la estabilización terminal (la tasa de `entered_no_hold` asciende a $28{,}13\%$).
3. La variación de acción inter-paso $\|\Delta \mathbf{a}\|$ experimenta un aumento sostenido desde el primer checkpoint ($0{,}0236 \to 0{,}2041$).

La atribución causal definitiva de estos patrones al colapso de entropía, al chattering o al sobreentrenamiento requiere de experimentos de ablación controlada (detallados como propuestas no evaluadas en la Sección 9).
