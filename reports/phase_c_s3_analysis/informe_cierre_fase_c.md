# Informe Técnico de Cierre: Fase C
**Evaluación de Soft Actor-Critic (SAC) en Manipulación Planar 2 DOF: Variantes Unifactoriales, Paquete Combinado y Comparativa frente a PPO a 245.760 Pasos**

---

## 1. Contexto Experimental y Objetivo de la Fase C

El presente informe documenta la investigación experimental realizada sobre el algoritmo Soft Actor-Critic (SAC) en el entorno de simulación cinemática planar de 2 grados de libertad (`Arm2DEnv`). Tras la evaluación inicial del modelo base `S1` documentada en el diagnóstico técnico previo (`reports/phase_c_s1_analysis/diagnostico_s1.md`), se diseñó un protocolo empírico orientado a determinar si ajustes en la dinámica de actualización, en la función de recompensa o en la regularización entrópica permitían a SAC alcanzar o superar el desempeño de Proximal Policy Optimization (`PPO X3`) dentro de un límite presupuestario de $245\,760$ pasos de interacción.

La investigación se articuló en torno a una regla de decisión preregistrada (*Gatekeeper*): dado que los algoritmos fuera de política (*off-policy*) introducen una mayor complejidad estructural y un tiempo de cómputo por paso habitualmente superior al de los métodos sobre la política (*on-policy*), su adopción en este proyecto requería constatar una ventaja neta en eficiencia de muestras frente a `PPO X3`. En la Fase B, `PPO X3` alcanzó una tasa de éxito estricto de $93,53\,\% \pm 2,50\,\%$ en el paso $245\,760$ sobre el conjunto determinista de 500 episodios de `data/datasets/phase_c/validation_c_500.json`. De no observarse un rendimiento competitivo respecto a dicha referencia bajo las configuraciones probadas, el protocolo contemplaba la interrupción de la exploración de SAC bajo el criterio de costo-beneficio del proyecto.

---

## 2. Metodología y Variantes Evaluadas

La exploración comprendió dos fases: una ablación unifactorial estricta sobre la arquitectura base `S1` y un piloto exploratorio combinado (`S3_combined`), evaluados en su totalidad sobre tres semillas pseudoaleatorias (`42`, `100`, `2026`):

1. **Línea Base SAC (`S1`, `configs/phase_c/S1.yaml`):** Redes de política y crítico con dos capas ocultas de 256 neuronas y activación ReLU, normalización de observaciones mediante `VecNormalize` (`clip_obs = 10.0`, `norm_reward = False`), factor de descuento $\gamma = 0{,}99$, tasa de actualización suave de objetivos $\tau = 0{,}005$, optimizador Adam con tasa de aprendizaje de $3 \times 10^{-4}$, tamaño de buffer de repetición de $300\,000$ transiciones, tamaño de lote de 256, frecuencia de actualización `train_freq: 1` con `gradient_steps: 1` (ratio UTD = 1,0), coeficiente de temperatura de entropía adaptativo $\alpha$ con objetivo por defecto $\mathcal{H}_{\text{tgt}} = -\dim(\mathcal{A}) = -2{,}0$, y función de recompensa `smooth_precision` ($w_a = 0{,}0$, $w_v = 0{,}0$).
2. **Variante Unifactorial de Frecuencia (`S2_freq`, `configs/phase_c/S2_freq.yaml`):** Idéntica a `S1` en todos sus hiperparámetros, modificando exclusivamente `sac.train_freq: 4` con `gradient_steps: 1`, lo que reduce el ratio de actualizaciones por paso de entorno a $\text{UTD} = 0{,}25$.
3. **Variante Unifactorial de Suavidad de Acción (`S2_wa`, `configs/phase_c/S2_wa.yaml`):** Idéntica a `S1` en todos sus hiperparámetros, modificando exclusivamente la inclusión del término de penalización cuadrática por variación de acción $r_{\text{smooth}} = -w_a \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$ con `env.reward_params.w_a: 0.10`, valor calibrado a partir de las escalas cinemáticas de la Etapa C-1.
4. **Piloto Combinado (`S3_combined`, `configs/phase_c/S3_combined.yaml`):** Paquete exploratorio no atribuible que combinó simultáneamente tres intervenciones respecto de `S1`: regulación de actualizaciones (`train_freq: 4`), penalización moderada de acción (`w_a: 0.05`) y relajación del objetivo de entropía hacia un régimen de mayor exploración sostenida (`target_entropy: -1.0`).

En todas las variantes de SAC, el protocolo de hold de entrenamiento se mantuvo deliberadamente en su régimen relajado canónico (`hold_steps: 5`, `velocity_tolerance: 0.30`, $w_v = 0{,}0$). Cada variante se ejecutó hasta alcanzar $245\,760$ pasos por semilla, registrándose puntos de control atómicos cada $40\,960$ pasos (`model.zip` junto a `vecnormalize_stats.pkl`).

La evaluación se ejecutó de forma estrictamente determinista ($\mathbf{a} = \tanh(\boldsymbol{\mu}(\mathbf{s}))$ para SAC; $\mathbf{a} = \boldsymbol{\mu}(\mathbf{s})$ para PPO) sobre los 500 episodios de `data/datasets/phase_c/validation_c_500.json` con el envoltorio `VecNormalize` congelado (`training = False`, `norm_reward = False`), aplicando el criterio de éxito estricto de parada: permanencia dentro del radio de meta de $0{,}06\text{ m}$ durante al menos 20 pasos de control consecutivos con velocidad articular menor a $0{,}10\text{ rad/s}$.

---

## 3. Resultados Experimentales en el Punto de Corte (Paso 245.760)

Los datos medidos de forma empírica sobre los puntos de control en el paso $245\,760$ para las tres semillas se presentan en la siguiente tabla consolidada **[HECHO MEDIDO]**:

\begin{table}[htbp]
\centering
\caption{Comparación consolidada de desempeño y estabilidad en el paso 245\,760 sobre \texttt{validation\_c\_500.json} (3 semillas: 42, 100 y 2026; media $\pm$ desviación estándar con $\text{ddof}=1$, 500 episodios deterministas por punto de control).}
\label{tab:comparativa_3semillas_245k_cierre}
\small
\begin{tabular}{lcccccc}
\hline
\textbf{Modelo} & \textbf{Éxito estricto} & \textbf{Never entered} & \textbf{Entered no hold} & \textbf{$\|\Delta \mathbf{a}\|$} & \textbf{Salidas meta} & \textbf{Tiempo ent.} \\
 & (\%) & (\%) & (\%) & (rad/s) & ($n_{\text{exits}}$) & (min) \\
\hline
PPO X3 & $93,53 \pm 2,50$ & $4,47 \pm 1,96$ & $2,00 \pm 0,60$ & $0,0402 \pm 0,0021$ & $0,020 \pm 0,004$ & $7,62$ \\
SAC S1 & $49,40 \pm 9,17$ & $30,13 \pm 6,85$ & $20,47 \pm 2,32$ & $0,0517 \pm 0,0036$ & $0,633 \pm 0,319$ & $94,86$ \\
SAC S2\_freq & $58,93 \pm 3,80$ & $23,40 \pm 4,51$ & $17,67 \pm 2,31$ & $0,0490 \pm 0,0138$ & $0,229 \pm 0,108$ & $28,24$ \\
SAC S2\_wa & $44,53 \pm 2,44$ & $34,07 \pm 0,61$ & $21,40 \pm 2,80$ & $0,0378 \pm 0,0054$ & $0,221 \pm 0,032$ & $99,23$ \\
\hline
\textbf{SAC S3\_combined}\textsuperscript{*} & $\mathbf{84,53 \pm 4,70}$ & $\mathbf{9,33 \pm 3,69}$ & $\mathbf{6,13 \pm 1,33}$ & $\mathbf{0,0457 \pm 0,0015}$ & $\mathbf{0,067 \pm 0,017}$ & $\mathbf{30,40}$ \\
\hline
\end{tabular}
\begin{tablenotes}
\small
\item \textsuperscript{*} Paquete combinado, no atribuible: modifica simultáneamente $\texttt{train\_freq} = 4$, $w_a = 0{,}05$ y $\texttt{target\_entropy} = -1{,}0$ respecto de S1.
\end{tablenotes}
\end{table}

### Desglose Individual por Semilla al Paso 245.760 **[HECHO MEDIDO]**

El desglose empírico de las tres semillas confirma la dispersión muestral de cada configuración:
* **`PPO X3`:** $92{,}40\,\%$ (semilla 42), $96{,}40\,\%$ (semilla 100), $91{,}80\,\%$ (semilla 2026).
* **`SAC S3_combined`:** $79{,}80\,\%$ (semilla 42), $89{,}20\,\%$ (semilla 100), $84{,}60\,\%$ (semilla 2026).
* **`SAC S2_freq`:** $62{,}80\,\%$ (semilla 42), $55{,}20\,\%$ (semilla 100), $58{,}80\,\%$ (semilla 2026).
* **`SAC S1`:** $59{,}40\,\%$ (semilla 42), $47{,}40\,\%$ (semilla 100), $41{,}40\,\%$ (semilla 2026).
* **`SAC S2_wa`:** $42{,}40\,\%$ (semilla 42), $47{,}20\,\%$ (semilla 100), $44{,}00\,\%$ (semilla 2026).

---

## 4. Evolución Temporal Comparativa a lo Largo del Presupuesto

La dinámica de aprendizaje a lo largo de los 6 puntos de control periódicos muestra perfiles diferenciados entre algoritmos y variantes **[HECHO MEDIDO]**:

\begin{table}[htbp]
\centering
\caption{Evolución de la tasa de éxito estricto (\%) sobre \texttt{validation\_c\_500.json} en la grilla de puntos de control (media $\pm$ std con $\text{ddof}=1$, 3 semillas).}
\label{tab:grilla_checkpoints_completa}
\small
\begin{tabular}{cccccc}
\hline
\textbf{Paso} & \textbf{PPO X3} & \textbf{SAC S1} & \textbf{SAC S2\_freq} & \textbf{SAC S2\_wa} & \textbf{SAC S3\_combined} \\
\hline
$40\,960$  & $0,27 \pm 0,23$  & $28,07 \pm 7,27$ & $30,47 \pm 1,53$ & $27,73 \pm 7,11$ & $31,67 \pm 2,84$ \\
$81\,920$  & $10,53 \pm 5,13$ & $40,20 \pm 3,70$ & $37,80 \pm 6,58$ & $28,00 \pm 3,41$ & $47,13 \pm 6,52$ \\
$122\,880$ & $37,20 \pm 7,29$ & $43,27 \pm 4,09$ & $56,00 \pm 8,81$ & $33,93 \pm 5,11$ & $71,07 \pm 3,06$ \\
$163\,840$ & $65,40 \pm 6,90$ & $47,80 \pm 6,00$ & $61,07 \pm 1,10$ & $40,07 \pm 0,76$ & $78,27 \pm 3,06$ \\
$204\,800$ & $88,40 \pm 2,03$ & $49,13 \pm 4,09$ & $56,13 \pm 0,81$ & $42,93 \pm 1,29$ & $83,87 \pm 3,82$ \\
$245\,760$ & $93,53 \pm 2,50$ & $49,40 \pm 9,17$ & $58,93 \pm 3,80$ & $44,53 \pm 2,44$ & $84,53 \pm 4,70$ \\
\hline
\end{tabular}
\end{table}

### Observaciones sobre la Grilla Temporal:
* **Fase inicial (pasos $40\,960$ a $81\,920$):** Los agentes SAC presentan un arranque más temprano que PPO. A $40\,960$ pasos, todas las variantes de SAC superan el $27\,\%$, mientras que `PPO X3` registra apenas un $0,27\,\%$. Hacia el paso $81\,920$, `S3_combined` alcanza el $47,13\,\%$ frente al $10,53\,\%$ de PPO.
* **Punto de cruce a $163\,840$ pasos:** En este paso, `SAC S2_freq` alcanza su valor máximo de $61,07\,\% \pm 1,10\,\%$, momento en el cual `PPO X3` registra $65,40\,\% \pm 6,90\,\%$ y `S3_combined` asciende a $78,27\,\% \pm 3,06\,\%$.
* **Fase de convergencia (pasos $204\,800$ a $245\,760$):** `PPO X3` experimenta una aceleración pronunciada, alcanzando $88,40\,\%$ a $204\,800$ pasos y $93,53\,\%$ al cierre. Las variantes unifactoriales de SAC (`S1`, `S2_freq`, `S2_wa`) muestran una saturación temprana en torno al $45\,\%$–$59\,\%$. Por su parte, `S3_combined` continúa progresando de manera estable hasta el $84,53\,\% \pm 4,70\,\%$.

---

## 5. Análisis de las Variantes de SAC

### 5.1. Frecuencia de Actualización (`S2_freq`) **[HECHO MEDIDO]**
La reducción de la frecuencia de entrenamiento a una actualización cada 4 pasos de entorno incrementó la tasa de éxito de $49,40\,\%$ a $58,93\,\%$ ($+9,53\text{ pp}$) en el paso $245\,760$, alcanzando un pico de $61,07\,\%$ a $163\,840$ pasos. La proporción de episodios sin entrada al objetivo (`never_entered`) descendió de $30,13\,\%$ a $23,40\,\%$, y las salidas del radio de meta se redujeron de $0,633$ a $0,229$ eventos por episodio. En términos de cómputo, la reducción en retropropagaciones elevó la tasa de simulación de $129,7\text{ FPS}$ a $435,1\text{ FPS}$, completando las tres semillas en $28,24$ minutos frente a los $94,86$ minutos de `S1`.

### 5.2. Penalización de Variación de Acción (`S2_wa`, $w_a = 0{,}10$) **[HECHO MEDIDO]**
La penalización directa sobre $\|\Delta \mathbf{a}\|^2$ redujo la variación media de acción de $0,0517$ a $0,0378\text{ rad/s}$ ($-26,9\,\%$) y disminuyó la dispersión inter-semilla a $\pm 2,44\,\%$. Sin embargo, la tasa de éxito estricto cayó a $44,53\,\%$ ($-4,87\text{ pp}$ respecto a `S1`), y los episodios que nunca alcanzaron el objetivo aumentaron al $34,07\,\%$ ($+3,94\text{ pp}$). El costo acumulado por modificar acciones generó una resistencia que ralentizó la aproximación a objetivos distantes. A su vez, los fallos por sobrepaso sin parada (`entered_no_hold`) permanecieron en $21,40\,\%$.

### 5.3. Paquete Combinado (`S3_combined`) **[HECHO MEDIDO]**
La integración conjunta de `train_freq = 4`, $w_a = 0{,}05$ y `target_entropy = -1.0` redujo sensiblemente la brecha frente a `PPO X3`:
* El éxito estricto medio se situó en **$84,53\,\% \pm 4,70\,\%$** ($+35,13\text{ pp}$ sobre `S1` y $+25,60\text{ pp}$ sobre `S2_freq`).
* La tasa de `never_entered` descendió al **$9,33\,\%$** y la de `entered_no_hold` al **$6,13\,\%$**.
* Las salidas de meta disminuyeron a **$0,067$ por episodio**, y la variación de acción se situó en $0,0457\text{ rad/s}$.
* La telemetría de TensorBoard reveló que fijar `target_entropy = -1.0` evitó el colapso temprano de la temperatura de entropía observado en las variantes previas ($\alpha \approx 0{,}018$–$0{,}024$), estabilizándola en torno a $\alpha \approx 0{,}060$.
* **Evaluación de la regla preregistrada:** En la semilla piloto 42, el resultado ($79,80\,\%$) se situó en la **Zona B (Ambiguo / Zona Gris, $[67,47\,\%, 87,73\,\%))$**. Al replicarse en las semillas 100 ($89,20\,\%$) y 2026 ($84,60\,\%$), la media consolidada de $84,53\,\%$ ratificó una mejora considerable respecto a las ablaciones aisladas, aunque manteniéndose por debajo de `PPO X3` ($93,53\,\%$) por $9,00\text{ pp}$. Al tratarse de un paquete simultáneo, los efectos individuales de cada componente no son directamente atribuibles de forma aislada.

---

## 6. Eficiencia Computacional y Rendimiento de Muestreo

Es metodológicamente necesario distinguir entre la **eficiencia de muestras** (rendimiento obtenido por paso de interacción en el entorno) y el **tiempo de reloj** (costo computacional de ejecución) **[HECHO MEDIDO]**:

1. **Eficiencia de muestras:** En igualdad estricta de pasos de interacción ($245\,760$ pasos), `PPO X3` obtiene un rendimiento superior al de todas las variantes de SAC evaluadas: supera a `S1` por $+44,13\text{ pp}$, a `S2_freq` por $+34,60\text{ pp}$ y a `S3_combined` por $+9,00\text{ pp}$. Esto demuestra que en este problema de control con parada fina, PPO aprovecha cada muestra de interacción de manera más eficaz que las variantes de SAC evaluadas.
2. **Tiempo de reloj y rendimiento de ejecución:**
   * **`PPO X3` (3 semillas, $245\,760$ pasos):** Requirió **$7{,}62\text{ minutos}$** ($457{,}1\text{ s}$) a una tasa media de **$1\,614{,}3\text{ FPS}$**, medido directamente desde las marcas de tiempo de ejecución de sus tres semillas hasta el punto de control correspondiente.
   * **`SAC S2_freq` (3 semillas, $245\,760$ pasos):** Requirió **$28{,}24\text{ minutos}$** a una tasa de **$435{,}1\text{ FPS}$**.
   * **`SAC S3_combined` (3 semillas, $245\,760$ pasos):** Requirió **$30{,}40\text{ minutos}$** a una tasa de **$404{,}2\text{ FPS}$**.
   * **`SAC S1` y `S2_wa` (3 semillas, $245\,760$ pasos):** Requirieron **$94{,}86\text{ min}$** y **$99{,}23\text{ min}$** respectivamente ($\approx 125\text{ FPS}$).

PPO X3 no solo alcanza una tasa de éxito $9\text{ pp}$ superior a la mejor configuración de SAC probada, sino que lo hace requiriendo aproximadamente una cuarta parte del tiempo de cómputo de `S3_combined` y una doceava parte del tiempo de `S1`.

---

## 7. Palancas Experimentales: Probadas vs. No Evaluadas

Para delimitar con rigor el alcance de este estudio y evitar extrapolaciones no fundamentadas sobre el algoritmo SAC en general, se sintetizan a continuación las palancas evaluadas frente a aquellas que permanecen fuera de la exploración:

\begin{table}[htbp]
\centering
\caption{Clasificación de palancas algorítmicas y de diseño en la Fase C.}
\label{tab:palancas_evaluadas_no_evaluadas}
\small
\begin{tabular}{lll}
\hline
\textbf{Categoría} & \textbf{Estado} & \textbf{Detalle de la Configuración} \\
\hline
Frecuencia de actualización ($\text{UTD}$) & Evaluada & $\text{UTD} = 1{,}0$ (`S1`) vs. $\text{UTD} = 0{,}25$ (`S2\_freq`, `S3\_combined`) \\
Penalización de acción ($w_a$) & Evaluada & $w_a = 0{,}0$ (`S1`), $w_a = 0{,}10$ (`S2\_wa`), $w_a = 0{,}05$ (`S3\_combined`) \\
Temperatura de entropía ($\mathcal{H}_{\text{tgt}}$) & Evaluada & Auto ($\mathcal{H}_{\text{tgt}} = -2{,}0$) vs. $\mathcal{H}_{\text{tgt}} = -1{,}0$ (`S3\_combined`) \\
Combinación simultánea & Evaluada & Paquete `S3\_combined` (3 semillas, 245\,760 pasos) \\
\hline
Hold de entrenamiento alineado & No evaluada & Entrenar con hold estricto (20 pasos, $0{,}10\text{ rad/s}$) \\
Penalización de velocidad ($w_v$) & No evaluada & Término gaussiano $w_v > 0$ en la meta \\
Temperatura con piso explícito & No evaluada & Coeficiente $\alpha$ fijo o con límite inferior $\alpha_{\text{min}}$ \\
Capacidad del buffer de repetición & No evaluada & Tamaños distintos de $300\,000$ transiciones \\
Tasa y calendario de aprendizaje & No evaluada & Valores distintos de $3 \times 10^{-4}$ constante \\
Arquitectura de redes separadas & No evaluada & Capas dedicadas desacopladas o activaciones alternativas \\
Normalización de recompensas & No evaluada & `norm_reward = True` en `VecNormalize` \\
\hline
\end{tabular}
\end{table}

---

## 8. Hipótesis Mecanísticas No Evaluadas

La literatura y el análisis teórico sugieren hipótesis para explicar la discrepancia de desempeño observada entre ambos algoritmos en este entorno cinemático. Estas consideraciones se presentan estrictamente como **[HIPÓTESIS NO EVALUADAS]**, dado que no fueron aisladas mediante experimentos específicos:

1. **Observabilidad parcial y propagación temporal:** El contador de permanencia en meta (`hold_counter`) no forma parte del vector de observación de 10 dimensiones en ningún modelo. Se ha postulado que la estimación generalizada de ventaja (`GAE`, $\lambda = 0{,}95$) en PPO propaga la señal de recompensa a lo largo de rollouts continuos de $2\,048$ pasos de manera más coherente que el bootstrap $\text{TD}(0)$ de SAC sobre transiciones barajadas. No obstante, debe señalarse que **PPO tampoco observa el contador de hold**, por lo que la observabilidad parcial por sí sola no constituye una explicación suficiente sin considerar la estructura de actualización de cada algoritmo.
2. **Dinámica sin inercia y actualización acotada:** El simulador modela un manipulador estrictamente cinemático ($\boldsymbol{\dot{\theta}} \propto \mathbf{a}$). Se hipotetiza que el recorte en el espacio de políticas de PPO (`clip_range = 0.2`) introduce una regularización implícita que mitiga gradientes locales excesivos en la frontera de meta, mientras que el gradiente analítico $\nabla_a Q(s,a)$ en SAC podría inducir cambios bruscos en presencia de superficies de valor discontinuas. Esta conjetura no ha sido contrastada mediante análisis de gradientes in situ.
3. **Dispersión estocástica ($\sigma$):** La desviación típica paramétrica de la distribución de acciones de SAC se estimó en $\sigma \approx 0{,}20$ en el diagnóstico inicial. Se señala explícitamente que **dicho valor no fue verificado sistemáticamente sobre los estados reales visitados cerca de la meta**, por lo que permanece como una estimación preliminar no validada.

---

## 9. Veredicto y Cierre Formal de la Fase C

En conformidad con las directrices del proyecto y el conjunto de resultados empíricos consolidados:

1. **Desempeño relativo medido [HECHO MEDIDO]:** Con $245\,760$ pasos de interacción por semilla (3 semillas evaluadas sobre 500 episodios deterministas), ninguna de las variantes de SAC superó la tasa de éxito de `PPO X3` ($93,53\,\% \pm 2,50\,\%$):
   * Las variantes unifactoriales quedaron por detrás por más de 34 puntos porcentuales (`S2_freq`: $58,93\,\% \pm 3,80\,\%$; `S1`: $49,40\,\% \pm 9,17\,\%$; `S2_wa`: $44,53\,\% \pm 2,44\,\%$).
   * La variante combinada `S3_combined` redujo sustancialmente la brecha, alcanzando un **$84,53\,\% \pm 4,70\,\%$**, situándose a $9,00\text{ pp}$ de `PPO X3`.
2. **Costo-beneficio del proyecto:** `PPO X3` no solo mantiene la mayor tasa de éxito y la menor variabilidad inter-semilla ($2{,}50\,\%$ frente a $4{,}70\,\%$), sino que completó su entrenamiento en $7{,}62$ minutos frente a los $30{,}40$ minutos requeridos por `S3_combined` y los $94$–$99$ minutos de `S1` y `S2_wa`.
3. **Veredicto formal de cierre:** En el estado actual del proyecto y para las palancas específicas probadas, se determina que continuar la optimización de SAC no resulta favorable en términos de costo-beneficio frente al desempeño ya alcanzado por PPO. No se concluye nada sobre la capacidad intrínseca de SAC en general ni sobre palancas que no fueron evaluadas. Por consiguiente, **se da por concluida la exploración de SAC en la Fase C**, adoptándose las conclusiones para los informes finales del repositorio.

---

## 10. Verificación de Integridad y Trazabilidad

Se certifica el cumplimiento estricto de las directivas de no destructividad del repositorio **[HECHO MEDIDO]**:
* El dataset de prueba final `data/datasets/phase_c/final_c_500.json` permaneció estrictamente sellado durante todas las etapas de la investigación (firma SHA-256: `72f3bb0556b01b391e85ba03fb595480f12b703596b6bc968c8789f5a114d3eb`).
* El entorno de simulación `arm2d_env.py` y el evaluador determinista `src/evaluation/evaluator.py` no sufrieron modificación alguna.
* Las configuraciones y checkpoints históricos de Fase A, Fase B (`experiments/phase_b/`), `S1`, `S2_freq`, `S2_wa` y `S3_combined` conservan sus firmas de integridad y aislamiento.
* Todos los artefactos de datos, telemetría y métricas correspondientes a esta etapa se encuentran archivados con rutas relativas dentro de `reports/phase_c_s3_analysis/`.
