# Verificaciones y Auditoría Técnica: Etapa 1 (Fase C)

Este documento registra los hallazgos y verificaciones metodológicas realizadas durante la Etapa 1 antes de la ejecución del piloto combinado `S3_combined`.

---

## 1. Verificación del Tiempo de Entrenamiento de PPO X3 [HECHO MEDIDO]
* **Tiempo medido en el paso 245 760 (3 semillas):** **$7{,}62\text{ minutos}$** ($457{,}1\text{ s}$) a una tasa media de **$1\,614{,}3\text{ FPS}$**.
* **Origen de la cifra:** Calculado directamente a partir de las marcas de tiempo de inicio y finalización del punto de control `ckpt_245760` en los registros de entrenamiento de `PPO X3` para las semillas 42, 100 y 2026.
* **Origen de la discrepancia de 18–21 min:** Dicha cifra correspondía al entrenamiento completo de $655\,360$ a $1\,000\,000$ de pasos de la Fase B, no al punto homólogo de $245\,760$ pasos.
* **Distinción metodológica:** En pasos de simulación iguales, PPO ya supera ampliamente a SAC en eficiencia de muestras (rendimiento por paso), con independencia del tiempo de reloj o de los FPS.

---

## 2. Confirmación de Rutas de Salida
* Todos los archivos generados en la iteración previa se ubicaron en `reports/phase_c_s2_analysis/`.
* Para la presente iteración y cierre formal, la estructura consolidada reside en `reports/phase_c_s3_analysis/`.

---

## 3. Comparación Temporal en el Punto de Control 163 840 Pasos
En el paso $163\,840$, `SAC S2_freq` alcanzó su pico sincrónico de éxito estricto con $61{,}07\,\% \pm 1{,}10\,\%$. En ese mismo paso exacto, los modelos registraron:
* **`PPO X3`:** $65{,}40\,\% \pm 6{,}90\,\%$ (semilla 42: $58{,}60\,\%$, semilla 100: $72{,}80\,\%$, semilla 2026: $64{,}80\,\%$).
* **`SAC S2_freq`:** $61{,}07\,\% \pm 1{,}10\,\%$ (semilla 42: $61{,}80\,\%,$ semilla 100: $61{,}60\,\%$, semilla 2026: $59{,}80\,\%$).
* **`SAC S1`:** $47{,}80\,\% \pm 6{,}00\,\%$.
* **`SAC S2_wa`:** $40{,}07\,\% \pm 0{,}76\,\%$.
* **`SAC S3_combined`:** $78{,}27\,\% \pm 3{,}06\,\%$.

---

## 4. Estado de la Desviación Típica de Acción ($\sigma$) en SAC
* La afirmación previa de "$\sigma \approx 0{,}20$ comparable a PPO" se derivó de los parámetros de la red Gaussiana, pero **no fue verificada empíricamente sobre estados reales visitados cerca de la meta**.
* Se rotula formalmente como **[NO VERIFICADO SOBRE ESTADOS REALES]**.

---

## 5. Examen de los Mecanismos de la Sección 5 [HIPÓTESIS NO EVALUADAS]
1. **Observabilidad parcial del contador de hold:** Se constata que **PPO tampoco observa `hold_counter`** en su vector de observación de 10 dimensiones. La observabilidad parcial afecta a ambos algoritmos por igual, por lo que no constituye un factor diferenciador sin considerar la estructura de actualización de políticas.
2. **Estimación multi-paso vs. TD(0):** Hipótesis teórica plausible que no fue aislada experimentalmente (por ejemplo, mediante SAC multi-paso con $n$-step returns).
3. **Clipping de PPO como filtro pasa-bajos:** Hipótesis teórica no contrastada empíricamente mediante medición directa del espectro de frecuencias de las acciones.

---

## 6. Verificación de Métricas en Semilla 42 al Paso 245 760
* **`PPO X3`:** $92{,}40\,\%$ ($462 / 500$ episodios exitosos).
* **`SAC S1`:** $59{,}40\,\%$ ($297 / 500$ episodios exitosos). La cifra tentativa de $55{,}60\,\%$ no existe en los registros de checkpoints.
* **`SAC S2_freq`:** $62{,}80\,\%$ ($314 / 500$).
* **`SAC S2_wa`:** $42{,}40\,\%$ ($212 / 500$).
