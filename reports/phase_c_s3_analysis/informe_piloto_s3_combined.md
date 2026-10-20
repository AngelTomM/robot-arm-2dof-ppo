# Informe del Piloto Exploratorio Combinado: `S3_combined`

Este informe documenta los resultados del piloto combinado `S3_combined`, la aplicación formal de la regla de selección preregistrada en la semilla 42 y la replicación subsiguiente en las semillas 100 y 2026.

---

## 1. Definición del Paquete de Intervención

El agente `S3_combined` (`configs/phase_c/S3_combined.yaml`) introdujo simultáneamente tres modificaciones sobre el modelo base `S1`:
* `sac.train_freq: 4` con `sac.gradient_steps: 1` (reducción de actualizaciones de gradiente, UTD $= 0{,}25$).
* `env.reward_params.w_a: 0.05` (penalización moderada por variación de acción consecutiva).
* `sac.target_entropy: -1.0` (frente a $-\dim(\mathcal{A}) = -2{,}0$, favoreciendo exploración sostenida).
* Hold de entrenamiento relajado idéntico a `S1` (`hold_steps: 5`, `velocity_tolerance: 0.30`, $w_v = 0{,}0$).
* Presupuesto de $245\,760$ pasos por semilla con puntos de control atómicos cada $40\,960$ pasos.

---

## 2. Aplicación de la Regla Preregistrada en la Semilla 42

Antes de iniciar el entrenamiento, se preregistró formalmente en `configs/phase_c/selection_rule_s3_combined.yaml` la siguiente regla de decisión basada en la referencia de ruido muestral (ancho total del intervalo de Wilson al 95 % para $n = 500$ sobre el éxito de `PPO X3` en la semilla 42, $462 / 500 = 92{,}40\,\%$, ancho de banda $= 4{,}67\text{ pp}$):

* **Umbral A (Replicar a 3 semillas):** $\text{Éxito} \ge 92{,}40\,\% - 4{,}67\,\% = \mathbf{87{,}73\,\%}$
* **Umbral C (Cerrar exploración de SAC):** $\text{Éxito} < 62{,}80\,\% + 4{,}67\,\% = \mathbf{67{,}47\,\%}$ (referencia `S2_freq` semilla 42: $62{,}80\,\%$)
* **Zona B (Ambiguo / Decisión humana):** $[\mathbf{67{,}47\,\%}, \;\mathbf{87{,}73\,\%})$

### Resultado Medido en Semilla 42 al Paso 245 760 [HECHO MEDIDO]
* **Éxito estricto determinista:** **$79{,}80\,\%$** ($399 / 500$ episodios exitosos, IC Wilson 95 %: $[76{,}06\,\%, 83{,}09\,\%]$).
* **Fallo `never_entered`:** $13{,}40\,\%$
* **Fallo `entered_no_hold`:** $6{,}80\,\%$
* **Variación media de acción $\|\Delta \mathbf{a}\|$:** $0{,}0444\text{ rad/s}$
* **Salidas de meta ($n_{\text{exits}}$):** $0{,}072$ por episodio
* **Temperatura de entropía ($\alpha$):** $0{,}0602$
* **Tiempo de entrenamiento:** $10{,}14\text{ minutos}$ ($403{,}9\text{ FPS}$)

### Veredicto de la Regla Preregistrada:
El resultado de $79{,}80\,\%$ se ubicó estrictamente en la **Zona B (Ambiguo / Zona Gris)**: superó con holgura el umbral C ($+12{,}33\text{ pp}$) demostrando que el paquete combinado superaba la mejor variante previa, pero no alcanzó el umbral A de no inferioridad frente a PPO (a $-7{,}93\text{ pp}$).

Conforme al protocolo ("Decido yo"), el usuario autorizó la replicación en las semillas restantes (`100` y `2026`).

---

## 3. Resultados de la Replicación Completa ($N = 3$ Semillas)

\begin{table}[htbp]
\centering
\caption{Resultados consolidados de \texttt{S3\_combined} en el paso 245\,760 sobre \texttt{validation\_c\_500.json} (3 semillas deterministas, 500 episodios).}
\label{tab:s3_combined_3seeds_pilot}
\small
\begin{tabular}{lcccccc}
\hline
\textbf{Semilla} & \textbf{Éxito estricto} & \textbf{Never entered} & \textbf{Entered no hold} & \textbf{$\|\Delta \mathbf{a}\|$} & \textbf{Salidas meta} & \textbf{Tiempo ent.} \\
 & (\%) & (\%) & (\%) & (rad/s) & ($n_{\text{exits}}$) & (min) \\
\hline
Semilla 42 & 79,80 & 13,40 & 6,80 & 0,0444 & 0,072 & 10,14 \\
Semilla 100 & 89,20 & 6,20 & 4,60 & 0,0473 & 0,048 & 10,32 \\
Semilla 2026 & 84,60 & 8,40 & 7,00 & 0,0454 & 0,082 & 9,95 \\
\hline
\textbf{Media $\pm$ std} & $\mathbf{84,53 \pm 4,70}$ & $\mathbf{9,33 \pm 3,69}$ & $\mathbf{6,13 \pm 1,33}$ & $\mathbf{0,0457 \pm 0,0015}$ & $\mathbf{0,067 \pm 0,017}$ & $\mathbf{10,13 \pm 0,19}$ \\
\hline
\end{tabular}
\end{table}

---

## 4. Conclusiones del Piloto [HECHO MEDIDO]

1. **Eficacia del paquete combinado:** La integración simultánea de regulación del ratio UTD, amortiguación moderada de acción y temperatura de entropía sostenida elevó el éxito estricto de SAC al **$84{,}53\,\% \pm 4{,}70\,\%$**, representando una ganancia neta de $+35{,}13\text{ pp}$ sobre la línea base `S1` ($49{,}40\,\%$) y de $+25{,}60\text{ pp}$ sobre la variante unifactorial `S2_freq` ($58{,}93\,\%$).
2. **Brecha residual frente a PPO X3:** A pesar de la sustancial mejora, `PPO X3` retiene una ventaja neta de **$9{,}00\text{ puntos porcentuales}$** ($93{,}53\,\% \pm 2{,}50\,\%$ frente a $84{,}53\,\% \pm 4{,}70\,\%$), con menor variabilidad inter-semilla y requiriendo una cuarta parte del tiempo de cómputo ($7{,}62\text{ min}$ vs. $30{,}40\text{ min}$).
3. **No atribuibilidad:** Al evaluarse como paquete conjunto, no es posible atribuir cuantitativamente la fracción de mejora que corresponde de forma aislada a cada palanca.
