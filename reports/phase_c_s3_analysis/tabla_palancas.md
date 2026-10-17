# Clasificación de Palancas de Optimización en SAC (Fase C)

Este documento detalla el estado experimental de cada una de las palancas algorítmicas consideradas en el proyecto `robot-arm-2dof-ppo` para el agente Soft Actor-Critic (SAC).

---

## 1. Palancas Evaluadas Empíricamente

| Palanca | Configuración | Variantes | Efecto Medido al Paso 245 760 |
| :--- | :--- | :--- | :--- |
| **Frecuencia de Actualización** | Modificación de `train_freq` de 1 a 4 pasos ($\text{UTD} = 0{,}25$) con `gradient_steps: 1`. | `S2_freq`, `S3_combined` | Incrementó el éxito de $49{,}40\,\%$ a $58{,}93\,\%$ (unifactorial) y aceleró el entrenamiento a $435\text{ FPS}$. |
| **Penalización por Variación de Acción** | Regularización cuadrática $r_{\text{smooth}} = -w_a \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$ con $w_a = 0{,}10$. | `S2_wa` | Redujo la variación de acción a $0{,}0378\text{ rad/s}$ ($-26{,}9\,\%$) pero degradó el éxito a $44{,}53\,\%$ al ralentizar la aproximación. |
| **Regularización Moderada de Acción** | Regularización cuadrática con $w_a = 0{,}05$. | `S3_combined` | Amortiguó oscilaciones sin frenar la aproximación, contribuyendo a una tasa de éxito agregada de $84{,}53\,\%$. |
| **Temperatura de Entropía Relajada** | Ajuste explícito de `target_entropy = -1.0` frente al valor automático de $-\dim(\mathcal{A}) = -2{,}0$. | `S3_combined` | Evitó el colapso temprano de entropía, estabilizando $\alpha \approx 0{,}060$ frente a $\alpha \approx 0{,}018$–$0{,}024$ de las variantes previas. |
| **Paquete Combinado No Atribuible** | Integración simultánea de `train_freq: 4`, $w_a = 0{,}05$ y `target_entropy: -1.0`. | `S3_combined` | Alcanzó $84{,}53\,\% \pm 4{,}70\,\%$ de éxito estricto en 3 semillas ($30{,}40\text{ min}$ de cómputo acumulado). |

---

## 2. Palancas No Evaluadas

| Palanca | Razón para no evaluar en esta fase | Impacto Teórico Potencial |
| :--- | :--- | :--- |
| **Hold de Entrenamiento Alineado** | Se mantuvo el hold relajado histórico (`hold_steps = 5`, `velocity_tolerance = 0.30`) para preservar la pureza unifactorial frente a `S1`. | Podría incentivar la parada estacionaria dentro de la meta durante el entrenamiento. |
| **Penalización Gaussiana de Velocidad ($w_v$)** | Descartada temporalmente tras calibración de escalas en C-1 para no mezclar dos términos de penalización simultáneos. | Ayudaría a disipar energía cinética en las proximidades del objetivo sin penalizar el trayecto inicial. |
| **Piso Inferior de Temperatura ($\alpha_{\text{min}}$)** | Se evaluó únicamente `target_entropy = -1.0`, sin acotar explícitamente un piso estricto sobre $\alpha$. | Evitaría que la política colapse a determinismo en regímenes prolongados de entrenamiento. |
| **Capacidad del Buffer de Repetición** | Se mantuvo fijo en $300\,000$ transiciones en todas las ejecuciones. | Modificar la capacidad altera la frescura de los datos y la diversidad de transiciones en el bootstrap. |
| **Tasa y Calendario de Aprendizaje** | Se mantuvo constante en $3 \times 10^{-4}$ con optimizador Adam. | Un esquema de decaimiento (*learning rate schedule*) podría favorecer el ajuste fino cerca de la convergencia. |
| **Arquitecturas Alternativas de Red** | Se mantuvo fija la estructura de dos capas ocultas de 256 neuronas ReLU. | Redes con capas desacopladas para actor y crítico podrían reducir interferencias representacionales. |
| **Normalización de Recompensas** | Se fijó `norm_reward = False` en `VecNormalize` para evitar distorsionar la señal de recompensa aditiva. | Podría estabilizar la escala del crítico $Q$ ante la bonificación terminal de $+15{,}0$. |
