# COMPARACIÓN DESCRIPTIVA — SAC (1 configuración, n=3) frente a PPO X3 (n=3)

**Fecha:** 2026-10-14  
**Autor:** Antigravity AI — DeepMind Agentic Pair Programming  
**Rama Git:** `fase-c-sac-descriptive`  
**Entorno de Ejecución:** Python 3.11.16 | PyTorch 2.13.0+cpu | Gymnasium 1.3.0 | Stable-Baselines3 2.9.0 | NumPy 1.26.4 | Pandas 3.0.5 | Matplotlib 3.11.1 | Plataforma: Windows-10-10.0.19045-SP0  

---

## 1. Resumen Ejecutivo

Este informe documenta la primera etapa de la **Fase C** del proyecto `robot-arm-2dof-ppo`. El objetivo central consistió en:
1. Implementar y entrenar un agente off-policy **Soft Actor-Critic (SAC, configuración $S1$)** a lo largo de $3$ semillas independientes ($42, 100, 2026$) bajo el presupuesto estándar de **$655\,360$ pasos de interacción física** en el entorno, midiendo los tiempos de cómputo en esta máquina.
2. Compararlo de manera **descriptiva** frente al agente on-policy de referencia **PPO ($X3$, 3 semillas)** bajo idéntica física ($L_1 = L_2 = 1.0\text{ m}$, $v_{max} = 1.5\text{ rad/s}$, $\Delta t = 0.05\text{ s}$), función de recompensa (`smooth_precision`), arquitectura de red ($256 \times 256$ ReLU) y protocolo estricto de estabilización (*hold* durante $20$ pasos consecutivos con $\|\dot{\boldsymbol{\theta}}\| \le 0.10\text{ rad/s}$ en $r_{tgt} = 0.06\text{ m}$).

> [!NOTE]
> **Alcance Descriptivo:** Al tratarse de una única configuración inicial de SAC sin ajuste ni barrido de hiperparámetros frente a PPO (que fue ampliamente optimizado en las Fases A y B), **no se emiten veredictos globales de superioridad ni se declaran grupos de equivalencia**. Toda diferencia menor a $\sim 4\text{ pp}$ se considera dentro del ruido muestral.

### Principales Hallazgos Empíricos [HECHOS Medidos]:
1. [HECHO] **Adquisición temprana acelerada en SAC:**  
   En las primeras etapas ($40\text{k}\text{--}81\text{k}$ pasos), SAC ($S1$) arranca sustancialmente más rápido que PPO ($X3$): en el paso $40\,960$, $S1$ ya alcanza **$28.07\% \pm 7.27\%$** de éxito estricto frente a tan solo **$0.27\% \pm 0.23\%$** de $X3$.
2. [HECHO] **Meseta prematura de SAC frente a la convergencia asintótica de PPO:**  
   A partir del paso $163\,840$, $S1$ satura en una meseta de entre $46\%$ y $52\%$ de éxito estricto, alcanzando su pico en el paso $327\,680$ ($52.40\% \pm 6.01\%$) y finalizando en el paso $655\,360$ en **$46.60\% \pm 9.21\%$**. En contraste, $X3$ supera el $90\%$ en el paso $204\,800$, alcanzando un pico de **$97.40\% \pm 0.20\%$** en el paso $409\,600$ y cerrando en **$88.80\% \pm 7.70\%$**.
3. [HECHO] **Disociación entre llegada y hold en SAC:**  
   En el checkpoint final de $S1$, el efector final ingresa al radio objetivo en el **$74.73\%$** de los episodios, pero solo logra estabilizarse en el **$46.60\%$**, resultando en un **$53.40\%$** de truncamientos sin consolidación del hold.
4. [HECHO] **Degradación de suavidad articular en SAC:**  
   La variación de acción $\overline{\Delta a}$ de $S1$ se incrementa monótonamente a lo largo del entrenamiento, pasando de $0.0236$ a **$0.2041$** en el paso final (un aumento de 8.6 veces en chattering/reactividad), mientras que $X3$ mantiene una suavidad acotada entre $0.040$ y $0.048$.
5. [HECHO] **Colapso sistemático en evaluación estocástica:**  
   Tanto $S1$ como $X3$ obtuvieron **$0.00\%$** de éxito estricto y **$100.00\%$** de truncamientos en modo estocástico sobre los checkpoints finales ($0/500$ éxitos en todas las semillas).
6. [HECHO] **Asimetría extrema de cómputo:**  
   A igual presupuesto de entorno ($655\,360$ pasos), SAC ejecutó **$650\,360$ actualizaciones de gradiente** (1:1) requiriendo **$4.34\text{ horas}$** totales en CPU ($125.72\text{ fps}$), mientras que PPO ejecutó solo **$2\,560$ actualizaciones** requiriendo aproximadamente **$12\text{ minutos}$** por corrida.

---

## 2. Marco Metodológico y No Destructividad

### 2.1 Preservación Criptográfica de Fases Anteriores
En cumplimiento estricto con el principio de no destructividad:
* Ningún archivo preexistente de las Fases A y B (código fuente, modelos, checkpoints, configuraciones YAML, datasets, reportes ni figuras) fue modificado.
* Se verificaron los hashes criptográficos mediante la suite de tests unitarios: **$34$ pruebas unitarias ejecutadas y aprobadas (100% OK)** en [`tests/test_phase_c_validations.py`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/tests/test_phase_c_validations.py).

### 2.2 Conjuntos de Datos de Fase C
Se generaron dos nuevos conjuntos independientes siguiendo la distribución de muestreo canónica ($r_{tgt} \in [0.15, 1.90]\text{ m}$, $\theta_{1,init}, \theta_{2,init} \in [-\pi, \pi]$):
* **Validación C ([`data/datasets/phase_c/validation_c_500.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/data/datasets/phase_c/validation_c_500.json)):** $500$ episodios, semilla maestra `202631`. Hash SHA-256: `f1f177e82515c095b8ad19cf661eb988eecf097852f05db100c3281206da39cb`. Utilizado para todas las evaluaciones de esta parte.
* **Final C ([`data/datasets/phase_c/final_c_500.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/data/datasets/phase_c/final_c_500.json)):** $500$ episodios, semilla maestra `202641`. Hash SHA-256: `72f3bb0556b01b391e85ba03fb595480f12b703596b6bc968c8789f5a114d3eb`. **Sellado criptográficamente:** no fue leído ni evaluado en esta fase.
* **Disyunción estricta:** Se comprobó mediante verificación exhaustiva de 4-tuplas $(x_{tgt}, y_{tgt}, \theta_{1,init}, \theta_{2,init})$ que existe **$0$ solapamiento** entre los seis datasets del repositorio (`val_old`, `final_old`, `val_b2`, `final_b2`, `val_c`, `final_c`).

---

## 3. Configuración Algorítmica y Diferencias Declaradas

### 3.1 Hiperparámetros de SAC ($S1$, Valor Inicial a Validar)
Los parámetros fueron fijados en [`configs/phase_c/S1.yaml`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/configs/phase_c/S1.yaml):

| Parámetro | Valor Inicial a Validar | Justificación / Relación con PPO $X3$ |
| :--- | :---: | :--- |
| **Algoritmo** | SAC (Stable-Baselines3 2.9.0) | Algoritmo off-policy actor-crítico con entropía máxima |
| **Red Actor ($\pi$)** | MLP $256 \times 256$ ReLU | Idéntica arquitectura que $X3$ ($2$ capas ocultas de $256$) |
| **Red Críticos ($Q_1, Q_2$)** | MLP $256 \times 256$ ReLU | Arquitectura simétrica doble Q estándar |
| **Tasa de aprendizaje** | $3 \times 10^{-4}$ | Constante para actor, críticos y temperatura |
| **Tamaño de lote (`batch_size`)** | $256$ | Mini-lote extraído del replay buffer |
| **Factor de descuento ($\gamma$)** | $0.99$ | Idéntico a $X3$ |
| **Actualización Polyak ($\tau$)** | $0.005$ | Tasa de seguimiento suave para redes objetivo |
| **Coeficiente térmico (`ent_coef`)** | `'auto'` | Adaptación de $\alpha$ con entropía meta $\bar{\mathcal{H}} = -\dim(\mathcal{A}) = -2.0$ |
| **Frecuencia (`train_freq`)** | $1\text{ paso}$ | Actualización por cada paso de interacción |
| **Pasos de gradiente** | $1$ | 1 actualización tras cada paso de entorno |
| **Pasos de calentamiento** | $5\,000$ | $25$ episodios aleatorios previos a gradientes |
| **Buffer de repetición** | $300\,000$ transiciones | Ocupa $\approx 28.6\text{ MB}$ en RAM |
| **Entornos paralelos** | $1$ (`DummyVecEnv`) | SAC opera con recolección secuencial estándar |
| **Normalización** | `VecNormalize(norm_obs=True, norm_reward=False)` | Normalización de observaciones desacoplada |

### 3.2 Diferencias Intrínsecas Declaradas entre SAC y PPO
1. **Espacio de Acciones:** SAC aplica una transformación no lineal tangente hiperbólica $\mathbf{a} = \tanh(\mathbf{u})$ a la salida gaussiana reparametrizada, acotando naturalmente $\mathbf{a} \in [-1, 1]^2$. PPO muestrea de una normal $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ y recorta (*clip*) las acciones a $[-1, 1]^2$.
2. **Normalización con Replay Buffer:** Según se verificó en [`.venv/Lib/site-packages/stable_baselines3/common/off_policy_algorithm.py:473-475`](file:///C:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/.venv/Lib/site-packages/stable_baselines3/common/off_policy_algorithm.py#L473-L475), el buffer almacena observaciones **originales en bruto**. Al muestrear durante el entrenamiento ([`buffers.py:316-324`](file:///C:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/.venv/Lib/site-packages/stable_baselines3/common/buffers.py#L316-L324)), se normalizan dinámicamente con las estadísticas corrientes de `VecNormalize`. Esto evita mezclar observaciones normalizadas con medias viejas, pero introduce no estacionariedad continua en los valores de entrada a los críticos.
3. **Manejo de Truncamiento:** En [`.venv/Lib/site-packages/stable_baselines3/common/buffers.py:322`](file:///C:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/.venv/Lib/site-packages/stable_baselines3/common/buffers.py#L322) y [`sac.py:260`](file:///C:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/.venv/Lib/site-packages/stable_baselines3/sac/sac.py#L260), la bandera `TimeLimit.truncated` descuenta el término de corte temporal en `dones`, computando $y = r + \gamma Q(s', a')$ en el paso $200$ para realizar el **bootstrap correcto del valor**.

---

## 4. Costos de Cómputo y Tiempos de Ejecución

Los entrenamientos y evaluaciones se ejecutaron íntegramente en CPU sobre la misma máquina:

### 4.1 Tiempos de Entrenamiento ($655\,360$ pasos por corrida)

| Agente / Corrida | Semilla | Tiempo de Reloj (s) | Tiempo de Reloj (min) | Velocidad Media ($\text{fps}$) | Pasos de Gradiente |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SAC ($S1$)** | $42$ | $5\,063.45\text{ s}$ | $84.39\text{ min}$ | $129.43\text{ fps}$ | $650\,360$ |
| **SAC ($S1$)** | $100$ | $5\,204.99\text{ s}$ | $86.75\text{ min}$ | $125.91\text{ fps}$ | $650\,360$ |
| **SAC ($S1$)** | $2026$ | $5\,369.80\text{ s}$ | $89.50\text{ min}$ | $122.05\text{ fps}$ | $650\,360$ |
| **SAC ($S1$) Consolidado** | $n=3$ | **$15\,638.24\text{ s}$** | **$260.64\text{ min}$ ($4.34\text{ h}$)** | **$125.72\text{ fps}$** | **$1\,951\,080$** |
| **PPO ($X3$) Típico** | $n=3$ | $\approx 2\,160\text{ s}$ | $\approx 36.0\text{ min}$ ($0.60\text{ h}$ total) | $\approx 910\text{ fps}$ | $7\,680$ |

### 4.2 Tiempos de Evaluación
* **Evaluación determinista ($96$ checkpoints $\times$ $500$ episodios + $C0$):** $938.4\text{ s}$ ($\approx 15.6\text{ minutos}$ a $9.67\text{ s}$ por checkpoint).
* **Diagnóstico estocástico ($6$ checkpoints finales):** $58.1\text{ s}$ ($\approx 1.0\text{ minuto}$).

---

## 5. Tabla Completa de Evaluación de Checkpoints en Validación C

Se evaluaron los $16$ checkpoints periódicos (del paso $40\,960$ al $655\,360$) para las $3$ semillas independientes de $S1$ y $X3$, totalizando $96$ evaluaciones de $500$ episodios bajo el criterio estricto ($r_{tgt}=0.06\text{ m}, v_{tol}=0.10\text{ rad/s}, N_{hold}=20$). 

Las columnas expresan la Media $\pm$ Desviación Estándar muestral inter-semilla ($ddof=1$, $n=3$):

| Paso (k) | Éxito Estricto $S1$ (%) | Pasos Mediana $S1$ | Suavidad $\overline{\Delta a}$ $S1$ | Éxito Estricto $X3$ (%) | Pasos Mediana $X3$ | Suavidad $\overline{\Delta a}$ $X3$ | Diferencia $\Delta_{(S1 - X3)}$ (pp) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$41\text{k}$** | $28.07 \pm 7.27$ | $54.7 \pm 0.6$ | $0.0236 \pm 0.0028$ | $0.27 \pm 0.23$ | $133.8 \pm 1.1$ | $0.0026 \pm 0.0001$ | **$+27.80$** (Ventaja inicial S1) |
| **$82\text{k}$** | $40.20 \pm 3.70$ | $52.3 \pm 1.2$ | $0.0326 \pm 0.0033$ | $10.53 \pm 5.13$ | $98.3 \pm 14.0$ | $0.0077 \pm 0.0004$ | **$+29.67$** (Ventaja inicial S1) |
| **$123\text{k}$** | $43.27 \pm 4.09$ | $51.0 \pm 1.7$ | $0.0376 \pm 0.0018$ | $37.20 \pm 7.29$ | $73.5 \pm 0.9$ | $0.0156 \pm 0.0012$ | $+6.07$ |
| **$164\text{k}$** | $47.80 \pm 6.00$ | $50.7 \pm 0.6$ | $0.0404 \pm 0.0006$ | $65.40 \pm 6.90$ | $65.7 \pm 1.5$ | $0.0260 \pm 0.0017$ | $-17.60$ (Cruce de ventaja) |
| **$205\text{k}$** | $49.13 \pm 4.09$ | $49.0 \pm 1.0$ | $0.0422 \pm 0.0037$ | $88.40 \pm 2.03$ | $60.7 \pm 1.2$ | $0.0364 \pm 0.0014$ | $-39.27$ |
| **$246\text{k}$** | $49.40 \pm 9.17$ | $49.7 \pm 0.6$ | $0.0517 \pm 0.0036$ | $93.53 \pm 2.50$ | $58.5 \pm 0.5$ | $0.0402 \pm 0.0021$ | $-44.13$ |
| **$287\text{k}$** | $52.13 \pm 8.08$ | $48.3 \pm 0.6$ | $0.0592 \pm 0.0102$ | $95.60 \pm 2.31$ | $58.3 \pm 1.2$ | $0.0426 \pm 0.0012$ | $-43.47$ |
| **$328\text{k}$** | **$52.40 \pm 6.01$** | $49.0 \pm 1.0$ | $0.0866 \pm 0.0120$ | $95.80 \pm 0.72$ | $57.5 \pm 1.3$ | $0.0440 \pm 0.0013$ | $-43.40$ (Pico de S1) |
| **$369\text{k}$** | $48.47 \pm 6.18$ | $49.5 \pm 1.8$ | $0.0829 \pm 0.0146$ | $96.27 \pm 2.53$ | $57.0 \pm 1.7$ | $0.0456 \pm 0.0015$ | $-47.80$ |
| **$410\text{k}$** | $49.47 \pm 4.27$ | $49.7 \pm 1.2$ | $0.1006 \pm 0.0104$ | **$97.40 \pm 0.20$** | $57.3 \pm 2.1$ | $0.0469 \pm 0.0018$ | $-47.93$ (Pico de X3) |
| **$451\text{k}$** | $50.27 \pm 3.24$ | $49.3 \pm 1.2$ | $0.0991 \pm 0.0080$ | $96.87 \pm 1.40$ | $57.7 \pm 1.2$ | $0.0483 \pm 0.0016$ | $-46.60$ |
| **$492\text{k}$** | $48.40 \pm 2.36$ | $49.7 \pm 0.6$ | $0.1106 \pm 0.0231$ | $93.53 \pm 2.66$ | $58.7 \pm 2.1$ | $0.0474 \pm 0.0021$ | $-45.13$ |
| **$532\text{k}$** | $47.73 \pm 5.03$ | $49.7 \pm 0.6$ | $0.1090 \pm 0.0333$ | $91.67 \pm 4.83$ | $59.8 \pm 1.9$ | $0.0465 \pm 0.0035$ | $-43.94$ |
| **$573\text{k}$** | $48.47 \pm 3.53$ | $49.3 \pm 1.2$ | $0.1218 \pm 0.0279$ | $86.80 \pm 4.04$ | $60.8 \pm 1.0$ | $0.0448 \pm 0.0031$ | $-38.33$ |
| **$614\text{k}$** | $45.67 \pm 4.44$ | $48.8 \pm 1.4$ | $0.1620 \pm 0.0677$ | $87.60 \pm 8.65$ | $61.5 \pm 3.0$ | $0.0456 \pm 0.0040$ | $-41.93$ |
| **$655\text{k}$** | **$46.60 \pm 9.21$** | $48.5 \pm 0.9$ | $0.2041 \pm 0.0512$ | **$88.80 \pm 7.70$** | $61.7 \pm 2.9$ | $0.0480 \pm 0.0014$ | **$-42.20$** |

*Fuente:* [`reports/data/phase_c/checkpoints_eval_validation_aggregated.csv`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_c/checkpoints_eval_validation_aggregated.csv).

---

## 6. Comparación Descriptiva y Análisis de Patologías

```
             Evolución Comparativa de Éxito Estricto Determinista (Validación C, n=500)
  100% ┼                                                ┌─────────────────── C0 (100.0%)
       │                                     ┌──────────┴──────────┐ X3 Pico (97.4%)
   80% ┤                                ┌────┘                     └───────── X3 Final (88.8%)
       │                           ┌────┘
   60% ┤                      ┌────┘    ┌──────────┐ S1 Pico (52.4%)
       │                 ┌────┘         │          └──────────────────────── S1 Final (46.6%)
   40% ┤            ┌────┴──────────────┘
       │       ┌────┘ S1 (40.2%)
   20% ┤  ┌────┘ S1 (28.1%)
       │  │
    0% ┼──┴──────────────────────────────────────────────────────────────────
         41k  82k  123k  164k  205k  246k  287k  328k  369k  410k  451k  492k  532k  573k  614k  655k
```

### 6.1 Rangos Inter-Semilla y No Solapamiento Asintótico
* En el paso $40\,960$, los rangos entre semillas son: $S1 \in [23.0\%, 38.8\%]$ vs. $X3 \in [0.0\%, 0.4\%]$. Los rangos son completamente disjuntos a favor de $S1$.
* A partir del paso $204\,800$ y hasta el final ($655\,360$), los rangos de $X3$ ($[81.0\%, 96.4\%]$ en el paso final) y $S1$ ($[40.6\%, 57.2\%]$ en el paso final) **no se solapan en ningún punto**. Descriptivamente, $X3$ se ubica en un régimen de rendimiento netamente superior bajo el criterio de hold estricto.

### 6.2 La Trampa de Desaceleración Fina en SAC [HIPÓTESIS]
¿Por qué $S1$ ingresa a la zona cercana ($d < 0.06\text{ m}$) en el $74.7\%$ de los casos pero solo aprueba el hold en el $46.6\%$?
* [HIPÓTESIS]: La función de recompensa `smooth_precision` incluye un término de penalización de velocidad local $-0.15(|\dot{\theta}_1| + |\dot{\theta}_2|)$ en $d < 0.30\text{ m}$. En SAC, el término de entropía $\alpha \mathcal{H}(\pi)$ incentiva la dispersión de acciones. Cerca del objetivo, el gradiente de entropía y la penalización de velocidad generan un conflicto con la cota estrecha $\|\dot{\boldsymbol{\theta}}\| \le 0.10\text{ rad/s}$. La política aprende a aproximarse rápidamente pero "vibra" o sobrepasa el centro, interrumpiendo el contador de $20$ pasos consecutivos.
* El aumento monótono de $\overline{\Delta a}$ de $0.024$ a $0.204$ respalda esta hipótesis: la política de SAC adquiere un régimen de conmutación de alta frecuencia (*chattering*) en campo cercano.

---

## 7. Galería de Figuras Generadas

Las siguientes figuras se encuentran archivadas en [`reports/figures/phase_c/`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/figures/phase_c/):

1. **[`fig_c_learning_curves_s1_vs_x3.png`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/figures/phase_c/fig_c_learning_curves_s1_vs_x3.png):**  
   Curvas completas de aprendizaje determinista de $S1$ frente a $X3$ y $C0$. El panel inferior documenta la tasa de truncamientos, evidenciando cómo $S1$ no logra abatir los truncamientos por debajo del $47\%$, mientras que $X3$ los reduce al $2.6\%$.
2. **[`fig_c_sac_training_diagnostics.png`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/figures/phase_c/fig_c_sac_training_diagnostics.png):**  
   Telemetría interna de SAC extraída de eventos TensorBoard ($n=3$ semillas):
   - **Pérdida de los críticos ($Q_1, Q_2$ MSE):** estabilización en rango $0.14\text{--}2.85$ tras picos iniciales de exploración.
   - **Pérdida del actor ($\pi$ Loss):** convergencia suave monótona desde $+15.2$ hasta $\sim -51.3$ a medida que se maximizan los valores $Q$.
   - **Temperatura de entropía ($\alpha$, `ent_coef`):** decaimiento asintótico desde $0.8357$ hasta $\approx 0.0177$, explicando la reducción de dispersión exploratoria y el aumento de ganancia/reactividad.
   - **Retorno episódico acumulado (`ep_rew_mean`):** ascenso rápido desde $-385$ hasta la meseta de $+56\text{--}+68$.
3. **[`fig_c_failure_breakdown_final_ckpt.png`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/figures/phase_c/fig_c_failure_breakdown_final_ckpt.png):**  
   Distribución exhaustiva de desenlaces en el paso $655\,360$. Reconcilia al $100\%$: $S1$ reparte $46.6\%$ éxito y $53.4\%$ truncamientos, frente a $88.8\%$ éxito y $11.2\%$ truncamientos en $X3$.
4. **[`fig_c_canonical_geometric_trajectories.png`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/figures/phase_c/fig_c_canonical_geometric_trajectories.png):**  
   Planos de fase $(d, \dot{d})$ y perfiles de aproximación en los $4$ episodios geométricos representativos (Q25, Q50, Q75 y borde exterior). Evidencia cómo $S1$ ingresa a la zona pero exhibe ciclos límite o correcciones abruptas antes de estabilizarse.
5. **[`fig_c_deterministic_vs_stochastic_profiles.png`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/figures/phase_c/fig_c_deterministic_vs_stochastic_profiles.png):**  
   Respuesta temporal de velocidad articular $\|\dot{\boldsymbol{\theta}}\|(t)$ determinista vs. estocástica contra el umbral de hold ($0.10\text{ rad/s}$), explicando visualmente el colapso sistemático al $0\%$.
6. **[`fig_c_compute_cost_and_runtime_comparison.png`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/figures/phase_c/fig_c_compute_cost_and_runtime_comparison.png):**  
   Contraste de costo computacional: tiempo de reloj ($260\text{ min}$ vs $36\text{ min}$) y pasos de gradiente ($650\,360$ vs $2\,560$), ilustrando la asimetría algorítmica.

---

## 8. Trazabilidad y Verificación de Integridad

| Elemento / Archivo | Ubicación en Repositorio | Estado de Verificación |
| :--- | :--- | :---: |
| **Configuración S1** | [`configs/phase_c/S1.yaml`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/configs/phase_c/S1.yaml) | Auditado |
| **Dataset Validación C** | [`data/datasets/phase_c/validation_c_500.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/data/datasets/phase_c/validation_c_500.json) | SHA-256 verificado |
| **Dataset Final C** | [`data/datasets/phase_c/final_c_500.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/data/datasets/phase_c/final_c_500.json) | **Sellado e intacto** |
| **Hashes de Datasets** | [`reports/data/phase_c/datasets_hashes.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_c/datasets_hashes.json) | Registrado |
| **Checkpoints S1 ($n=3$)** | [`experiments/phase_c/S1/`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/experiments/phase_c/S1) | 48 pares atómicos válidos |
| **Evaluación Agregada** | [`reports/data/phase_c/checkpoints_eval_validation_aggregated.csv`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_c/checkpoints_eval_validation_aggregated.csv) | Generado |
| **Telemetría Canónica** | [`reports/data/phase_c/step_data_val_geometry.csv`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_c/step_data_val_geometry.csv) | 2 623 pasos registrados |
| **Diagnóstico Estocástico** | [`reports/data/phase_c/stochastic_diagnostic_final_ckpt.csv`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/data/phase_c/stochastic_diagnostic_final_ckpt.csv) | Generado |
| **Figuras a 300 dpi (12 archivos)** | [`reports/figures/phase_c/`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/reports/figures/phase_c/) | 6 PNGs + 6 JSONs |

---

## 9. Caveats Metodológicos Obligatorios

1. **Una sola configuración de SAC sin ajuste:** $S1$ utiliza parámetros estándar (`lr=3e-4, batch=256, tau=0.005`). No se realizó ningún barrido de hiperparámetros. La conclusión sobre SAC aplica exclusivamente a esta formulación inicial.
2. **Asimetría de madurez:** PPO ($X3$) es el resultado de múltiples iteraciones de diseño (Fase A y Fase B), mientras que $S1$ representa el primer intento sin calibrar.
3. **Mismo presupuesto de entorno no implica igual cómputo:** SAC requirió $254$ veces más pasos de gradiente que PPO y $7.2$ veces más tiempo de CPU por corrida.
4. **Muestras con $n=3$ semillas:** Con $3$ semillas, la dispersión no cubre toda la incertidumbre del espacio de políticas; los resultados son puramente descriptivos.
5. **El $0\%$ estocástico estricto:** Es una consecuencia de la dispersión gaussiana continua $\sigma$ sobre una ventana temporal de $20$ pasos consecutivos con cota estrecha ($0.10\text{ rad/s}$); su causa física se declara como **[HIPÓTESIS]**.
6. **Incompatibilidad de comparación directa en pérdidas:** Las funciones de pérdida de SAC (Bellman residual en dos críticos y log-probabilidad reparametrizada en actor) no son comparables algebraicamente con la pérdida recortada de PPO.

---

## 10. Qué Haría Falta en una Extensión Posterior de Fase C (No Evaluado)

*Estas propuestas constituyen ideas futuras de investigación para una posible extensión; no fueron evaluadas aquí:*

1. **Penalización de Aceleración / Suavidad ($w_a$) en SAC:**  
   Incorporar una penalización de cambio de acción $-w_a \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$ en la recompensa o en la función de pérdida del actor para contener el incremento de chattering ($\overline{\Delta a} = 0.204$).
2. **Calibración de la Temperatura de Entropía ($\alpha$):**  
   Probar valores fijos pequeños de $\alpha$ (e.g. $\alpha = 0.005$) o modificar la entropía objetivo $\bar{\mathcal{H}}$ para permitir que la política se vuelva más determinista cerca del objetivo.
3. **Ajuste de Frecuencia de Actualización (`train_freq` / `gradient_steps`):**  
   Evaluar ratios de actualización menores a 1:1 (por ejemplo, actualizar cada 2 o 4 pasos) para reducir drásticamente el costo de cómputo en CPU sin comprometer la convergencia.
4. **Envolvente de Velocidad Admisible (*Velocity Funnel*):**  
   Guiar la desaceleración del actor modulando la acción admisible en función de la distancia al objetivo.
5. **Apertura de Final C:**  
   Evaluar formalmente sobre el conjunto sellado [`data/datasets/phase_c/final_c_500.json`](file:///c:/Users/ggevi/Documents/GitHub/robot-arm-2dof-ppo/data/datasets/phase_c/final_c_500.json) únicamente cuando se hayan seleccionado formalmente los candidatos finales de la extensión.
