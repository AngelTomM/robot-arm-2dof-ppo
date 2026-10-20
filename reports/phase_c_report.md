# COMPARACIÓN DESCRIPTIVA — SAC (1 configuración, n=3) frente a PPO X3 (n=3)

**Fecha:** 2026-10-14  
**Autor:** Antigravity AI — DeepMind Agentic Pair Programming  
**Rama Git:** `fase-c-sac-descriptive`  
**Entorno de Ejecución:** Python 3.11.16 | PyTorch 2.13.0+cpu | Gymnasium 1.3.0 | Stable-Baselines3 2.9.0 | NumPy 1.26.4 | Pandas 3.0.5 | Matplotlib 3.11.1 | Plataforma: Windows-10-10.0.19045-SP0  

---

## 1. Resumen Ejecutivo

Este informe documenta la primera etapa de la **Fase C** del proyecto `robot-arm-2dof-ppo`. El objetivo central consistió en:
1. Implementar y entrenar un agente off-policy **Soft Actor-Critic (SAC, configuración $S1$)** a lo largo de $3$ semillas independientes ($42, 100, 2026$) bajo el presupuesto estándar de **$655\,360$ pasos de interacción física** en el entorno, midiendo los tiempos de cómputo en esta máquina.
2. Compararlo de manera **descriptiva** frente al agente on-policy de referencia **PPO ($X3$, 3 semillas)** bajo idéntica física ($L_1 = L_2 = 1{,}0\text{ m}$, $v_{\max} = 1{,}5\text{ rad/s}$, $\Delta t = 0{,}05\text{ s}$), función de recompensa (`smooth_precision`), arquitectura de red ($256 \times 256$ ReLU) y protocolo estricto de estabilización (*hold* durante $20$ pasos consecutivos con $\|\dot{\boldsymbol{\theta}}\| \le 0{,}10\text{ rad/s}$ en $r_{\text{tgt}} = 0{,}06\text{ m}$).

> [!NOTE]
> **Alcance Descriptivo:** Al tratarse de una única configuración inicial de SAC sin ajuste ni barrido de hiperparámetros frente a PPO (que fue optimizado en las Fases A y B), **no se emiten veredictos globales de superioridad ni se declaran grupos de equivalencia**. Toda diferencia menor a $\sim 4\text{ pp}$ se considera dentro del ruido muestral.

### Principales Hallazgos Empíricos [HECHOS Medidos e HIPÓTESIS Rotuladas]:
1. [HECHO] **Ventaja temprana de $S1$:**  
   En el paso $40\,960$, $S1$ alcanza un éxito estricto de **$28{,}07\% \pm 7{,}27\%$** frente a **$0{,}27\% \pm 0{,}23\%$** de $X3$ (rangos inter-semilla disjuntos: $[23{,}0\%, 36{,}4\%]$ vs. $[0{,}0\%, 0{,}4\%]$). [HIPÓTESIS]: Atribuir este arranque a la eficiencia muestral del buffer de repetición requiere verificación empírica mediante comparación aislada de un solo factor.
2. [HECHO] **Evolución comparativa y no solapamiento:**  
   Entre los pasos $163\,840$ y $655\,360$, $S1$ se mantiene en un rango de éxito estricto medio de entre $45\%$ y $53\%$, alcanzando su máximo en el paso $327\,680$ ($52{,}40\% \pm 6{,}01\%$) y finalizando en el paso $655\,360$ en **$46{,}60\% \pm 9{,}21\%$**. Por su parte, $X3$ alcanza un máximo de **$97{,}40\% \pm 0{,}20\%$** en el paso $409\,600$, experimentando posteriormente un descenso de $-8{,}60\text{ pp}$ hasta cerrar en **$88{,}80\% \pm 7{,}70\%$** (comportamiento decreciente que ya se había medido en la Fase B). Desde el paso $204\,800$ hasta el final ($655\,360$), los rangos entre semillas de ambos algoritmos no se solapan en ningún checkpoint.
3. [HECHO] **Descomposición de fallos en el paso $655\,360$:**  
   En $S1$, el efector final ingresa al radio objetivo en el $74{,}73\%$ de los episodios ($25{,}27\%$ de los episodios nunca ingresan al radio de tolerancia). Del total de episodios, un $28{,}13\%$ entra pero no consolida el hold estricto, alcanzando un éxito del $46{,}60\%$ (razón éxito/entrada $\approx 0{,}62$). La tasa total de truncamientos ($53{,}40\%$) equivale exactamente al complemento del éxito ($25{,}27\% + 28{,}13\%$) y no constituye un hallazgo independiente. En $X3$, el $92{,}00\%$ ingresa ($8{,}00\%$ nunca ingresa), un $3{,}20\%$ entra sin hold y el $88{,}80\%$ alcanza el éxito estricto (razón éxito/entrada $\approx 0{,}97$).
4. [HECHO] **Evolución de suavidad articular $\overline{\|\Delta \mathbf{a}\|}$:**  
   La variación media de acción $\overline{\|\Delta \mathbf{a}\|}$ (promediada sobre la totalidad de los $500$ episodios evaluados) pasa en $S1$ de $0{,}0236 \pm 0{,}0028$ a **$0{,}2041 \pm 0{,}0512$** en el paso final (incremento de $8{,}6\times$). En $X3$, pasa de $0{,}0026 \pm 0{,}0001$ a **$0{,}0480 \pm 0{,}0014$**. [HIPÓTESIS]: La asociación entre el incremento de $\overline{\|\Delta \mathbf{a}\|}$ y un régimen oscilatorio en campo cercano permanece como hipótesis hasta inspeccionar perfiles espaciales de error por episodio.
5. [HECHO] **Colapso sistemático en evaluación estocástica:**  
   Tanto $S1$ como $X3$ registraron **$0{,}00\%$** de éxito estricto y **$100{,}00\%$** de truncamientos en modo estocástico sobre los checkpoints finales ($0/500$ episodios por semilla). [HIPÓTESIS]: La dispersión gaussiana continua $\sigma$ sobre una ventana temporal de $20$ pasos consecutivos con cota estrecha ($\|\dot{\boldsymbol{\theta}}\| \le 0{,}10\text{ rad/s}$) interrumpe sistemáticamente la condición de parada.
6. [HECHO / DERIVADO] **Asimetría de cómputo medida y derivada:**  
   A igual presupuesto de entorno ($655\,360$ pasos):
   - **Pasos de gradiente [DERIVADO del YAML]:** SAC ejecutó **$650\,360$ pasos** (ratio 1:1 tras $5\,000$ pasos de calentamiento); PPO ejecutó **$102\,400$ pasos** ($80\text{ rollouts} \times 10\text{ épocas} \times 128\text{ lotes}$). Razón SAC/PPO: **$\approx 6{,}35\times$** (no $254\times$).
   - **Tiempo de reloj por corrida:** SAC requirió una media de **$86{,}88\text{ min}$** ($5\,212{,}75\text{ s}$ a $125{,}72\text{ fps}$; $4{,}34\text{ h}$ acumuladas en $n=3$) [MEDIDO]; PPO requirió en Fase B **$\approx 7{,}15\text{ min}$** ($429{,}5\text{ s}$ a $\approx 1\,526\text{ fps}$; $\approx 21{,}45\text{ min}$ acumuladas en $n=3$) [DERIVADO de Fase B]. Razón de tiempo SAC/PPO: **$\approx 12{,}15\times \approx 12\times$** [DERIVADA].

---

## 2. Marco Metodológico y No Destructividad

### 2.1 Preservación de Fases Anteriores
En cumplimiento con el principio de no destructividad:
* Ningún archivo preexistente de las Fases A y B (código fuente, modelos, checkpoints, configuraciones YAML, datasets, reportes ni figuras) fue modificado.
* Se verificó la integridad mediante la suite de pruebas unitarias en `tests/test_phase_c_validations.py`.

### 2.2 Conjuntos de Datos de Fase C
Se generaron dos nuevos conjuntos independientes siguiendo la distribución de muestreo canónica ($r_{\text{tgt}} \in [0{,}15, 1{,}90]\text{ m}$, $\theta_{1,\text{init}}, \theta_{2,\text{init}} \in [-\pi, \pi]$):
* **Validación C (`data/datasets/phase_c/validation_c_500.json`):** $500$ episodios, semilla maestra `202631`. Hash SHA-256: `f1f177e82515c095b8ad19cf661eb988eecf097852f05db100c3281206da39cb`. Utilizado para todas las evaluaciones descriptivas de esta parte.
* **Final C (`data/datasets/phase_c/final_c_500.json`):** $500$ episodios, semilla maestra `202641`. Hash SHA-256: `72f3bb0556b01b391e85ba03fb595480f12b703596b6bc968c8789f5a114d3eb`. **No se abre; integridad verificable por hash:** no fue leído ni evaluado en esta fase.
* **Disyunción estricta:** Se comprobó mediante verificación exhaustiva de 4-tuplas $(x_{\text{tgt}}, y_{\text{tgt}}, \theta_{1,\text{init}}, \theta_{2,\text{init}})$ que existe $0$ solapamiento entre los seis datasets del repositorio (`val_old`, `final_old`, `val_b2`, `final_b2`, `val_c`, `final_c`).

---

## 3. Configuración Algorítmica y Diferencias Declaradas

### 3.1 Hiperparámetros de SAC ($S1$, Valor Inicial a Validar)
Los parámetros fueron fijados en `configs/phase_c/S1.yaml`:

| Parámetro | Valor Inicial a Validar | Justificación / Relación con PPO $X3$ |
| :--- | :---: | :--- |
| **Algoritmo** | SAC (Stable-Baselines3 2.9.0) | Algoritmo off-policy actor-crítico con entropía máxima |
| **Red Actor ($\pi$)** | MLP $256 \times 256$ ReLU | Idéntica arquitectura que $X3$ ($2$ capas ocultas de $256$) |
| **Red Críticos ($Q_1, Q_2$)** | MLP $256 \times 256$ ReLU | Arquitectura simétrica doble Q estándar |
| **Tasa de aprendizaje** | $3 \times 10^{-4}$ | Constante para actor, críticos y temperatura |
| **Tamaño de lote (`batch_size`)** | $256$ | Mini-lote extraído del replay buffer |
| **Factor de descuento ($\gamma$)** | $0{,}99$ | Idéntico a $X3$ |
| **Actualización Polyak ($\tau$)** | $0{,}005$ | Tasa de seguimiento suave para redes objetivo |
| **Coeficiente térmico (`ent_coef`)** | `'auto'` | Adaptación de $\alpha$ con entropía meta $\bar{\mathcal{H}} = -\dim(\mathcal{A}) = -2{,}0$ |
| **Frecuencia (`train_freq`)** | $1\text{ paso}$ | Actualización por cada paso de interacción |
| **Pasos de gradiente** | $1$ | 1 actualización tras cada paso de entorno |
| **Pasos de calentamiento** | $5\,000$ | $25$ episodios aleatorios previos a gradientes |
| **Buffer de repetición** | $300\,000$ transiciones | Ocupa $\approx 28{,}6\text{ MB}$ en RAM |
| **Entornos paralelos** | $1$ (`DummyVecEnv`) | SAC opera con recolección secuencial estándar |
| **Normalización** | `VecNormalize(norm_obs=True, norm_reward=False)` | Normalización de observaciones desacoplada |

### 3.2 Diferencias Intrínsecas Declaradas entre SAC y PPO
1. **Espacio de Acciones:** SAC aplica una transformación no lineal tangente hiperbólica $\mathbf{a} = \tanh(\mathbf{u})$ a la salida gaussiana reparametrizada, acotando naturalmente $\mathbf{a} \in [-1, 1]^2$. PPO muestrea de una normal $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ y recorta (*clip*) las acciones a $[-1, 1]^2$.
2. **Normalización con Replay Buffer:** Según se verificó en `stable_baselines3/common/off_policy_algorithm.py` (líneas 473–475), el buffer almacena observaciones **originales en bruto**. Al muestrear durante el entrenamiento (`stable_baselines3/common/buffers.py`, líneas 316–324), se normalizan dinámicamente con las estadísticas corrientes de `VecNormalize`. Esto evita mezclar observaciones normalizadas con medias viejas, pero introduce no estacionariedad continua en los valores de entrada a los críticos.
3. **Manejo de Truncamiento:** En `stable_baselines3/common/buffers.py` (línea 322) y `stable_baselines3/sac/sac.py` (línea 260), la bandera `TimeLimit.truncated` descuenta el término de corte temporal en `dones`, computando $y = r + \gamma Q(s', a')$ en el paso $200$ para realizar el bootstrap del valor.
4. **Criterio de Entrenamiento:** PPO $X3$ fue entrenado con el mismo protocolo de hold relajado ($5$ pasos, $0{,}30\text{ rad/s}$) y alcanzó un éxito estricto elevado en validación ($97{,}40\%$ de pico); por consiguiente, el criterio relajado de entrenamiento no explica por sí solo la brecha de rendimiento frente a $S1$.

---

## 4. Costos de Cómputo y Tiempos de Ejecución

Los entrenamientos y evaluaciones se ejecutaron íntegramente en CPU sobre la misma máquina:

### 4.1 Tiempos de Entrenamiento ($655\,360$ pasos por corrida)

| Agente / Corrida | Semilla | Tiempo de Reloj (s) | Tiempo de Reloj (min) | Velocidad Media ($\text{fps}$) | Pasos de Gradiente | Tipo de Cifra |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **SAC ($S1$)** | $42$ | $5\,063{,}45\text{ s}$ | $84{,}39\text{ min}$ | $129{,}43\text{ fps}$ | $650\,360$ | Medida / Derivada |
| **SAC ($S1$)** | $100$ | $5\,204{,}99\text{ s}$ | $86{,}75\text{ min}$ | $125{,}91\text{ fps}$ | $650\,360$ | Medida / Derivada |
| **SAC ($S1$)** | $2026$ | $5\,369{,}80\text{ s}$ | $89{,}50\text{ min}$ | $122{,}05\text{ fps}$ | $650\,360$ | Medida / Derivada |
| **SAC ($S1$) Consolidado** | $n=3$ | **$15\,638{,}24\text{ s}$** | **$260{,}64\text{ min}$ ($4{,}34\text{ h}$)** | **$125{,}72\text{ fps}$** | **$1\,951\,080$** | Medida / Derivada |
| **PPO ($X3$) por semilla** | $1\text{ semilla}$ | $\approx 429{,}5\text{ s}$ | $\approx 7{,}15\text{ min}$ | $\approx 1\,526\text{ fps}$ | $102\,400$ | Derivada Fase B / YAML |
| **PPO ($X3$) Consolidado** | $n=3$ | **$\approx 1\,288{,}5\text{ s}$** | **$\approx 21{,}45\text{ min}$ ($0{,}36\text{ h}$)** | **$\approx 1\,526\text{ fps}$** | **$307\,200$** | Derivada Fase B / YAML |

### 4.2 Tiempos de Evaluación
* **Evaluación determinista ($96$ checkpoints $\times$ $500$ episodios + $C0$):** $938{,}4\text{ s}$ ($\approx 15{,}6\text{ minutos}$ a $9{,}67\text{ s}$ por checkpoint).
* **Diagnóstico estocástico ($6$ checkpoints finales):** $58{,}1\text{ s}$ ($\approx 1{,}0\text{ minuto}$).

---

## 5. Tabla Completa de Evaluación de Checkpoints en Validación C

Se evaluaron los $16$ checkpoints periódicos (del paso $40\,960$ al $655\,360$) para las $3$ semillas independientes de $S1$ y $X3$, totalizando $96$ evaluaciones de $500$ episodios bajo el criterio estricto ($r_{\text{tgt}}=0{,}06\text{ m}, v_{\text{tol}}=0{,}10\text{ rad/s}, N_{\text{hold}}=20$). 

Las columnas expresan la Media $\pm$ Desviación Estándar muestral inter-semilla ($ddof=1$, $n=3$). La mediana de pasos al éxito de $S1$ ($48{,}5$) frente a $X3$ ($61{,}7$) es estrictamente condicional a los episodios exitosos; [HIPÓTESIS]: $S1$ podría estar resolviendo preferentemente los episodios geométricamente más accesibles.

| Paso (k) | Éxito Estricto $S1$ (%) | Pasos Mediana $S1$ | Suavidad $\overline{\|\Delta \mathbf{a}\|}$ $S1$ | Éxito Estricto $X3$ (%) | Pasos Mediana $X3$ | Suavidad $\overline{\|\Delta \mathbf{a}\|}$ $X3$ | Diferencia $\Delta_{(S1 - X3)}$ (pp) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$41\text{k}$** | $28{,}07 \pm 7{,}27$ | $54{,}7 \pm 0{,}6$ | $0{,}0236 \pm 0{,}0028$ | $0{,}27 \pm 0{,}23$ | $133{,}8 \pm 1{,}1$ | $0{,}0026 \pm 0{,}0001$ | $+27{,}80$ |
| **$82\text{k}$** | $40{,}20 \pm 3{,}70$ | $52{,}3 \pm 1{,}2$ | $0{,}0326 \pm 0{,}0033$ | $10{,}53 \pm 5{,}13$ | $98{,}3 \pm 14{,}0$ | $0{,}0077 \pm 0{,}0004$ | $+29{,}67$ |
| **$123\text{k}$** | $43{,}27 \pm 4{,}09$ | $51{,}0 \pm 1{,}7$ | $0{,}0376 \pm 0{,}0018$ | $37{,}20 \pm 7{,}29$ | $73{,}5 \pm 0{,}9$ | $0{,}0156 \pm 0{,}0012$ | $+6{,}07$ |
| **$164\text{k}$** | $47{,}80 \pm 6{,}00$ | $50{,}7 \pm 0{,}6$ | $0{,}0404 \pm 0{,}0006$ | $65{,}40 \pm 6{,}90$ | $65{,}7 \pm 1{,}5$ | $0{,}0260 \pm 0{,}0017$ | $-17{,}60$ |
| **$205\text{k}$** | $49{,}13 \pm 4{,}09$ | $49{,}0 \pm 1{,}0$ | $0{,}0422 \pm 0{,}0037$ | $88{,}40 \pm 2{,}03$ | $60{,}7 \pm 1{,}2$ | $0{,}0364 \pm 0{,}0014$ | $-39{,}27$ |
| **$246\text{k}$** | $49{,}40 \pm 9{,}17$ | $49{,}7 \pm 0{,}6$ | $0{,}0517 \pm 0{,}0036$ | $93{,}53 \pm 2{,}50$ | $58{,}5 \pm 0{,}5$ | $0{,}0402 \pm 0{,}0021$ | $-44{,}13$ |
| **$287\text{k}$** | $52{,}13 \pm 8{,}08$ | $48{,}3 \pm 0{,}6$ | $0{,}0592 \pm 0{,}0102$ | $95{,}60 \pm 2{,}31$ | $58{,}3 \pm 1{,}2$ | $0{,}0426 \pm 0{,}0012$ | $-43{,}47$ |
| **$328\text{k}$** | $52{,}40 \pm 6{,}01$ | $49{,}0 \pm 1{,}0$ | $0{,}0866 \pm 0{,}0120$ | $95{,}80 \pm 0{,}72$ | $57{,}5 \pm 1{,}3$ | $0{,}0440 \pm 0{,}0013$ | $-43{,}40$ |
| **$369\text{k}$** | $48{,}47 \pm 6{,}18$ | $49{,}5 \pm 1{,}8$ | $0{,}0829 \pm 0{,}0146$ | $96{,}27 \pm 2{,}53$ | $57{,}0 \pm 1{,}7$ | $0{,}0456 \pm 0{,}0015$ | $-47{,}80$ |
| **$410\text{k}$** | $49{,}47 \pm 4{,}27$ | $49{,}7 \pm 1{,}2$ | $0{,}1006 \pm 0{,}0104$ | $97{,}40 \pm 0{,}20$ | $57{,}3 \pm 2{,}1$ | $0{,}0469 \pm 0{,}0018$ | $-47{,}93$ |
| **$451\text{k}$** | $50{,}27 \pm 3{,}24$ | $49{,}3 \pm 1{,}2$ | $0{,}0991 \pm 0{,}0080$ | $96{,}87 \pm 1{,}40$ | $57{,}7 \pm 1{,}2$ | $0{,}0483 \pm 0{,}0016$ | $-46{,}60$ |
| **$492\text{k}$** | $48{,}40 \pm 2{,}36$ | $49{,}7 \pm 0{,}6$ | $0{,}1106 \pm 0{,}0231$ | $93{,}53 \pm 2{,}66$ | $58{,}7 \pm 2{,}1$ | $0{,}0474 \pm 0{,}0021$ | $-45{,}13$ |
| **$532\text{k}$** | $47{,}73 \pm 5{,}03$ | $49{,}7 \pm 0{,}6$ | $0{,}1090 \pm 0{,}0333$ | $91{,}67 \pm 4{,}83$ | $59{,}8 \pm 1{,}9$ | $0{,}0465 \pm 0{,}0035$ | $-43{,}94$ |
| **$573\text{k}$** | $48{,}47 \pm 3{,}53$ | $49{,}3 \pm 1{,}2$ | $0{,}1218 \pm 0{,}0279$ | $86{,}80 \pm 4{,}04$ | $60{,}8 \pm 1{,}0$ | $0{,}0448 \pm 0{,}0031$ | $-38{,}33$ |
| **$614\text{k}$** | $45{,}67 \pm 4{,}44$ | $48{,}8 \pm 1{,}4$ | $0{,}1620 \pm 0{,}0677$ | $87{,}60 \pm 8{,}65$ | $61{,}5 \pm 3{,}0$ | $0{,}0456 \pm 0{,}0040$ | $-41{,}93$ |
| **$655\text{k}$** | $46{,}60 \pm 9{,}21$ | $48{,}5 \pm 0{,}9$ | $0{,}2041 \pm 0{,}0512$ | $88{,}80 \pm 7{,}70$ | $61{,}7 \pm 2{,}9$ | $0{,}0480 \pm 0{,}0014$ | $-42{,}20$ |

*Fuente:* `reports/data/phase_c/checkpoints_eval_validation_aggregated.csv`.

---

## 6. Comparación Descriptiva y Análisis de Patologías

### 6.1 Rangos Inter-Semilla y No Solapamiento
* En el paso $40\,960$, los rangos entre semillas medidos son: $S1 \in [23{,}0\%, 36{,}4\%]$ vs. $X3 \in [0{,}0\%, 0{,}4\%]$. Los rangos son completamente disjuntos a favor de $S1$.
* A partir del paso $204\,800$ y hasta el paso final ($655\,360$), los rangos de $X3$ ($[81{,}0\%, 96{,}4\%]$ en el paso final) y $S1$ ($[40{,}6\%, 57{,}2\%]$ en el paso final) **no se solapan en ningún checkpoint**. Descriptivamente, $X3$ se ubica en un régimen de rendimiento superior bajo el criterio de hold estricto.

### 6.2 Disociación entre Llegada y Hold en SAC [HIPÓTESIS]
¿Por qué $S1$ ingresa a la zona cercana ($d < 0{,}06\text{ m}$) en el $74{,}73\%$ de los casos pero solo aprueba el hold en el $46{,}60\%$?
* [HIPÓTESIS]: La función de recompensa `smooth_precision` incluye un término de penalización de velocidad local $-0{,}15(|\dot{\theta}_1| + |\dot{\theta}_2|)$ en $d < 0{,}30\text{ m}$. En SAC, el término de entropía $\alpha \mathcal{H}(\pi)$ incentiva la dispersión de acciones. Cerca del objetivo, el gradiente de entropía y la penalización de velocidad podrían entrar en conflicto con la cota estrecha $\|\dot{\boldsymbol{\theta}}\| \le 0{,}10\text{ rad/s}$. La política aprende a aproximarse rápidamente pero no logra sostener la cota de velocidad baja durante $20$ pasos consecutivos.
* El aumento medido de $\overline{\|\Delta \mathbf{a}\|}$ de $0{,}0236$ a $0{,}2041$ es compatible con esta hipótesis, pero su verificación causal requiere estudios específicos de ablación de un solo factor.

---

## 7. Galería de Figuras Generadas

Las siguientes figuras se encuentran archivadas en `reports/figures/phase_c/`:

1. **`fig_c_learning_curves_s1_vs_x3.png`:**  
   Curvas completas de aprendizaje determinista de $S1$ frente a $X3$ y $C0$. El panel inferior documenta la tasa de truncamientos, evidenciando que $S1$ mantiene truncamientos cercanos al $53\%$, mientras que $X3$ los reduce al $11{,}2\%$ en el paso final.
2. **`fig_c_sac_training_diagnostics.png`:**  
   Telemetría interna de SAC extraída de eventos TensorBoard ($n=3$ semillas):
   - **Pérdida de los críticos ($Q_1, Q_2$ MSE):** estabilización en rango $0{,}14\text{--}2{,}85$ tras fluctuaciones iniciales de exploración.
   - **Pérdida del actor ($\pi$ Loss):** convergencia suave desde $+15{,}2$ hasta $\sim -51{,}3$ a medida que se maximizan los valores $Q$.
   - **Temperatura de entropía ($\alpha$, `ent_coef`):** decaimiento asintótico desde $0{,}8357$ hasta $\approx 0{,}0180$, explicando la reducción de dispersión exploratoria y el aumento de reactividad.
   - **Retorno episódico acumulado (`ep_rew_mean`):** ascenso rápido desde $-385$ hasta la meseta de $+56\text{--}+68$.
3. **`fig_c_failure_breakdown_final_ckpt.png`:**  
   Distribución de desenlaces en el paso $655\,360$. Reconcilia al $100\%$: $S1$ reparte $46{,}60\%$ éxito y $53{,}40\%$ truncamientos, frente a $88{,}80\%$ éxito y $11{,}20\%$ truncamientos en $X3$.  
   *(Desviación de diseño: Se solicitó originalmente un mapa espacial 2D de fallos $(x_{\text{tgt}}, y_{\text{tgt}})$ para cada una de las 3 semillas; se entregó un gráfico de barras apiladas con el desglose porcentual agregado).*
4. **`fig_c_canonical_geometric_trajectories.png`:**  
   Planos de fase y perfiles de aproximación en los $4$ episodios geométricos representativos (Q25, Q50, Q75 y borde exterior).  
   *(Desviación de diseño: Se solicitó un plano de fase de velocidad articular $\|\dot{\boldsymbol{\theta}}\|$ contra distancia $d$; se graficó la distancia euclidiana cartesiana $d$ frente a su derivada temporal radial $\dot{d} = \frac{\mathrm{d}d}{\mathrm{d}t}$).*
5. **`fig_c_deterministic_vs_stochastic_profiles.png`:**  
   Respuesta temporal de velocidad articular $\|\dot{\boldsymbol{\theta}}\|(t)$ determinista vs. estocástica contra el umbral de hold ($0{,}10\text{ rad/s}$), explicando visualmente el colapso sistemático al $0\%$.
6. **`fig_c_compute_cost_and_runtime_comparison.png`:**  
   Contraste de costo computacional: tiempo de reloj ($260{,}64\text{ min}$ vs $\approx 21{,}45\text{ min}$, razón $\approx 12{,}15\times$) y pasos de gradiente ($650\,360$ vs $102\,400$, razón $\approx 6{,}35\times$).

---

## 8. Trazabilidad y Verificación de Integridad

| Elemento / Archivo | Ubicación en Repositorio | Estado de Verificación |
| :--- | :--- | :---: |
| **Configuración S1** | `configs/phase_c/S1.yaml` | Auditado |
| **Dataset Validación C** | `data/datasets/phase_c/validation_c_500.json` | SHA-256 verificado |
| **Dataset Final C** | `data/datasets/phase_c/final_c_500.json` | No se abre; integridad verificable |
| **Hashes de Datasets** | `reports/data/phase_c/datasets_hashes.json` | Registrado |
| **Checkpoints S1 ($n=3$)** | `experiments/phase_c/S1/` | 48 pares atómicos válidos |
| **Evaluación Agregada** | `reports/data/phase_c/checkpoints_eval_validation_aggregated.csv` | Generado |
| **Telemetría Canónica** | `reports/data/phase_c/step_data_val_geometry.csv` | 2 623 pasos registrados |
| **Diagnóstico Estocástico** | `reports/data/phase_c/stochastic_diagnostic_final_ckpt.csv` | Generado |
| **Figuras y Metadatos (12 archivos)** | `reports/figures/phase_c/` | 6 PNGs + 6 JSONs |

---

## 9. Caveats Metodológicos Obligatorios

1. **Una sola configuración de SAC sin ajuste:** $S1$ utiliza parámetros estándar (`lr=3e-4, batch=256, tau=0.005`). No se realizó ningún barrido de hiperparámetros. La conclusión sobre SAC aplica exclusivamente a esta formulación inicial.
2. **Asimetría de madurez:** PPO ($X3$) es el resultado de múltiples iteraciones de diseño (Fase A y Fase B), mientras que $S1$ representa el primer intento sin calibrar.
3. **Mismo presupuesto de entorno no implica igual cómputo:** SAC requirió $6{,}35\times$ más pasos de gradiente que PPO ($650\,360$ vs $102\,400$) y $12{,}15\times$ más tiempo de CPU por corrida ($86{,}88\text{ min}$ vs $7{,}15\text{ min}$).
4. **Muestras con $n=3$ semillas:** Con $3$ semillas, la dispersión no cubre toda la incertidumbre del espacio de políticas; los resultados son puramente descriptivos.
5. **Determinismo del entrenamiento de SAC:** No verificado con la misma semilla.
6. **Entorno de software:** PPO y SAC se evaluaron bajo la misma pila de librerías (`Python 3.11.16`, `PyTorch 2.13.0+cpu`, `SB3 2.9.0`).
7. **El $0\%$ estocástico estricto:** Es una consecuencia de la dispersión gaussiana continua $\sigma$ sobre una ventana temporal de $20$ pasos consecutivos con cota estrecha ($0{,}10\text{ rad/s}$); su causa física se declara como **[HIPÓTESIS]**.
8. **Incompatibilidad de comparación directa en pérdidas:** Las funciones de pérdida de SAC (Bellman residual en dos críticos y log-probabilidad reparametrizada en actor) no son comparables algebraicamente con la pérdida recortada de PPO.

### 9.1 Qué NO se puede concluir
* **Nada sobre SAC en general:** No se puede extrapolar este resultado como limitación intrínseca del algoritmo SAC frente a PPO, dado que se evaluó una única configuración inicial sin sintonización de hiperparámetros.
* **Ningún mecanismo causal como hecho:** Ninguna atribución causal (conflicto de entropía, dinámica de replay buffer, etc.) puede afirmarse como hecho demostrado sin un experimento controlado de ablación de un solo factor.
* **Ningún veredicto global ni ganador:** No se emite veredicto global de superioridad algorítmica ni se declara un algoritmo ganador.

---

## 10. Qué Haría Falta en una Extensión Posterior de Fase C (No Evaluado)

*Estas propuestas constituyen ideas futuras de investigación para una posible extensión; no fueron evaluadas aquí:*

1. **Penalización de Aceleración / Suavidad ($w_a$) en SAC:**  
   Incorporar una penalización de cambio de acción $-w_a \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$ en la recompensa o en la función de pérdida del actor para evaluar si contiene el incremento de $\overline{\|\Delta \mathbf{a}\|}$. *(Nota metodológica: en la Fase B con PPO, añadir $w_a = 0{,}01$ no mejoró la suavidad, por lo que esta palanca carece de garantía previa).*
2. **Calibración de la Temperatura de Entropía ($\alpha$):**  
   Evaluar valores fijos de $\alpha$ o modificar la entropía objetivo $\bar{\mathcal{H}}$ para estudiar su impacto en la dispersión en campo cercano.
3. **Ajuste de Frecuencia de Actualización (`train_freq` / `gradient_steps`):**  
   Evaluar ratios menores a 1:1 para reducir el costo computacional en CPU.
4. **Envolvente de Velocidad Admisible (*Velocity Funnel*):**  
   Modular la acción admisible en función de la distancia al objetivo.
5. **Apertura de Final C:**  
   Evaluar formalmente sobre el conjunto `data/datasets/phase_c/final_c_500.json` únicamente cuando se hayan seleccionado formalmente los candidatos finales de la extensión.
