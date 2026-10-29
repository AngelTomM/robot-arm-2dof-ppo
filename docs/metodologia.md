# Metodología — Manipulador Planar 2-DOF (Fase D: MDP Continuo con Regulación LQR en MuJoCo)

El presente documento expone los fundamentos físicos, cinemáticos, metodológicos y de teoría de control del proyecto: la caracterización del manipulador en el motor de física dinámica MuJoCo (`Reacher-v5`), la transición formal desde formulaciones parcialmente observables (POMDP con *hold counter*) hacia un Proceso de Decisión de Markov (MDP) continuo puro, el diseño de la función de recompensa continua de regulación LQR y el protocolo de evaluación dual desacoplada.

---

## 1. Modelo Físico y Cinemático en MuJoCo

La planta física corresponde al modelo de dos grados de libertad `Reacher-v5` provisto por Gymnasium bajo el motor de física de cuerpos rígidos MuJoCo. El robot se encuentra anclado a una base estática en el plano $(x, y)$ a una cota constante $z = 0.01\text{ m}$.

### 1.1 Parámetros Geométricos, Inerciales y de Actuación
* **Eslabón 1 (Brazo proximal):** Longitud $L_1 = 0.10\text{ m}$, masa $m_1 = 0.035605\text{ kg}$, articulado a la base mediante la junta de revolución continua $\theta_1 \in (-\infty, \infty)$.
* **Eslabón 2 y Efector Final (Brazo distal):** Longitud combinada $L_2 = 0.11\text{ m}$ (compuesta por el eslabón rígido de $0.10\text{ m}$ y el cuerpo esférico del efector final *fingertip* de radio $0.01\text{ m}$ con masa $m_{\text{fingertip}} = 0.004189\text{ kg}$). Posee tope mecánico articular en $\theta_2 \in [-3.0, 3.0]\text{ rad} \approx [-171.89^\circ, 171.89^\circ]$.
* **Alcance Cinemático:** El radio de alcance máximo del efector final es $r_{\max} = L_1 + L_2 = 0.21\text{ m}$, mientras que el radio mínimo admisible por la restricción angular es $r_{\min} = \sqrt{L_1^2 + L_2^2 + 2 L_1 L_2 \cos(3.0\text{ rad})} \approx 0.037\text{ m}$.
* **Actuadores y Transmisión:** Cada articulación es excitada directamente por torque mediante actuadores continuos de relación de engrane $\text{gear} = 200.0\text{ N}\cdot\text{m}$. La señal de control normalizada del agente $\mathbf{a}_t \in [-1.0, 1.0]^2$ se escala linealmente al rango físico de saturación.
* **Integración Temporal:** El resolvedor numérico de MuJoCo opera con un paso temporal interno de $\Delta t = 0.01\text{ s}$ y un factor de submuestreo $\text{frame\_skip} = 2$, estableciendo un período de control discreto efectivo de $\Delta t_{\text{eff}} = 0.02\text{ s}$ ($50\text{ Hz}$).

### 1.2 Cinemática Directa Planar
La posición cartesiana de la articulación intermedia del codo $(x_{\text{codo}}, y_{\text{codo}})$ y del efector final $(x_{ee}, y_{ee})$ respecto al marco de referencia inercial de la base se determina como:
$$x_{\text{codo}} = L_1 \cos(\theta_1), \quad y_{\text{codo}} = L_1 \sin(\theta_1)$$
$$x_{ee} = x_{\text{codo}} + L_2 \cos(\theta_1 + \theta_2)$$
$$y_{ee} = y_{\text{codo}} + L_2 \sin(\theta_1 + \theta_2)$$

### 1.3 Matriz Jacobiana Analítica ($J(\boldsymbol{\theta}) \in \mathbb{R}^{2 \times 2}$)
La relación diferencial de velocidades cartesianas respecto a las velocidades articulares $\dot{\boldsymbol{\theta}} = [\dot{\theta}_1, \dot{\theta}_2]^T$ se obtiene a partir de la derivada temporal de la cinemática directa:
$$\dot{\mathbf{x}}_{ee} = J(\boldsymbol{\theta}) \, \dot{\boldsymbol{\theta}}$$
$$J(\boldsymbol{\theta}) = \begin{bmatrix}
-L_1 \sin(\theta_1) - L_2 \sin(\theta_1 + \theta_2) & -L_2 \sin(\theta_1 + \theta_2) \\
L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2) & L_2 \cos(\theta_1 + \theta_2)
\end{bmatrix}$$
Esta formulación analítica fue contrastada unitariamente contra la función cinemática matricial `mj_jacBody` de MuJoCo, garantizando consistencia numérica con un error absoluto residual inferior a $10^{-5}\text{ m/(rad}\cdot\text{s)}$.

### 1.4 Espacio de Observaciones $\mathbf{s}_t \in \mathbb{R}^{10}$
El vector de realimentación suministrado a las políticas de control y al baseline analítico se extrae directamente de las coordenadas cinemáticas y dinámicas de MuJoCo:
$$\mathbf{s}_t = \left[\cos(\theta_1), \, \cos(\theta_2), \, \sin(\theta_1), \, \sin(\theta_2), \, x_{\text{tgt}}, \, y_{\text{tgt}}, \, \dot{\theta}_1, \, \dot{\theta}_2, \, (x_{ee} - x_{\text{tgt}}), \, (y_{ee} - y_{\text{tgt}})\right]^T$$
donde la descomposición en seno y coseno elimina la discontinuidad trigonométrica en $\pm \pi$, y el vector diferencial $(x_{ee} - x_{\text{tgt}}, y_{ee} - y_{\text{tgt}})$ provee la distancia cartesiana relativa al punto de consigna.

---

## 2. Transición hacia la Fase D: De POMDP a MDP Continuo Puro

### 2.1 Limitaciones Teóricas del Paradigma Anterior (*Hold Condition*)
En iteraciones previas del entorno, la condición de éxito se definía mediante la permanencia del efector final dentro de una tolerancia esférica durante $N = 20$ pasos consecutivos, momento en el cual se aplicaba un bono terminal positivo y se truncaba prematuramente el episodio.

Dicha formulación introducía una violación estructural a la propiedad de Markov:
1. **Parcial Observabilidad (POMDP):** El contador interno $k \in \{0, \dots, 20\}$ no formaba parte del vector de observación $\mathbf{s}_t \in \mathbb{R}^{10}$. En consecuencia, dos estados con idéntica postura cinemática y velocidad poseían valores de retorno esperado radicalmente divergentes dependiendo del estado de la variable latente $k$, desestabilizando la convergencia del crítico y la estimación de ventajas (GAE).
2. **Incentivos Perversos:** La recompensa terminal incentivaba políticas que alcanzaban transitoriamente la esfera objetivo sin amortiguamiento suficiente, provocando oscilaciones de alta frecuencia al intentar maximizar el bono terminal antes de la terminación anticipada.

### 2.2 Formulación de Horizonte Fijo Estrictamente Markoviana
La Fase D elimina el contador de permanencia y las terminaciones prematuras por éxito:
* **Horizonte Temporal Fijo:** Todo episodio concluye estrictamente al alcanzar el límite prefijado de $T_{\max} = 200$ pasos ($4.0\text{ s}$ de tiempo simulado).
* **Señales de Estado:** Se establece invariablemente `terminated = False`. La señal `truncated = True` se emite de forma exclusiva al cumplirse $t = T_{\max}$.
* **Propiedad Markoviana:** La distribución de probabilidad de transición $P(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t)$ y la función de recompensa instantánea $r(\mathbf{s}_t, \mathbf{a}_t)$ dependen de manera exclusiva del par estado-acción actual.

---

## 3. Función de Recompensa Continua Cuadrática (LQR / Lyapunov)

El aprendizaje del agente se guía mediante un funcional continuo cuadrático inspirado en el Regulador Lineal Cuadrático (LQR) y en funciones de Lyapunov para control de posición:
$$r(\mathbf{s}_t, \mathbf{a}_t) = - \left(w_{\text{pos}} \|\mathbf{x}_{ee} - \mathbf{x}_{\text{tgt}}\|^2 + w_{\text{ctrl}} \|\mathbf{a}_t\|^2 + w_{\text{vel}} \|\dot{\boldsymbol{\theta}}_t\|^2\right)$$
donde los términos componentes cumplen funciones de regulación física específicas:
* **Error Cartesiano Cuadrático ($\|\mathbf{x}_{ee} - \mathbf{x}_{\text{tgt}}\|^2$):** Provee un gradiente continuo y suave en todo el espacio de trabajo accesible, incrementando la atracción hacia el objetivo a medida que el efector se aproxima sin presentar discontinuidades de frontera. Se pondera con $w_{\text{pos}} = 1.0$.
* **Disipación de Energía Cinética ($\|\dot{\boldsymbol{\theta}}_t\|^2 = \dot{\theta}_1^2 + \dot{\theta}_2^2$):** Penaliza la velocidad angular residual en las articulaciones, forzando al controlador a amortiguar el movimiento para lograr reposo en el régimen permanente. Se pondera con $w_{\text{vel}} = 0.05$.
* **Regularización del Esfuerzo de Control ($\|\mathbf{a}_t\|^2 = a_1^2 + a_2^2$):** Penaliza los torques excesivos aplicados por los motores, reduciendo la saturación de los actuadores y previniendo oscilaciones parásitas. Se pondera con $w_{\text{ctrl}} = 0.01$.

---

## 4. Protocolo de Evaluación Dual Desacoplada

Para garantizar objetividad y evitar que la normalización de observaciones o recompensas en tiempo de entrenamiento afecte las métricas de desempeño, la evaluación se efectúa sobre la trayectoria física cruda a lo largo de conjuntos deterministas reproducibles de episodios (`validation_set_200.json` y `final_eval_set_300.json`).

### 4.1 Métrica Principal — Estabilización Terminal en Ventana $k = 20$
Se evalúa el comportamiento asintótico en los últimos 20 pasos de control del episodio ($t \in [T-19, T]$):
$$\bar{e}_{\text{ss}} = \frac{1}{20} \sum_{t=T-19}^T \|\mathbf{x}_{ee}(t) - \mathbf{x}_{\text{tgt}}\|, \qquad \bar{v}_{\text{ss}} = \frac{1}{20} \sum_{t=T-19}^T \|\dot{\boldsymbol{\theta}}(t)\|$$
Un episodio se clasifica formalmente como exitoso si y solo si satisface simultáneamente las tolerancias de reposo y proximidad:
$$\text{is\_success} = \mathbb{I}\left(\bar{e}_{\text{ss}} \le 0.02\text{ m} \;\wedge\; \bar{v}_{\text{ss}} \le 0.10\text{ rad/s}\right)$$

La tasa de éxito agregada sobre $N$ episodios se complementa con el cálculo del intervalo de confianza asintótico de Wilson al 95%:
$$\hat{p} = \frac{S}{N}, \qquad \text{IC}_{95\%} = \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}} \quad (z = 1.96)$$

### 4.2 Métrica Comparativa Canónica — Costo Integral Cuadrático ($J_{\text{LQR}}$)
Permite comparar agentes de aprendizaje y controladores analíticos bajo una métrica escalar independiente del algoritmo:
$$J_{\text{LQR}} = - \sum_{t=1}^T \left(1.0 \cdot \|\mathbf{x}_{ee}(t) - \mathbf{x}_{\text{tgt}}\|^2 + 0.01 \cdot \|\mathbf{a}(t)\|^2 + 0.05 \cdot \|\dot{\boldsymbol{\theta}}(t)\|^2\right)$$

### 4.4 Protocolo Multisemilla y Muestra Consolidada
Para cuantificar la variabilidad intrínseca inducida por la estocasticidad en la inicialización de los pesos de la red neuronal y en el proceso de optimización por gradientes de política, se implementa una evaluación estándar sobre tres réplicas pseudoaleatorias independientes (semillas 42, 100 y 2026). Al evaluarse cada réplica sobre el banco congelado de referencia de 300 episodios, los datos se consolidan bajo el identificador canónico del modelo conformando una muestra total de $N = 900$ episodios ($N = 600$ para configuraciones con dos réplicas). Dicha muestra acumulada se emplea directamente para calcular las proporciones binomiales de estabilización, los intervalos de Wilson al 95%, los cuartiles en los diagramas de caja de error y esfuerzo, y la frontera empírica de compromiso de Pareto.

### 4.3 Métricas Dinámicas y Temporales Complementarias
* **Tiempo de Establecimiento ($t_s$):** Instante continuo a partir del cual $\|\mathbf{x}_{ee}(t) - \mathbf{x}_{\text{tgt}}\| \le 0.02\text{ m}$ y permanece confinado dentro de dicha cota hasta el término del horizonte.
* **Esfuerzo Acumulado de Control ($J_u$):** Integral discreta del torque aplicado $J_u = \sum_{t=1}^T \|\mathbf{a}(t)\|^2 \Delta t_{\text{eff}}$.
* **Suavidad de Acción ($J_{\Delta u}$):** Variación cuadrática del comando entre pasos sucesivos $J_{\Delta u} = \frac{1}{T-1} \sum_{t=2}^T \|\mathbf{a}(t) - \mathbf{a}(t-1)\|$.

---

## 5. Algoritmos Evaluados y Baseline Analítico

### 5.1 Proximal Policy Optimization (`PPO_1`)
Algoritmo de gradiente de política *on-policy* con función objetivo truncada para limitar la divergencia de Kullback-Leibler. Utiliza estimación de ventajas generalizada (GAE) con $\gamma = 0.99$ y $\lambda_{\text{gae}} = 0.95$, redes densas totalmente conectadas de 3 capas ocultas ($[256, 256, 128]$) y normalización adaptativa de observaciones mediante `VecNormalize`.

### 5.2 Soft Actor-Critic (`SAC_1`)
Algoritmo *off-policy* basado en el marco de máxima entropía. Optimiza simultáneamente la recompensa esperada y la entropía de la política para evitar convergencias prematuras en mínimos locales de torque. Emplea doble red crítica Q, amortiguamiento suave de pesos de red objetivo con $\tau = 0.005$ y capas ocultas $[512, 512]$.

### 5.3 Baseline Analítico: Controlador Cartesiano PD con Transposición del Jacobiano (`C0`)
Como patrón de comparación analítico desacoplado de procesos de aprendizaje, se implementa un lazo de control operacional en coordenadas de tarea:
$$e_{\text{pos}} = \mathbf{x}_{\text{tgt}} - \mathbf{x}_{ee} \in \mathbb{R}^2$$
$$\tau = J(\boldsymbol{\theta})^T \left(K_p \, e_{\text{pos}}\right) - K_d \, \dot{\boldsymbol{\theta}}$$
$$\mathbf{a}_t = \text{clip}(\tau, -1.0, 1.0)$$
donde $K_p = 50.0$ actúa como ganancia cartesiana de posición y $K_d = 1.5$ proporciona amortiguamiento derivativo articular. La saturación en $[-1.0, 1.0]$ sitúa al controlador analítico bajo idénticas limitaciones físicas que los agentes de aprendizaje por refuerzo.
