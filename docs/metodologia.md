# 📐 Metodología — Brazo Robótico 2 DOF (Fase D: MDP Continuo con Regulación LQR)

Fundamento técnico del proyecto: modelo cinemático, transición desde POMDP (*hold*) hacia MDP continuo puro, formulación de recompensa cuadrática de regulación LQR y protocolo de evaluación dual desacoplada.
Para el uso operativo del código, consulte [`docs/manual-de-uso.md`](manual-de-uso.md).

---

## 1. Fundamento Físico y Cinemático

El robot es un manipulador planar articulado anclado en el origen $(0, 0)$ con dos eslabones rígidos de longitud $L_1 = 1.0\text{ m}$ y $L_2 = 1.0\text{ m}$.

```
      (x1, y1) Codo [Articulación 2]
         o───────┐ L2
        /         \
    L1 /           \
      /             o (x_ee, y_ee) Efector Final
     o (0, 0) Base
```

### 1.1 Cinemática Directa Exacta
$$\begin{aligned}
x_{codo} &= L_1 \cos(\theta_1), \quad y_{codo} = L_1 \sin(\theta_1) \\
x_{ee} &= x_{codo} + L_2 \cos(\theta_1 + \theta_2) \\
y_{ee} &= y_{codo} + L_2 \sin(\theta_1 + \theta_2)
\end{aligned}$$

### 1.2 Matriz Jacobiana Analítica ($J(\boldsymbol{\theta}) \in \mathbb{R}^{2 \times 2}$)
La relación diferencial entre velocidades articulares $\dot{\boldsymbol{\theta}} = [\dot{\theta}_1, \dot{\theta}_2]^T$ y velocidad cartesiana del efector final $\dot{\mathbf{x}} = [\dot{x}, \dot{y}]^T$ está dada por:
$$J(\boldsymbol{\theta}) = \begin{bmatrix}
-L_1 \sin(\theta_1) - L_2 \sin(\theta_1 + \theta_2) & -L_2 \sin(\theta_1 + \theta_2) \\
L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2) & L_2 \cos(\theta_1 + \theta_2)
\end{bmatrix}$$

### 1.3 Espacio de Estados y Acciones (10D Cinemático Puro)
* **Acciones continuas** $\mathbf{a}_t \in [-1.0, 1.0]^2$: mapeadas linealmente a velocidad angular articular:
  $$\dot{\boldsymbol{\theta}}_t = \mathbf{a}_t \cdot v_{max} \quad (v_{max} = 1.5\text{ rad/s}, \, \Delta t = 0.05\text{ s})$$
* **Vector de Observación** $\mathbf{s}_t \in \mathbb{R}^{10}$:
  $$[\cos\theta_1, \, \sin\theta_1, \, \cos\theta_2, \, \sin\theta_2, \, \dot{\theta}_1, \, \dot{\theta}_2, \, x_{tgt}, \, y_{tgt}, \, \Delta x_{ee}, \, \Delta y_{ee}]$$
  donde $\Delta x_{ee} = x_{tgt} - x_{ee}$ y $\Delta y_{ee} = y_{tgt} - y_{ee}$.

---

## 2. Transición hacia la Fase D: De POMDP a MDP Continuo Puro

### 2.1 El Problema del Paradigma Anterior (*Hold Condition* como POMDP)
En las Fases A, B y C del proyecto, el éxito se definía mediante un contador temporal acumulativo (*hold counter*): mantener $d \le 0.06\text{ m}$ y $\|\dot{\boldsymbol{\theta}}\| \le 0.10\text{ rad/s}$ durante $N=20$ pasos consecutivos para gatillar una terminación temprana con bono terminal $+15.0$.

Dicha formulación introducía severas patologías teóricas y empíricas:
1. **Parcial Observabilidad (POMDP):** El contador de pasos $N$ era una variable de estado oculta ajena al vector de observación de 10 dimensiones. Dos estados con idéntica pose cinemática $(\boldsymbol{\theta}, \dot{\boldsymbol{\theta}})$ tenían valores de función de valor $V(s)$ radicalmente diferentes dependiendo de si el contador interno estaba en $k=0$ o $k=19$.
2. **Explotación de Recompensa y Parálisis:** El corte terminal anticipado incentivaba estrategias subóptimas de llegada abrupta o parálisis en fronteras de zona.

### 2.2 Formulación Fase D: MDP Puro con Regulación LQR
La Fase D elimina por completo el concepto de corte por éxito temprano:
* **Horizonte Fijo:** Todos los episodios duran estrictamente $T_{max} = 200$ pasos ($10.0\text{ s}$).
* **Señal de Terminación:** `terminated = False` siempre; solo se emite `truncated = True` al alcanzar $T_{max}$.
* **Propiedad de Markov Restaurada:** El par $(\mathbf{s}_t, \mathbf{a}_t)$ determina de forma completa y determinista la distribución de transición y la recompensa instantánea.

---

## 3. Funciones de Recompensa

### 3.1 Recompensa Principal: `lyapunov_lqr`
La señal principal de aprendizaje modela una función cuadrática de costo de regulación óptima (LQR / Función de Lyapunov de control):
$$r(s_t, a_t) = -w_{\text{pos}} \|\mathbf{x}_{ee} - \mathbf{x}_{tgt}\|^2 - w_{\text{vel}} \|\dot{\boldsymbol{\theta}}_t\|^2 - w_{\text{ctrl}} \|\mathbf{a}_t\|^2$$
donde:
* $\|\mathbf{x}_{ee} - \mathbf{x}_{tgt}\|$ es la distancia euclidiana cartesiana al objetivo.
* $\|\dot{\boldsymbol{\theta}}_t\|_2 = \sqrt{\dot{\theta}_1^2 + \dot{\theta}_2^2}$ es la norma euclidiana de la velocidad articular (`joint_speed`).
* $\|\mathbf{a}_t\|_2^2 = a_1^2 + a_2^2$ es el esfuerzo de control instantáneo.
* Pesos canónicos: $w_{\text{pos}} = 1.0$, $w_{\text{vel}} = 0.05$, $w_{\text{ctrl}} = 0.01$.

Esta función proporciona un gradiente suave en todo el espacio de trabajo, penaliza el movimiento residual cerca del objetivo y evita la saturación de los actuadores sin requerir discontinuidades ni zonas no lineales.

### 3.2 Recompensa Secundaria Aislada: `distance_only`
Como control y baseline secundario comparativo, se conserva la recompensa puramente geométrica:
$$r(s_t, a_t) = -\|\mathbf{x}_{ee} - \mathbf{x}_{tgt}\|$$
con penalizaciones de velocidad y control fijadas estrictamente en $0.0$.

---

## 4. Protocolo de Evaluación Dual Desacoplada

Para evitar que la normalización de recompensas de `VecNormalize` contamine las métricas de desempeño, la evaluación se ejecuta siempre sobre la **trayectoria física cruda**:

### 4.1 Métrica Principal — Opción 2 (Estabilización en la Ventana Final $k=20$)
Examina el comportamiento asintótico en los últimos 20 pasos del episodio ($t \in [T-19, T]$):
$$\bar{d}_{20} = \frac{1}{20}\sum_{t=T-19}^T d_t, \qquad \bar{v}_{20} = \frac{1}{20}\sum_{t=T-19}^T \|\dot{\boldsymbol{\theta}}_t\|_2$$
$$\text{is\_success} = \mathbb{I}\left(\bar{d}_{20} \le 0.06\text{ m} \;\wedge\; \bar{v}_{20} \le 0.10\text{ rad/s}\right)$$

### 4.2 Métrica Comparativa — Opción 3 (Costo Acumulado Integral LQR Canónico)
Calcula el costo integral acumulado en cada paso a partir de la cinemática cruda usando pesos fijos universales:
$$J_{\text{LQR}} = -\sum_{t=1}^{T} \left(1.0 \cdot d_t^2 + 0.05 \cdot \|\dot{\boldsymbol{\theta}}_t\|_2^2 + 0.01 \cdot \|\mathbf{a}_t\|_2^2\right)$$
Esta métrica permite comparar rigurosamente en la misma escala física modelos entrenados con `lyapunov_lqr`, modelos entrenados con `distance_only`, y el controlador analítico `C0`.

---

## 5. Agentes y Benchmarking

* **`PPO_1`:** Agente PPO con política Gaussiana continua y ventaja GAE.
* **`SAC_1`:** Agente Soft Actor-Critic con exploración por máxima entropía y doble crítico Q.
* **`C0` (Jacobiano Amortiguado DLS):** Controlador cinemático analítico:
  $$\dot{\boldsymbol{\theta}}_{cmd} = J^T (J J^T + \lambda^2 I)^{-1} \cdot K_p (\mathbf{x}_{tgt} - \mathbf{x}_{ee})$$
  con $K_p = 3.0$ y amortiguamiento Levenberg-Marquardt $\lambda = 0.01$.

