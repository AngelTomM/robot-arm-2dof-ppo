# 📐 Metodología — Brazo Robótico 2 DOF

Fundamento técnico del proyecto: modelo físico, representación de estados, criterio operativo de éxito, diseño de funciones de recompensa y protocolo experimental (normalización, checkpoints y evaluación).
Para los análisis monográficos por fase, consulte los informes en [`reports/`](../reports/). Para el uso operativo del código, consulte [`docs/manual-de-uso.md`](manual-de-uso.md).

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

### 1.3 Espacio de Estados y Acciones
* **Acciones continuas** $\mathbf{a}_t \in [-1.0, 1.0]^2$: mapeadas directamente a velocidad angular articular:
  $$\dot{\boldsymbol{\theta}}_t = \mathbf{a}_t \cdot v_{max} \quad (v_{max} = 1.5\text{ rad/s}, \, \Delta t = 0.05\text{ s})$$
* **Observación** $\mathbf{s}_t \in \mathbb{R}^{10}$:
  $$[\cos\theta_1, \, \sin\theta_1, \, \cos\theta_2, \, \sin\theta_2, \, \dot{\theta}_1, \, \dot{\theta}_2, \, x_{tgt}, \, y_{tgt}, \, \Delta x_{ee}, \, \Delta y_{ee}]$$

---

## 2. Criterio Operativo de Éxito Estricto (*Hold*)

Para considerar un episodio como **exitoso**, el efector final debe satisfacer simultáneamente:
1. **Zona de Precisión:** distancia euclídea al blanco $d = \|\mathbf{x}_{ee} - \mathbf{x}_{tgt}\| \le r_{tgt} = 0.06\text{ m}$.
2. **Condición de Frenado:** velocidad angular conjunta $\|\dot{\boldsymbol{\theta}}\| \le v_{tol} = 0.10\text{ rad/s}$.
3. **Persistencia Temporal (*Hold*):** mantener las dos condiciones anteriores de forma ininterrumpida durante $N_{hold}$ pasos consecutivos ($N_{hold} = 20$ en los experimentos de referencia de Fase A; $N_{hold} = 5$ en Fase B y C, $v_{tol} = 0.3\text{ rad/s}$).
4. **Horizonte:** completar la estabilización antes del límite de $T_{max} = 200$ pasos ($10.0\text{ s}$).

Cualquier episodio que agote el horizonte sin consolidar el *hold* se clasifica como **truncado**. La transición desde el criterio clásico de mero contacto (*touch*, $d < 0.08\text{ m}$) se motivó porque dicha condición incentivaba atravesar el objetivo a máxima velocidad, generando etiquetas de éxito no representativas de estabilidad real.

---

## 3. Diseño de Funciones de Recompensa

La recompensa descompuesta en cada paso $t$ está dada por:
$$R_t = R_{dist} + R_{action} + R_{near} + R_{vel} + R_{smooth} + R_{success}$$

| Término | Expresión | Nota |
| :--- | :---: | :--- |
| Distancia ($R_{dist}$) | $-1.0 \cdot d$ | Gradiente continuo negativo hacia el objetivo. |
| Esfuerzo de control ($R_{action}$) | $-0.02 \|\mathbf{a}_t\|^2$ | Penalización por saturación de actuadores. |
| Proximidad ($R_{near}$) | $(1.0 - d / 0.30) - 0.15 (|\dot{\theta}_1| + |\dot{\theta}_2|)$ | Activo en $d < 0.30\text{ m}$; configurable mediante `r_near_mode` (`historical`, `none`, `non_positive`). |
| Velocidad gaussiana ($R_{vel}$, X5) | $-w_v \exp\!\left(-\dfrac{d^2}{2\sigma_v^2}\right) \|\dot{\boldsymbol{\theta}}\|^2$ | $w_v = 0.10$, $\sigma_v = 0.15$ en la configuración de referencia. |
| Suavidad ($R_{smooth}$, X6) | $-w_a \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$ | $w_a = 0.01$ en X6; barrido $w_a \in \{0.01, 0.1, 0.5\}$ en X6_wa*. |
| Éxito ($R_{success}$) | $+15.0 \cdot \mathbb{I}_{succ}$ | Otorgado únicamente al consolidar el *hold*. |

### 3.1 Diagnóstico Empírico de los Términos

* **Efecto de $R_{near}$:** en la Fase A se hipotetizó que el término positivo $(1 - d/0.30)$ era perjudicial ("trampa de recompensa"). La evaluación posterior a igual presupuesto demostró lo contrario: suprimirlo (`X5_near`) redujo la tasa de éxito en $14.93$ puntos porcentuales, por lo que actúa como un *reward shaping* denso e indispensable para la convergencia temprana.
* **Valores elevados de $w_v$:** imponer $w_v \ge 1.0$ produce **parálisis cinemática**, pues el coste de aproximarse con velocidad supera la ganancia de proximidad. El valor óptimo en el barrido fue $w_v = 0.10$.
* **Recompensa `smooth_precision`:** combinación de referencia de las fases B y C (sin barrido adicional); es la base del agente líder.

---

## 4. Protocolo Experimental

### 4.1 Fases del proyecto

| Fase | Algoritmo | Presupuesto | Objetivo |
| :--- | :---: | :---: | :--- |
| **A** | PPO | 300.000 pasos | Diagnóstico: estabilización del criterio *hold*, arquitectura, normalización y barridos de recompensa (X1a–X6_wa05). |
| **B** | PPO | 1.000.000 de pasos | Confirmación a presupuesto extendido de los candidatos (X3, X5_near). |
| **B (extensión)** | PPO | 655.360 pasos | Evaluación multi-semilla ($n=500$ episodios × 3 semillas) y selección formal del agente líder (X4). |
| **C** | SAC | 245.760–655.360 pasos | Estudio unifactorial (frecuencia de entrenamiento, suavidad) y paquete combinado (S1, S2_freq, S2_wa, S3_combined). |

### 4.2 Normalización y política de checkpoints

* Se demostró que desacoplar la red neuronal de las estadísticas de `VecNormalize` distorsiona gravemente la evaluación: las observaciones normalizadas con estadísticas obsoletas dejan de ser comparables entre checkpoints.
* Toda la pipeline vigente aplica **guardado acoplado**: en cada checkpoint se persisten simultáneamente `model.zip` y `vecnormalize_stats.pkl` del mismo paso de entrenamiento, de modo que la recarga de cualquier punto de la curva de aprendizaje reproduce con exactitud el estado normalizado correspondiente.
* El entorno de evaluación se ejecuta en modo inferencia (`training=False`, `norm_reward=False`) y recibe las estadísticas sincronizadas desde el entorno de entrenamiento (`sync_envs_normalization`).

### 4.3 Congelamiento de datasets y evaluación

* Los datasets de evaluación ($n=200$, $n=300$, $n=500$) se generan una única vez con semillas documentadas y se fijan con hashes SHA-256; el separador por conjunto (validación / evaluación final) se verifica mediante test para impedir solapamientos y fuga de información.
* La métrica principal es la **tasa de éxito estricta** con intervalo de Wilson al 95%; la comparación entre agentes se resuelve con el **intervalo de indistinguibilidad operacional** ($U_{equiv} = 3.16$ pp), calculado por el script de selección formal.
* La evaluación se ejecuta en dos modos separados: determinista (acción media de la política) y estocástico (muestreo), sin mezcla de modos dentro de una misma ejecución.
* La selección del *best_model* durante el entrenamiento se basa en la tasa de éxito de la evaluación en línea ($n_{eval}=15$ episodios por defecto), que utiliza objetivos aleatorios y no el dataset congelado final, garantizando la independencia entre selección de checkpoint y evaluación terminal.

### 4.4 Reproducibilidad y configuración

* Cada experimento deriva su fase e identificador de la **ruta del archivo YAML** (fuente única de verdad) y escribe la salida en `experiments/<fase>/<exp_id>/seed_<seed>/`.
* El archivo `experiment_meta.json` registra el config completo y parámetros efectivos de la ejecución; `train_metrics.json` registra el tiempo y el rendimiento medio.
* Las semillas se fijan de forma global (Python, NumPy, torch y CUDA) para reproducibilidad determinista.

---

## 5. Referencias a Monografías

* Fase A: [`reports/phase_a_report.md`](../reports/phase_a_report.md)
* Fase B y Extensión: [`reports/phase_b_report.md`](../reports/phase_b_report.md)
* Fase C: [`reports/phase_c_report.md`](../reports/phase_c_report.md)
