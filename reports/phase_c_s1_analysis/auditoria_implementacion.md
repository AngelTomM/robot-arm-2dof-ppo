# Auditoría de Implementación — SAC S1 (Fase C)

**Proyecto:** `robot-arm-2dof-ppo`  
**Entorno auditado:** [Arm2DEnv](arm2d_env.py)  
**Evaluador auditado:** [evaluator.py](src/evaluation/evaluator.py)  
**Configuración de entrenamiento:** [S1.yaml](configs/phase_c/S1.yaml)  
**Configuración de evaluación:** [thresholds.yaml](configs/phase_a/thresholds.yaml)  
**Dataset evaluado:** `data/datasets/phase_c/validation_c_500.json` (500 episodios)  
**Estado:** Solo lectura — Inspección de código y verificación empírica sin alteraciones.

---

## 1. Resumen de la Auditoría

Se auditó de manera exhaustiva el código fuente del entorno de simulación, la lógica de cálculo de recompensas, el protocolo de evaluación estricta y la carga de normalizaciones.

El objetivo fue verificar la coherencia entre el entrenamiento de SAC S1 y la evaluación estricta, así como esclarecer los mecanismos matemáticos y dinámicos que gobiernan el comportamiento del brazo robótico.

---

## 2. Inspección Detallada de Componentes

### 2.1. Representación de la Observación (10 dimensiones)
- **Código:** [arm2d_env.py:L142-L158](arm2d_env.py#L142-L158) en `Arm2DEnv._get_obs()`
- **Composición:**
  $$\mathbf{s}_t = \begin{bmatrix} \cos(\theta_1), & \sin(\theta_1), & \cos(\theta_2), & \sin(\theta_2), & \dot{\theta}_1, & \dot{\theta}_2, & x_{\text{tgt}}, & y_{\text{tgt}}, & \Delta x, & \Delta y \end{bmatrix}^\top$$
  donde $\Delta x = x_{\text{tgt}} - x_{\text{ee}}$ y $\Delta y = y_{\text{tgt}} - y_{\text{ee}}$.
- **[HECHO]:** La observación no incluye la acción previa $\mathbf{a}_{t-1}$. La política aprende un mapeo reactivo Markoviano estrictamente dependiente del estado cinemático instantáneo y la posición del objetivo.
- **[HECHO]:** Existe redundancia cinemática entre $(x_{\text{tgt}}, y_{\text{tgt}})$ y $(\Delta x, \Delta y)$ junto con las posiciones articulares, lo cual facilita el cálculo directo del gradiente de distancia hacia el objetivo para el actor.

### 2.2. Interpretación y Escalado de Acciones
- **Código:** [arm2d_env.py:L234-L236](arm2d_env.py#L234-L236) en `Arm2DEnv.step()`
- **Interpretación:**
  La acción continua de la red $\mathbf{a}_t \in [-1, 1]^2$ comanda **directamente** la velocidad angular articular:
  $$\dot{\theta}_1 = a_1 \cdot v_{\max}, \quad \dot{\theta}_2 = a_2 \cdot v_{\max}$$
  con $v_{\max} = 1{,}5\text{ rad/s}$.
- **[HECHO]:** El sistema **no modela dinámica inercial de segundo orden** (no hay pares de torsión, masas ni aceleraciones calculadas dinámicamente). La velocidad articular cambia instantáneamente sin retardo inercial. En consecuencia, si la política alterna entre $-1$ y $+1$ en pasos contiguos, la velocidad articular pasa de $-1{,}5\text{ rad/s}$ a $+1{,}5\text{ rad/s}$ en un solo intervalo $\Delta t = 0{,}05\text{ s}$ ($\ddot{\theta} = 60\text{ rad/s}^2$).

### 2.3. Integración de Posiciones Articulares
- **Código:** [arm2d_env.py:L239-L242](arm2d_env.py#L239-L242)
- **Integración Euler:**
  $$\theta_i \leftarrow (\theta_i + \dot{\theta}_i \cdot \Delta t + \pi) \pmod{2\pi} - \pi, \quad i \in \{1, 2\}$$
- **[HECHO]:** Los ángulos permanecen acotados en $[-\pi, \pi)$ mediante mapeo modular idéntico al código histórico.

### 2.4. Cinemática Directa y Distancia
- **Código:** [arm2d_env.py:L126-L132](arm2d_env.py#L126-L132) en `_forward_kinematics()` y [arm2d_env.py:L245-L248](arm2d_env.py#L245-L248)
- **Ecuaciones:**
  $$x_1 = L_1 \cos(\theta_1), \quad y_1 = L_1 \sin(\theta_1)$$
  $$x_{\text{ee}} = x_1 + L_2 \cos(\theta_1 + \theta_2), \quad y_{\text{ee}} = y_1 + L_2 \sin(\theta_1 + \theta_2)$$
  $$d(\mathbf{p}_{\text{ee}}, \mathbf{p}_{\text{tgt}}) = \sqrt{(x_{\text{tgt}} - x_{\text{ee}})^2 + (y_{\text{tgt}} - y_{\text{ee}})^2}$$
- **[HECHO]:** $L_1 = L_2 = 1{,}0\text{ m}$. El alcance geométrico máximo es $R_{\max} = 2{,}0\text{ m}$.

### 2.5. Cálculo de Velocidad Articular
- **Código:** [arm2d_env.py:L250](arm2d_env.py#L250)
- **Definición:**
  $$v_{\text{joint}} = \|\dot{\boldsymbol{\theta}}\|_2 = \sqrt{\dot{\theta}_1^2 + \dot{\theta}_2^2} = v_{\max} \sqrt{a_1^2 + a_2^2}$$
- **[HECHO]:** Como la velocidad es directamente proporcional a la norma de la acción, cumplir la condición de baja velocidad $v_{\text{joint}} < 0{,}10\text{ rad/s}$ exige imperativamente que:
  $$\|\mathbf{a}_t\|_2 < \frac{0{,}10}{1{,}5} \approx 0{,}0667$$
  Es decir, cada componente de la acción debe estar prácticamente en reposo ($|a_i| \le 0{,}047$).

### 2.6. Cálculo de Recompensa y Sus Componentes
- **Código:** [arm2d_env.py:L307-L343](arm2d_env.py#L307-L343)
- **Componentes en modo `smooth_precision` (S1):**
  $$r_t = r_{\text{dist}} + r_{\text{action}} + r_{\text{near}} + r_{\text{vel}} + r_{\text{smooth}} + r_{\text{success}}$$
  1. Distancia: $r_{\text{dist}} = -1{,}0 \cdot d_t$.
  2. Acción: $r_{\text{action}} = -0{,}02 \cdot \|\mathbf{a}_t\|_2^2$.
  3. Zona cercana ($d_t < 0{,}30\text{ m}$):
     $$r_{\text{near}} = \left(1{,}0 - \frac{d_t}{0{,}30}\right) - 0{,}15 \cdot (|\dot{\theta}_1| + |\dot{\theta}_2|)$$
  4. Penalización gaussiana de velocidad: $r_{\text{vel}} = 0$ (ya que $w_v = 0$ en S1).
  5. Penalización de suavidad: $r_{\text{smooth}} = 0$ (ya que $w_a = 0$ en S1).
  6. Bonificación de éxito: $r_{\text{success}} = +15{,}0$ otorgada únicamente cuando `is_success == True`.

### 2.7. Condiciones de Éxito y Permanencia
- **Código:** [arm2d_env.py:L256-L269](arm2d_env.py#L256-L269)
- **Lógica de hold:**
  $$\text{inside\_target} = (d_t < r_{\text{tgt}})$$
  $$\text{slow\_enough} = (v_{\text{joint}} < v_{\text{tol}})$$
  - Si $\text{inside\_target} \land \text{slow\_enough}$:
    $$\text{hold\_counter} \leftarrow \text{hold\_counter} + 1$$
  - En caso contrario:
    $$\text{hold\_counter} \leftarrow 0$$
- **Éxito:**
  $$\text{is\_success} = (\text{hold\_counter} \ge N_{\text{hold}})$$
- **[HECHO - DISCREPANCIA CRÍTICA ENTRE ENTRENAMIENTO Y EVALUACIÓN]:**
  - **En Entrenamiento S1 ([S1.yaml:L10-L15](configs/phase_c/S1.yaml#L10-L15)):**
    $r_{\text{tgt}} = 0{,}06\text{ m}$, $v_{\text{tol}} = 0{,}30\text{ rad/s}$, $N_{\text{hold}} = 5\text{ pasos}$ ($0{,}25\text{ s}$).
  - **En Evaluación Estricta ([thresholds.yaml:L15-L21](configs/phase_a/thresholds.yaml#L15-L21)):**
    $r_{\text{tgt}} = 0{,}06\text{ m}$, $v_{\text{tol}} = 0{,}10\text{ rad/s}$, $N_{\text{hold}} = 20\text{ pasos}$ ($1{,}00\text{ s}$).

### 2.8. Terminación de Episodios
- **Código:** [arm2d_env.py:L348-L350](arm2d_env.py#L348-L350)
  $$\text{terminated} = \text{is\_success}$$
  $$\text{truncated} = (\text{step\_count} \ge 200 \land \neg \text{terminated})$$
- **[HECHO]:** No existen límites físicos angulares ni colisiones que fuercen `terminated` anticipadamente; el único motivo de `terminated` es alcanzar el éxito.

### 2.9. Normalización con VecNormalize
- **Código:** [evaluator.py:L46-L52](src/evaluation/evaluator.py#L46-L52)
- **Implementación:**
  Durante la evaluación, `VecNormalize.load(...)` carga las estadísticas acumuladas (`mean`, `var`) asociadas a cada checkpoint. Se desactiva la actualización (`training = False`) y la normalización de recompensas (`norm_reward = False`).
- **[HECHO]:** Cada checkpoint periódico de SAC S1 cuenta con su archivo correspondiente `vecnormalize_stats.pkl`. La carga congelada es reproducible y matemáticamente exacta.

---

## 3. Matriz Sintética de la Auditoría

| Elemento | Comportamiento Observado en el Código | Archivo y Función | Relevancia para los Resultados | Estado |
| :--- | :--- | :--- | :--- | :--- |
| **Observación (10D)** | $\cos,\sin$ articulares, velocidades, objetivo y vector error relativo. No incluye $\mathbf{a}_{t-1}$. | [arm2d_env.py:L142-L158](arm2d_env.py#L142-L158) | Facilita la reactividad pero impide a la red regular su propia aceleración $\Delta \mathbf{a}$. | **Verificado** |
| **Acción Cinemática** | Comanda directamente velocidad $\dot{\theta} = v_{\max} \mathbf{a}$. Sin inercia. | [arm2d_env.py:L234-L236](arm2d_env.py#L234-L236) | Permite conmutaciones bang-bang de velocidad instantáneas a alta frecuencia. | **Verificado** |
| **Tolerancia Velocidad Entrenamiento** | $v_{\text{tol}} = 0{,}30\text{ rad/s}$. | [S1.yaml:L15](configs/phase_c/S1.yaml#L15) | Permite frenado parcial $\|\mathbf{a}\| < 0{,}20$. | **Verificado** |
| **Tolerancia Velocidad Evaluación** | $v_{\text{tol}} = 0{,}10\text{ rad/s}$. | [thresholds.yaml:L17](configs/phase_a/thresholds.yaml#L17) | Requiere frenado casi nulo $\|\mathbf{a}\| < 0{,}067$. | **Discrepancia Entrenamiento/Evaluación** |
| **Permanencia Entrenamiento** | $N_{\text{hold}} = 5\text{ pasos}$ ($0{,}25\text{ s}$). | [S1.yaml:L14](configs/phase_c/S1.yaml#L14) | Facilita la obtención del bonus $+15$ en trayectorias de paso rápido. | **Verificado** |
| **Permanencia Evaluación** | $N_{\text{hold}} = 20\text{ pasos}$ ($1{,}00\text{ s}$). | [thresholds.yaml:L18](configs/phase_a/thresholds.yaml#L18) | Penaliza cualquier fluctuación o jitter durante 1 segundo entero. | **Discrepancia Entrenamiento/Evaluación** |
| **Penalización Suavidad** | $w_a = 0{,}0$ en S1 (`reward_smooth = 0`). | [arm2d_env.py:L335-L337](arm2d_env.py#L335-L337) | No existe coste en la recompensa por oscilaciones $\|\Delta \mathbf{a}\|$. | **Verificado** |
| **Penalización Velocidad Gaussiana** | $w_v = 0{,}0$ en S1 (`reward_vel = 0`). | [arm2d_env.py:L330-L333](arm2d_env.py#L330-L333) | No hay guía densa para frenar suavemente al acercarse. | **Verificado** |
| **Término Positivo $r_{\text{near}}$** | $(1 - d/0{,}30) > 0$ si $d < 0{,}30$. | [arm2d_env.py:L324](arm2d_env.py#L324) | Podría incentivar orbitar cerca de la meta sin estabilizarse. | **Verificado** |
| **Reinicio de Hold** | Resetea inmediatamente a 0 si $d \ge 0{,}06$ o $v \ge v_{\text{tol}}$. | [arm2d_env.py:L260-L264](arm2d_env.py#L260-L264) | Exige estabilidad ininterrumpida; 1 paso de vibración rompe la serie. | **Verificado** |
| **Carga de VecNormalize** | Se carga el archivo del checkpoint con `training=False`. | [evaluator.py:L46-L52](src/evaluation/evaluator.py#L46-L52) | Evaluación estadísticamente limpia, sin fuga ni descalibración. | **Verificado** |
