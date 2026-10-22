# 🤖 Brazo Robótico 2 DOF: Benchmarking RL (PPO / SAC) vs. Control Cinemático Clásico (Fase D — MDP LQR)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-informational.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.9%2B-green.svg)](https://stable-baselines3.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-12%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Entorno de simulación, benchmarking cinemático y control de un brazo robótico planar de 2 Grados de Libertad (2 DOF) mediante **Aprendizaje por Refuerzo Profundo** (**PPO** y **SAC**, Stable-Baselines3) frente a un **Controlador Cinemático Diferencial Analítico (Jacobiano DLS `C0`)**.

El proyecto opera bajo el paradigma de la **Fase D (MDP Continuo con Regulación LQR)**, eliminando contadores temporales no observables (*hold* / POMDP) en favor de una formulación continua estrictamente markoviana.

### Cinemática Directa Exacta

$$\begin{aligned}
x_{codo} &= L_1 \cos(\theta_1), \quad y_{codo} = L_1 \sin(\theta_1) \\
x_{ee} &= x_{codo} + L_2 \cos(\theta_1 + \theta_2) \\
y_{ee} &= y_{codo} + L_2 \sin(\theta_1 + \theta_2)
\end{aligned}$$

### Matriz Jacobiana Analítica ($J(\boldsymbol{\theta}) \in \mathbb{R}^{2 \times 2}$)
La relación diferencial entre velocidades articulares $\dot{\boldsymbol{\theta}} = [\dot{\theta}_1, \dot{\theta}_2]^T$ y velocidad cartesiana del efector final $\dot{\mathbf{x}} = [\dot{x}, \dot{y}]^T$ está dada por:
$$J(\boldsymbol{\theta}) = \begin{bmatrix}
-L_1 \sin(\theta_1) - L_2 \sin(\theta_1 + \theta_2) & -L_2 \sin(\theta_1 + \theta_2) \\
L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2) & L_2 \cos(\theta_1 + \theta_2)
\end{bmatrix}$$

### Espacio de Estados y Acciones
* **Acciones continuas** $\mathbf{a}_t \in [-1.0, 1.0]^2$: Mapeadas directamente a velocidad angular articular:
  $$\dot{\boldsymbol{\theta}}_t = \mathbf{a}_t \cdot v_{max} \quad (v_{max} = 1.5\text{ rad/s}, \, \Delta t = 0.05\text{ s})$$
* **Observación** $\mathbf{s}_t \in \mathbb{R}^{10}$ (puramente cinemática):
  $$[\cos\theta_1, \, \sin\theta_1, \, \cos\theta_2, \, \sin\theta_2, \, \dot{\theta}_1, \, \dot{\theta}_2, \, x_{tgt}, \, y_{tgt}, \, \Delta x_{ee}, \, \Delta y_{ee}]$$

### Función de Recompensa de Fase D (`lyapunov_lqr`)
Recompensa continua cuadrática sin zonas de proximidad ni bonos terminales (`success_mode: none`):
$$r(s_t, a_t) = -w_{\text{pos}} \|\mathbf{x}_{ee} - \mathbf{x}_{tgt}\|^2 - w_{\text{vel}} \|\dot{\boldsymbol{\theta}}_t\|^2 - w_{\text{ctrl}} \|\mathbf{a}_t\|^2$$
con pesos canónicos por defecto $w_{\text{pos}} = 1.0$, $w_{\text{vel}} = 0.05$, $w_{\text{ctrl}} = 0.01$. Se admite como baseline secundario aislado `distance_only` ($r = -d$).

### Evaluación Dual Desacoplada
La evaluación se ejecuta sobre la trayectoria física cruda sin normalizaciones:
1. **Opción 2 (Métrica Principal — Estabilización en los últimos $k=20$ pasos):**
   $$\bar{d}_{20} = \frac{1}{20}\sum_{t=T-19}^T d_t \le 0.06\text{ m} \quad \wedge \quad \bar{v}_{20} = \frac{1}{20}\sum_{t=T-19}^T \|\dot{\boldsymbol{\theta}}_t\|_2 \le 0.10\text{ rad/s}$$
2. **Opción 3 (Métrica Comparativa Canónica — Costo Integral LQR):**
   $$J_{\text{LQR}} = -\sum_{t=1}^{T} \left(1.0 \cdot d_t^2 + 0.05 \cdot \|\dot{\boldsymbol{\theta}}_t\|_2^2 + 0.01 \cdot \|\mathbf{a}_t\|_2^2\right)$$
   Permite comparar agentes RL (`PPO_1`, `SAC_1`), variantes `distance_only` y el controlador clásico `C0` en una escala física objetiva e invariable.

Detalle metodológico completo en [`docs/metodologia.md`](docs/metodologia.md) y guía operativa en [`docs/manual-de-uso.md`](docs/manual-de-uso.md).

