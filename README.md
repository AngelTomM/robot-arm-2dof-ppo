# 🤖 Brazo Robótico 2 DOF: Benchmarking por Aprendizaje por Refuerzo (PPO / SAC) vs. Control Cinemático Clásico

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-informational.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.9%2B-green.svg)](https://stable-baselines3.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-31%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


Entorno de simulación, benchmarking cinemático y control de un brazo robótico planar de 2 Grados de Libertad (2 DOF) mediante **Aprendizaje por Refuerzo Profundo** (**PPO** y **SAC**, Stable-Baselines3) frente a un **Controlador Cinemático Diferencial Analítico (Jacobiano DLS `C0`)**.

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
* **Observación** $\mathbf{s}_t \in \mathbb{R}^{10}$:
  $$[\cos\theta_1, \, \sin\theta_1, \, \cos\theta_2, \, \sin\theta_2, \, \dot{\theta}_1, \, \dot{\theta}_2, \, x_{tgt}, \, y_{tgt}, \, \Delta x_{ee}, \, \Delta y_{ee}]$$

*Intervalo de Indistinguibilidad Operacional:* $U_{equiv} = 3.16$ pp. Los modelos `X4`, `X3` y `X5` forman el grupo estadísticamente indistinguible de máximo desempeño. Detalle metodológico completo en [`docs/metodologia.md`](docs/metodologia.md).
