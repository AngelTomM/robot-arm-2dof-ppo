# Manipulador Robótico Planar 2-DOF: Benchmarking RL (PPO / SAC) vs. Control Analítico Clásico (MuJoCo Reacher-v5)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.17%2B-black.svg)](https://mujoco.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-informational.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.9%2B-green.svg)](https://stable-baselines3.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-14%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Entorno de simulación física, entrenamiento por refuerzo y evaluación comparativa de un manipulador robótico planar de 2 grados de libertad (2-DOF) accionado por torque. El sistema implementa algoritmos de aprendizaje por refuerzo profundo (**PPO** y **SAC** mediante Stable-Baselines3) y un controlador analítico operacional de referencia (**Controlador Cartesiano PD con Transposición del Jacobiano `C0`**).

La formulación corresponde a la **Fase D (Proceso de Decisión de Markov Continuo con Regulación LQR)**, descartando contadores temporales no observables (*hold counter* / POMDP) en favor de una formulación markoviana continua de horizonte temporal fijo ($T = 4.0\text{ s}$, 200 pasos de control).

---

## 1. Fundamentos Físicos y Cinemáticos

El manipulador opera sobre la planta física dinámica `Reacher-v5` de Gymnasium ejecutada en el motor MuJoCo. La base se encuentra fija en el origen $(0, 0, 0.01)\text{ m}$ y los eslabones cilíndricos rígidos poseen longitudes $L_1 = 0.10\text{ m}$ y $L_2 = 0.11\text{ m}$ (incluyendo el radio del efector final *fingertip*), alcanzando un radio máximo de $0.21\text{ m}$.

### Cinemática Directa Planar
$$x_{\text{codo}} = L_1 \cos(\theta_1), \quad y_{\text{codo}} = L_1 \sin(\theta_1)$$
$$x_{ee} = x_{\text{codo}} + L_2 \cos(\theta_1 + \theta_2)$$
$$y_{ee} = y_{\text{codo}} + L_2 \sin(\theta_1 + \theta_2)$$

### Matriz Jacobiana Analítica ($J(\boldsymbol{\theta}) \in \mathbb{R}^{2 \times 2}$)
La relación diferencial entre las velocidades articulares $\dot{\boldsymbol{\theta}} = [\dot{\theta}_1, \dot{\theta}_2]^T$ y la velocidad cartesiana del efector final $\dot{\mathbf{x}}_{ee} = [\dot{x}_{ee}, \dot{y}_{ee}]^T$ se define como:
$$J(\boldsymbol{\theta}) = \begin{bmatrix}
-L_1 \sin(\theta_1) - L_2 \sin(\theta_1 + \theta_2) & -L_2 \sin(\theta_1 + \theta_2) \\
L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2) & L_2 \cos(\theta_1 + \theta_2)
\end{bmatrix}$$

### Espacio de Estados y Acciones
* **Acciones continuas** $\mathbf{a}_t \in [-1.0, 1.0]^2$: Torques articulares continuos aplicados en las articulaciones rotatorias (con factor de transmisión $\text{gear} = 200.0\text{ N}\cdot\text{m}$).
* **Vector de Observación** $\mathbf{s}_t \in \mathbb{R}^{10}$: Vector continuo normalizado emitido por MuJoCo:
  $$[\cos\theta_1, \, \cos\theta_2, \, \sin\theta_1, \, \sin\theta_2, \, x_{\text{tgt}}, \, y_{\text{tgt}}, \, \dot{\theta}_1, \, \dot{\theta}_2, \, (x_{ee} - x_{\text{tgt}}), \, (y_{ee} - y_{\text{tgt}})]$$

---

## 2. Formulación de Recompensa Cuadrática (LQR / Lyapunov)

La función de recompensa en cada paso de tiempo $\Delta t_{\text{eff}} = 0.02\text{ s}$ penaliza simultáneamente el error de posicionamiento cartesiano, el esfuerzo de torque aplicado y la energía cinética articular:
$$r_t = - \left(w_{\text{pos}} \|\mathbf{x}_{ee} - \mathbf{x}_{\text{tgt}}\|^2 + w_{\text{ctrl}} \|\mathbf{a}_t\|^2 + w_{\text{vel}} \|\dot{\boldsymbol{\theta}}_t\|^2\right)$$
donde los coeficientes invariantes de ponderación son $w_{\text{pos}} = 1.0$, $w_{\text{ctrl}} = 0.01$ y $w_{\text{vel}} = 0.05$.

---

## 3. Protocolo de Evaluación Dual Desacoplada

La evaluación comparativa de las políticas y controladores se realiza sobre trayectorias físicas crudas sin normalizar a lo largo de conjuntos deterministas congelados:
1. **Métrica Principal — Estabilización en la Ventana Terminal ($k = 20$ pasos finales):**
   $$\bar{e}_{\text{ss}} = \frac{1}{20}\sum_{t=T-19}^T \|\mathbf{x}_{ee}(t) - \mathbf{x}_{\text{tgt}}\| \le 0.02\text{ m} \quad \wedge \quad \bar{v}_{\text{ss}} = \frac{1}{20}\sum_{t=T-19}^T \|\dot{\boldsymbol{\theta}}(t)\| \le 0.10\text{ rad/s}$$
   El éxito se cuantifica mediante la proporción binomial con intervalos de confianza de Wilson al 95%.
2. **Métrica Comparativa — Costo Integral LQR Canónico ($J_{\text{LQR}}$):**
   $$J_{\text{LQR}} = -\sum_{t=1}^T \left(1.0 \cdot \|\mathbf{x}_{ee}(t) - \mathbf{x}_{\text{tgt}}\|^2 + 0.01 \cdot \|\mathbf{a}(t)\|^2 + 0.05 \cdot \|\dot{\boldsymbol{\theta}}(t)\|^2\right)$$

---

## 4. Arquitectura del Repositorio

La organización del proyecto respeta estrictamente el principio de Fuente Única de Verdad (SSOT) y desacopla la física del sistema respecto a los hiperparámetros de los agentes:

```text
Fase_D_V3/
├── configs/
│   ├── environment.yaml                     # SSOT física, cinemática y función base MDP
│   └── phase_D/
│       ├── prueba/                          # Configuraciones de agentes (PPO_1, SAC_1, C0)
│       └── smoke/                           # Configuraciones de pruebas rápidas
├── data/
│   └── datasets/                            # Conjuntos congelados de validación (200 ep) y final (300 ep)
├── docs/                                    # Manual de uso, metodología y auditoría
├── reports/
│   ├── data/evaluation/                     # Archivos CSV y JSON con telemetría estructurada
│   └── figures/                             # Figuras comparativas generadas
├── src/                                     # Paquete único sin subcarpetas unitarias
│   ├── __init__.py                          # Exportación unificada de la API pública
│   ├── arm2d_env.py                         # Entorno Gymnasium adaptado sobre MuJoCo Reacher-v5
│   ├── callbacks.py                         # Callbacks de telemetría y sincronización VecNormalize
│   ├── config.py                            # Gestor desacoplado de configuraciones YAML
│   ├── datasets.py                          # Generador y cargador de episodios deterministas
│   ├── evaluator.py                         # Motor de evaluación agnóstico sobre datasets
│   ├── jacobian_controller.py               # Controlador cartesiano PD con transposición del Jacobiano
│   └── metrics.py                           # Métricas mecatrónicas e intervalos de Wilson
├── tests/
│   ├── test_config.py                       # Verificación de desacoplamiento y parámetros SSOT
│   ├── test_kinematics.py                   # Validación analítica contrastada con MuJoCo
│   └── test_normalization.py                # Wrapper VecNormalize y cálculo de recompensas
├── evaluate.py                              # Punto de entrada para evaluación de políticas y baseline
├── interactive_render.py                    # Visualizador interactivo 3D OpenGL en MuJoCo
├── plot.py                                  # Generación desacoplada de figuras mecatrónicas
├── train.py                                 # Punto de entrada unificado para entrenamiento PPO/SAC
├── requirements.txt                         # Dependencias fijadas del proyecto
└── README.md
```

---

## 5. Instrucciones Rápidas de Uso

```powershell
# 1. Activar el entorno virtual
..\.venv\Scripts\Activate.ps1

# 2. Ejecutar entrenamiento continuo
python train.py --config configs/phase_D/prueba/PPO_1.yaml --seed 42

# 3. Evaluar el baseline analítico C0 sobre el dataset de validación
python evaluate.py --c0 --dataset data/datasets/validation_set_200.json

# 4. Generar curvas y figuras comparativas
python plot.py

# 5. Ejecutar la suite completa de pruebas unitarias
python -m pytest tests/
```

Para consultar los fundamentos detallados, refiérase a [docs/metodologia.md](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/docs/metodologia.md) y a la guía de operación en [docs/manual-de-uso.md](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/docs/manual-de-uso.md).
