# 🤖 Brazo Robótico 2 DOF: Benchmarking y Control por PPO vs. Control Cinemático Clásico

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-informational.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.5%2B-green.svg)](https://stable-baselines3.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Entorno de simulación, benchmarking cinemático y control de un brazo robótico planar de 2 Grados de Libertad (2 DOF) mediante **Aprendizaje por Refuerzo Profundo (PPO)** frente a un **Controlador Cinemático Diferencial Analítico (Jacobiano DLS `C0`)**. 

El proyecto consolida la investigación y diagnósticos de la **Fase A** (estabilización rigurosa bajo criterio de *hold* y resolución de patologías de recompensa), la **Fase B** (presupuesto extendido a 1.000.000 de pasos con checkpointing atómico) y la **Extensión de Fase B** (evaluación multi-semilla a gran escala y selección del agente líder **`X4`**).

---

## 📌 Tabla de Contenidos

- [Visión General y Hallazgos Clave](#-visión-general-y-hallazgos-clave)
- [Fundamento Físico y Cinemático](#-fundamento-físico-y-cinemático)
- [Criterio Operativo de Éxito Estricto (*Hold*)](#-criterio-operativo-de-éxito-estricto-hold)
- [Ranking y Resultados Consolidados](#-ranking-y-resultados-consolidados)
- [Diseño de Funciones de Recompensa y Diagnósticos](#-diseño-de-funciones-de-recompensa-y-diagnósticos)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Guía de Uso y Reproducibilidad](#-guía-de-uso-y-reproducibilidad)
  - [1. Simulación Interactiva en Tiempo Real](#1-simulación-interactiva-en-tiempo-real)
  - [2. Entrenamiento de Agentes](#2-entrenamiento-de-agentes)
  - [3. Evaluación y Benchmarking Formal](#3-evaluación-y-benchmarking-formal)
  - [4. Ejecución de Tests Unitarios](#4-ejecución-de-tests-unitarios)
- [Reportes e Informes Monográficos](#-reportes-e-informes-monográficos)
- [Licencia](#-licencia)

---

## 🚀 Visión General y Hallazgos Clave

1. **Agente Líder Seleccionado (`X4`):**
   - Arquitectura MLP $256 \times 256$ ReLU, `VecNormalize(norm_obs=True)`, recompensa `smooth_precision` y `batch_size = 256`.
   - Alcanzó **96.73% de éxito estricto determinista** ($n=500$ episodios por semilla, 3 semillas independientes), con una mediana de **59.7 pasos** al éxito y una suavidad de control $\Delta a = 0.046$.
   - Comparte el grupo de equivalencia estadística superior con **`X3`** (94.53%–98.00%) y **`X5`** (93.93%).
2. **Cota Superior Analítica (`C0`):**
   - Controlador cinemático diferencial analítico basado en la pseudo-inversa amortiguada del Jacobiano (DLS, $K_p = 3.0, \lambda = 0.01$).
   - Alcanza **100.0% de éxito estricto**, 0.0% de truncamientos y 0.0% de oscilaciones con una distancia final media de $0.0028\text{ m}$.
3. **Superación del Modo *Touch* hacia el Modo *Hold*:**
   - La condición clásica de mero contacto ($d < 0.08\text{ m}$) incentivaba atravesar el objetivo a máxima velocidad (llegadas balísticas destructivas).
   - La implementación del estándar de estabilización estricta (*hold* durante 20 pasos con $\|\dot{\boldsymbol{\theta}}\| \le 0.10\text{ rad/s}$) garantizó estabilidad cinemática real.
4. **Normalización y Checkpointing Atómico:**
   - Se demostró que desacoplar las redes neuronales de las estadísticas de `VecNormalize` distorsionaba gravemente las evaluaciones históricas.
   - Todo el pipeline actual utiliza guardado acoplado (`model.zip` + `vecnormalize_stats.pkl`) en cada checkpoint.

---

## 📐 Fundamento Físico y Cinemático

El robot es un manipulador planar articulado anclado en el origen $(0, 0)$ con dos eslabones rígidos de longitud $L_1 = 1.0\text{ m}$ y $L_2 = 1.0\text{ m}$.

```
      (x1, y1) Codo [Articulación 2]
         o───────┐ L2
        /         \
    L1 /           \
      /             o (x_ee, y_ee) Efector Final
     o (0, 0) Base
```

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

---

## 🎯 Criterio Operativo de Éxito Estricto (*Hold*)

Para considerar un episodio como **exitoso**, el efector final debe satisfacer simultáneamente:
1. **Zona de Precisión:** Distancia euclídea al blanco $d = \|\mathbf{x}_{ee} - \mathbf{x}_{tgt}\| \le r_{tgt} = 0.06\text{ m}$.
2. **Condición de Frenado:** Velocidad angular conjunta total $\|\dot{\boldsymbol{\theta}}\| \le v_{tol} = 0.10\text{ rad/s}$.
3. **Persistencia Temporal (*Hold*):** Mantener las dos condiciones anteriores de forma ininterrumpida durante **$N_{hold} = 20$ pasos consecutivos** ($1.0\text{ segundo}$ físico).
4. **Horizonte:** Completar la estabilización antes del límite de $T_{max} = 200$ pasos ($10.0\text{ s}$).

Cualquier episodio que agote los 200 pasos sin completar el *hold* es clasificado como **Truncado**.

---

## 🏆 Ranking y Resultados Consolidados

Resultados consolidados de la **Extensión de Fase B** ($n=500$ episodios por semilla, 3 semillas independientes: `42`, `100`, `2026`) y evaluación asintótica a 1M pasos:

| Agente | Arquitectura / Normalización | Recompensa | Batch Size | Éxito Estricto Determ. | Mediana Pasos | Suavidad $\Delta a$ | Veredicto Operativo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`C0`** | Analítico (Jacobiano DLS) | N/A | N/A | **100.0%** | **59.0** | **0.000** | **Cota Superior Analítica** |
| **`X4`** | $256 \times 256$ ReLU + `VecNorm` | `smooth_precision` | 256 | **96.73%** | **59.7** | **0.046** | 🥇 **Agente Líder Ganador** |
| **`X3`** | $256 \times 256$ ReLU + `VecNorm` | `smooth_precision` | 64 | **94.53%** *(98.0% @1M)* | **57.7** | **0.042** | 🥈 **Grupo Equivalente Líder** |
| **`X5`** | $256 \times 256$ ReLU + `VecNorm` | + Gaussiana $w_v=0.10$ | 64 | **93.93%** | **60.0** | **0.039** | 🥉 **Grupo Equivalente Líder** |
| **`X6`** | $256 \times 256$ ReLU + `VecNorm` | + Suavidad $w_a=0.01$ | 64 | **92.67%** | **59.3** | **0.039** | Descarte ($\Delta > U_{equiv}$) |
| **`X5_near`** | $256 \times 256$ ReLU + `VecNorm` | Sin $R_{near}$ (`none`) | 64 | **79.60%** *(94.9% @1M)* | **67.2** | **0.029** | Descarte (Convergencia lenta) |
| **`X1a`** | $64 \times 64$ Tanh (Sin Norm) | `smooth_precision` | 64 | **77.00%** | **71.5** | **0.032** | Descarte (Capacidad insuficiente) |
| **`X2`** | $64 \times 64$ Tanh + `VecNorm` | `smooth_precision` | 64 | **74.73%** | **69.3** | **0.031** | Descarte |

*Intervalo de Indistinguibilidad Operacional:* $U_{equiv} = 3.16\text{ pp}$. Los modelos `X4`, `X3` y `X5` forman el cluster estadísticamente indistinguible de máximo desempeño.

---

## ⚖️ Diseño de Funciones de Recompensa y Diagnósticos

La recompensa descompuesta en cada paso $t$ está dada por:
$$R_t = R_{dist} + R_{action} + R_{near} + R_{vel} + R_{smooth} + R_{success}$$

* **Distancia ($R_{dist}$):** $-1.0 \cdot d$ (gradiente continuo negativo hacia el objetivo).
* **Esfuerzo de Control ($R_{action}$):** $-0.02 \|\mathbf{a}_t\|^2$ (penalización por saturación de actuadores).
* **Proximidad ($R_{near}$):** $(1.0 - d / 0.30) - 0.15 (|\dot{\theta}_1| + |\dot{\theta}_2|)$ en $d < 0.30\text{ m}$.
* **Penalización Gaussiana de Velocidad ($R_{vel}$ en `X5`):** $-w_v \exp\left(-\frac{d^2}{2\sigma_v^2}\right) \|\dot{\boldsymbol{\theta}}\|^2$ ($w_v = 0.10, \sigma_v = 0.15$).
* **Penalización de Variación de Acción ($R_{smooth}$ en `X6`):** $-w_a \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$ ($w_a = 0.01$).
* **Bonificación de Éxito ($R_{success}$):** $+15.0 \cdot \mathbb{I}_{succ}$ otorgado únicamente al consolidar el *hold*.

### Diagnóstico Empírico de los Signos:
* **El efecto de $R_{near}$:** En la Fase A se hipotetizó que el término positivo $(1 - d/0.30)$ era perjudicial ("trampa de recompensa"). Sin embargo, la evaluación a gran escala de la Fase B demostró que actúa como un *shaping* denso indispensable: suprimirlo (`X5_near`) reduce la tasa de éxito en $14.93\text{ pp}$ a igual presupuesto.
* **Valores excesivos de $w_v$:** Imponer $w_v \ge 1.0$ provoca **parálisis cinemática**, ya que el coste negativo por entrar velozmente supera la ganancia de acercarse. El valor óptimo es $w_v = 0.10$.

---

## 📂 Estructura del Proyecto

```text
robot-arm-2dof-ppo/
├── arm2d_env.py                           # Entorno Gymnasium (física, cinemática directa, hold, jacobiano)
├── interactive_render.py                  # Simulador interactivo Pygame a 60 FPS con auto-carga de normalización
├── configs/
│   ├── phase_a/                           # Configuraciones de diagnóstico de la Fase A (X0 a X6)
│   ├── phase_b/                           # Configuraciones a 1M pasos (X3, X5_near)
│   └── phase_b/extension/                 # Configuraciones de la Extensión (X1a, X2, X3, X4, X5, X6)
├── src/
│   ├── controllers/
│   │   └── jacobian_controller.py         # Controlador cinemático diferencial analítico C0 (Jacobiano DLS)
│   ├── evaluation/
│   │   ├── datasets.py                    # Generación y congelamiento criptográfico de datasets (n=200, 300, 500)
│   │   ├── evaluator.py                   # Motor formal de evaluación multi-episodio (determinista/estocástico)
│   │   └── metrics.py                     # Métricas operativas (éxito estricto, Wilson 95%, llegadas balísticas)
│   └── training/
│       ├── callbacks.py                   # Callbacks de telemetría y evaluación en línea
│       └── phase_b_callbacks.py           # Checkpointing atómico (acoplamiento model.zip + vecnormalize_stats.pkl)
├── scripts/
│   ├── phase_b_train.py                   # Script de entrenamiento para presupuestos a 1M pasos
│   ├── phase_b_ext_train.py               # Script de entrenamiento para la extensión (655k pasos)
│   ├── phase_b_ext_run_b12_final_eval.py  # Pipeline de evaluación terminal sobre n=500 episodios
│   ├── phase_b_ext_apply_selection.py     # Algoritmo de selección formal y cálculo de U_equiv
│   └── generate_phase_b_figures.py        # Generador de figuras vectoriales a 300 dpi con metadatos JSON
├── reports/
│   ├── phase_a_report.md                  # Informe exhaustivo de la Fase A
│   ├── phase_b_report.md                  # Informe consolidado de la Fase B y su Extensión
│   ├── data/phase_b/                      # CSVs de checkpoints, validación y datasets evaluados
│   └── figures/phase_b/                   # Figuras exportadas a 300 dpi con metadatos JSON asociados
├── tests/                                 # Suite formal de 29 tests unitarios (cinemática, hold, jacobiano)
├── requirements.txt                       # Dependencias fijadas del proyecto
└── README.md                              # Documentación técnica principal
```

---

## ⚙️ Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/AngelTomM/robot-arm-2dof-ppo.git
cd robot-arm-2dof-ppo
```

### 2. Entorno virtual (Python 3.11 recomendado)
- **Windows (PowerShell):**
  ```powershell
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  python3.11 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 🎮 Guía de Uso y Reproducibilidad

### 1. Simulación Interactiva en Tiempo Real
Permite interactuar visualmente con el agente entrenado a 60 FPS. Carga automáticamente el vector de normalización `vecnormalize_stats.pkl` acoplado al modelo:

```powershell
# Ejecutar con el agente líder X4:
python interactive_render.py experiments/phase_b/extension/X4/seed_42/checkpoints/ckpt_450560.zip --config configs/phase_b/extension/X4.yaml

# O con el agente X3:
python interactive_render.py experiments/phase_b/X3/seed_42/checkpoints/ckpt_409600.zip --config configs/phase_b/X3.yaml
```

**Controles interactivos:**
* 🖱️ **Clic Izquierdo:** Arrastra o reubica el objetivo hacia la posición del cursor.
* ⌨️ **Barra Espaciadora:** Genera un nuevo objetivo aleatorio dentro del espacio de trabajo alcanzable.
* 🔄 **Tecla 'R':** Reinicia la configuración articular del robot a cero.
* ❌ **Tecla 'ESC':** Cierra la ventana del simulador.

---

### 2. Entrenamiento de Agentes

#### Entrenamiento de Fase B Extensión (655.360 pasos, checkpoints atómicos cada 40.960 pasos):
```powershell
python scripts/phase_b_ext_train.py --config configs/phase_b/extension/X4.yaml --seed 42
```

#### Entrenamiento de Fase B a 1M de pasos:
```powershell
python scripts/phase_b_train.py --config configs/phase_b/X3.yaml --seed 42
```

---

### 3. Evaluación y Benchmarking Formal

Para replicar la evaluación multi-semilla a gran escala ($n=500$ episodios por semilla con dataset congelado y cálculo de intervalos Wilson 95%):

```powershell
python scripts/phase_b_ext_run_b12_final_eval.py
```

El resultado se registra de forma reproducible en `reports/data/phase_b/extension/final_b2_selection_verdict.json`.

---

### 4. Ejecución de Tests Unitarios

La suite de validación contiene **29 tests unitarios** que verifican la cinemática directa, el Jacobiano analítico, la lógica del contador de hold, la conservación de suavidad, la preservación de hashes históricos y la API de Gymnasium:

```powershell
python -m unittest discover -s tests
```

---

## 📄 Reportes e Informes Monográficos

Los análisis completos, desarrollos matemáticos y discusiones experimentales se encuentran en la carpeta `reports/`:

* 📘 [`reports/phase_a_report.md`](reports/phase_a_report.md): Informe monográfico de la Fase A (estabilización, transición de *touch* a *hold*, diagnóstico de $R_{near}$ y controlador analítico $C0$).
* 📗 [`reports/phase_b_report.md`](reports/phase_b_report.md): Informe monográfico de la Fase B y su Extensión (presupuesto a 1M pasos, dinámicas de checkpoints, evaluación de la familia X1–X6 y selección del agente líder $X4$).
* 📊 [`reports/data/phase_b/extension/final_b2_selection_verdict.json`](reports/data/phase_b/extension/final_b2_selection_verdict.json): Veredicto formal con datos crudos y rankings estadísticos.

---

## 📜 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más información.
