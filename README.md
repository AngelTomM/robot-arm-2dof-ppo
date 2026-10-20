# 🤖 Brazo Robótico 2 DOF: Benchmarking por Aprendizaje por Refuerzo (PPO / SAC) vs. Control Cinemático Clásico

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-informational.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.9%2B-green.svg)](https://stable-baselines3.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-31%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Entorno de simulación, benchmarking cinemático y control de un brazo robótico planar de 2 Grados de Libertad (2 DOF) mediante **Aprendizaje por Refuerzo Profundo** (**PPO** y **SAC**, Stable-Baselines3) frente a un **Controlador Cinemático Diferencial Analítico (Jacobiano DLS `C0`)**.

El proyecto consolida la investigación de la **Fase A** (estabilización rigurosa bajo criterio de *hold* y resolución de patologías de recompensa), la **Fase B** (presupuesto extendido a 1.000.000 de pasos con checkpointing acoplado), la **Extensión de Fase B** (evaluación multi-semilla y selección del agente líder **`X4`**) y la **Fase C** (**SAC** unifactorial y paquete combinado con suavidad). El entrenamiento de cualquier fase se realiza desde un **punto de entrada único** (`train.py`), que deriva la fase y el experimento de la ruta de configuración (fuente única de verdad).

---

## 📌 Documentación

| Documento | Contenido |
| :--- | :--- |
| 📘 [`docs/manual-de-uso.md`](docs/manual-de-uso.md) | Manual operativo: instalación, referencia completa de `train.py`, simulación interactiva, evaluación formal, tests y estructura del repositorio. |
| 📐 [`docs/metodologia.md`](docs/metodologia.md) | Fundamento técnico: física y cinemática, criterio de éxito *hold*, diseño de recompensas y protocolo experimental. |
| 📗 [`reports/`](reports/) | Informes monográficos por fase, datos agregados y figuras con metadatos de reproducibilidad. |

---

## 🚀 Hallazgos Clave

1. **Agente Líder Seleccionado (`X4`):**
   - Arquitectura MLP $256 \times 256$ ReLU, `VecNormalize(norm_obs=True)`, recompensa `smooth_precision` y `batch_size = 256`.
   - Alcanzó **96.73% de éxito estricto determinista** ($n=500$ episodios por semilla, 3 semillas independientes), con una mediana de **59.7 pasos** al éxito y una suavidad de control $\Delta a = 0.046$.
   - Comparte el grupo de equivalencia estadística superior con **`X3`** y **`X5`**.
2. **Cota Superior Analítica (`C0`):**
   - Controlador cinemático diferencial basado en la pseudo-inversa amortiguada del Jacobiano (DLS, $K_p = 3.0, \lambda = 0.01$).
   - Alcanza **100.0% de éxito estricto**, 0.0% de truncamientos y 0.0% de oscilaciones con una distancia final media de $0.0028\text{ m}$.
3. **Criterio de Éxito Estricto (*Hold*):**
   - La condición de mero contacto (*touch*) incentivaba atravesar el objetivo a máxima velocidad, generando éxitos no representativos de estabilidad real.
   - El estándar vigente exige **permanencia sostenida en la zona de precisión con velocidad reducida**, garantizando estabilidad cinemática verificable.
4. **Normalización y Checkpointing Acoplado:**
   - Se demostró que desacoplar la red neuronal de las estadísticas de `VecNormalize` distorsiona la evaluación.
   - Toda la pipeline aplica guardado acoplado (`model.zip` + `vecnormalize_stats.pkl`) en cada checkpoint, sincronizando el entorno de evaluación en modo inferencia.

---

## 🏆 Ranking Consolidado

Resultados de la **Extensión de Fase B** ($n=500$ episodios por semilla, semillas `42`, `100`, `2026`) y evaluación a 1M pasos:

| Agente | Arquitectura / Normalización | Recompensa | Batch Size | Éxito Estricto Determ. | Mediana Pasos | Veredicto |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`C0`** | Analítico (Jacobiano DLS) | N/A | N/A | **100.0%** | **59.0** | **Cota Superior Analítica** |
| **`X4`** | $256 \times 256$ ReLU + `VecNorm` | `smooth_precision` | 256 | **96.73%** | **59.7** | 🥇 **Agente Líder** |
| **`X3`** | $256 \times 256$ ReLU + `VecNorm` | `smooth_precision` | 64 | **94.53%** *(98.0% @1M)* | **57.7** | 🥈 Grupo Equivalente Líder |
| **`X5`** | $256 \times 256$ ReLU + `VecNorm` | + Gaussiana $w_v=0.10$ | 64 | **93.93%** | **60.0** | 🥉 Grupo Equivalente Líder |
| **`X6`** | $256 \times 256$ ReLU + `VecNorm` | + Suavidad $w_a=0.01$ | 64 | **92.67%** | **59.3** | Descarte ($\Delta > U_{equiv}$) |
| **`X5_near`** | $256 \times 256$ ReLU + `VecNorm` | Sin $R_{near}$ | 64 | **79.60%** *(94.9% @1M)* | **67.2** | Descarte (convergencia lenta) |
| **`X1a`** | $64 \times 64$ Tanh (sin norm) | `smooth_precision` | 64 | **77.00%** | **71.5** | Descarte (capacidad insuficiente) |
| **`X2`** | $64 \times 64$ Tanh + `VecNorm` | `smooth_precision` | 64 | **74.73%** | **69.3** | Descarte |

*Intervalo de Indistinguibilidad Operacional:* $U_{equiv} = 3.16$ pp. Los modelos `X4`, `X3` y `X5` forman el grupo estadísticamente indistinguible de máximo desempeño. Detalle metodológico completo en [`docs/metodologia.md`](docs/metodologia.md).

---

## ⚡ Inicio Rápido

### Instalación
```bash
git clone https://github.com/AngelTomM/robot-arm-2dof-ppo.git
cd robot-arm-2dof-ppo
python3.11 -m venv .venv      # o active el entorno consolidado: ..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Entrenamiento (punto de entrada único)
La fase y el experimento se derivan de la ruta del config; el algoritmo, de su sección (`ppo:` / `sac:`):

```powershell
# PPO — Extensión de Fase B (agente líder X4, fase_b_ext)
python train.py --config configs/phase_b/extension/X4.yaml --seed 42

# SAC — Fase C
python train.py --config configs/phase_c/S1.yaml --seed 42
```

Salida: `experiments/<fase>/<exp_id>/seed_<n>/` con modelos, checkpoints acoplados al vector de normalización, logs de TensorBoard y metadatos. Referencia completa de argumentos y artefactos en [`docs/manual-de-uso.md`](docs/manual-de-uso.md).

### Validación
```powershell
python -m pytest tests/ -q
```

---

## 📋 Vista de la Estructura

```text
├── train.py            # Punto de entrada único de entrenamiento (PPO/SAC)
├── arm2d_env.py        # Entorno Gymnasium (física, cinemática directa, hold, jacobiano)
├── interactive_render.py   # Simulador interactivo Pygame
├── configs/            # YAMLs por fase: phase_a/, phase_b/, phase_b/extension/, phase_c/
├── experiments/        # Salidas: <fase>/<exp_id>/seed_<seed>/
├── src/                # controllers/ (C0), evaluation/ (datasets, evaluador, métricas), training/ (callbacks)
├── scripts/            # Evaluación formal, selección y generación de figuras
├── reports/            # Informes monográficos y datos de las fases
├── docs/               # Manual de uso y metodología
└── tests/              # Suite de 31 pruebas unitarias
```

---

## 📜 Licencia

Este proyecto está bajo la Licencia MIT. Consulte el archivo [`LICENSE`](LICENSE) para más información.
