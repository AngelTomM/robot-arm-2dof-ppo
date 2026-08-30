# 🤖 Aprendizaje por Refuerzo (PPO) - Brazo Robótico 2D (2 DOF)

Simulación y entrenamiento de un brazo robótico planar de 2 Grados de Libertad (2 DOF) mediante **Aprendizaje por Refuerzo Profundo** usando el algoritmo **PPO (Proximal Policy Optimization)** con **Gymnasium**, **Stable-Baselines3** y renderizado interactivo en tiempo real con **Pygame**.

---

## 📌 Tabla de Contenidos

- [Características](#-características)
- [Fundamento Teórico y Cinemática Directa](#-fundamento-teórico-y-cinemática-directa)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso y Flujo de Trabajo en Equipo](#-uso-y-flujo-de-trabajo-en-equipo)
  - [1. Configuración Modular con YAML](#1-configuración-modular-con-yaml)
  - [2. Entrenar un Experimento](#2-entrenar-un-experimento)
  - [3. Simulación Interactiva en Tiempo Real](#3-simulación-interactiva-en-tiempo-real)
  - [4. Monitoreo con TensorBoard](#4-monitoreo-con-tensorboard)
  - [5. Comparar Experimentos y Reportes](#5-comparar-experimentos-y-reportes)
- [Buenas Prácticas para Trabajo Colaborativo](#-buenas-prácticas-para-trabajo-colaborativo)
- [Licencia](#-licencia)

---

## 🚀 Características

- **Configuración desacoplada en YAML**: Ajusta la física, hiperparámetros de PPO o funciones de recompensa sin modificar el código central.
- **Entorno Gymnasium Estándar (`Arm2DEnv`)**: Implementación estándar con espacios continuos de acción y observación (10 dimensiones).
- **Estrategias de Recompensa Modulares**:
  - `distance_only`: Penalización base por distancia euclidiana.
  - `smooth_precision`: Recompensa de proximidad y penalización de velocidad cerca de la meta para posarse con precisión sin oscilaciones ni sobreimpulsos.
  - `energy_efficient`: Minimización de esfuerzo y torques angulares.
- **Registro Automático de Metadatos**: Cada ejecución almacena `experiment_meta.json` dentro de `models/<nombre_experimento>/`.
- **Generación de Reportes Automática**: `compare_experiments.py` consolida todos los modelos en `reports/EXPERIMENTS_SUMMARY.md`.
- **Simulador Interactivo a 60 FPS**: Control visual en tiempo real con soporte para puntero de mouse y atajos de teclado.

---

## 📐 Fundamento Teórico y Cinemática Directa

El brazo robótico consta de dos articulaciones rotacionales y dos eslabones de longitudes $L_1$ y $L_2$, anclado a la base $(0, 0)$. Los ángulos son $\theta_1$ (ángulo absoluto del eslabón 1) y $\theta_2$ (ángulo relativo del eslabón 2 respecto al 1).

### Ecuaciones de Cinemática Directa
1. **Posición del Codo (Articulación 1)**:
   $$x_1 = L_1 \cdot \cos(\theta_1)$$
   $$y_1 = L_1 \cdot \sin(\theta_1)$$

2. **Posición del Efector Final (Articulación 2)**:
   $$x = x_1 + L_2 \cdot \cos(\theta_1 + \theta_2)$$
   $$y = y_1 + L_2 \cdot \sin(\theta_1 + \theta_2)$$

---

## 📂 Estructura del Proyecto

```text
├── configs/
│   ├── exp1_baseline.yaml              # Configuración base (baseline)
│   ├── exp2_smooth_precision.yaml      # Configuración de alta precisión y frenado suave
│   └── template_custom_experiment.yaml # Plantilla para crear nuevos experimentos
├── reports/
│   └── EXPERIMENTS_SUMMARY.md          # Tabla consolidada con todos los experimentos del equipo
├── arm2d_env.py                        # Entorno Gymnasium configurable con Pygame
├── train.py                            # Script de entrenamiento PPO con soporte YAML y metadatos
├── interactive_render.py               # Simulador interactivo en tiempo real (control con mouse)
├── plot_results.py                     # Generador de gráficas de rendimiento desde logs
├── compare_experiments.py              # Consolidador y generador de reportes comparativos
├── test_env.py                         # Test de validación con Gymnasium check_env
├── requirements.txt                    # Dependencias del proyecto
├── .gitignore                          # Archivos y carpetas ignorados por Git
└── README.md                           # Documentación completa
```

---

## ⚙️ Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/AngelTomM/robot-arm-2dof-ppo.git
   cd robot-arm-2dof-ppo
   ```

2. **Crear y activar el entorno virtual (Python 3.11 recomendado):**
   - **En Windows (PowerShell):**
     ```powershell
     py -3.11 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **En Linux / macOS:**
     ```bash
     python3.11 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Uso y Flujo de Trabajo en Equipo

### 1. Configuración Modular con YAML
Copia la plantilla `configs/template_custom_experiment.yaml` o modifica uno existente (ej. `configs/exp2_smooth_precision.yaml`):
```yaml
experiment:
  name: "exp2_smooth_precision"
  author: "Uli/Tomi"

env:
  max_speed: 1.5
  reward_type: "smooth_precision"

ppo:
  learning_rate: 0.00015
  n_steps: 2048

training:
  total_timesteps: 300000
```

### 2. Entrenar un Experimento
Inicia el entrenamiento pasando la ruta de la configuración deseada:
```powershell
.venv\Scripts\python train.py --config configs/exp2_smooth_precision.yaml
```
El modelo y sus metadatos se guardarán en `models/exp2_smooth_precision/`.

### 3. Simulación Interactiva en Tiempo Real
Prueba el modelo entrenado en la ventana interactiva:
```powershell
.venv\Scripts\python interactive_render.py models/exp2_smooth_precision/best_model/best_model.zip
```
**Controles:**
- 🖱️ **Click Izquierdo**: Mueve el objetivo (Target) a la posición del cursor del mouse.
- ⌨️ **Barra Espaciadora**: Genera un objetivo aleatorio dentro del área alcanzable.
- 🔄 **Tecla 'R'**: Reinicia los ángulos del brazo.
- ❌ **Tecla 'ESC'**: Cierra el simulador.

### 4. Monitoreo con TensorBoard
Visualiza las curvas de recompensa, entropía y pérdida en tiempo real:
```powershell
.venv\Scripts\tensorboard --logdir logs
```
Abre tu navegador en `http://localhost:6006`.

### 5. Comparar Experimentos y Reportes
Genera o actualiza la tabla de comparación de todos los modelos entrenados por el equipo:
```powershell
.venv\Scripts\python compare_experiments.py
```
El reporte se actualizará automáticamente en [`reports/EXPERIMENTS_SUMMARY.md`](reports/EXPERIMENTS_SUMMARY.md).

---

## 🤝 Buenas Prácticas para Trabajo Colaborativo

1. **Desacople de Parámetros**: No modifiques código central para cambiar valores; añade nuevos archivos en `configs/`.
2. **Registro Compartido**: Ejecuta y comitea `reports/EXPERIMENTS_SUMMARY.md` tras completar nuevos experimentos para que tu compañero vea el resumen en GitHub.
3. **Modelos Pesados**: Los binarios `.zip` de los modelos están excluidos de Git por `.gitignore`. Compártelos vía GitHub Releases o almacenamiento en la nube.
4. **Flujo de Ramas**:
   - `git checkout -b feature/nombre-de-la-mejora`
   - `git add configs/mi_exp.yaml reports/EXPERIMENTS_SUMMARY.md`
   - `git commit -m "feat: add nuevo experimento"`
   - `git push origin feature/nombre-de-la-mejora`

---

## 📄 Licencia
Este proyecto está bajo la Licencia MIT.
