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
  - [5. Comparar Experimentos y Resumen](#5-comparar-experimentos-y-resumen)
- [Buenas Prácticas para Trabajo Colaborativo](#-buenas-prácticas-para-trabajo-colaborativo)
- [Licencia](#-licencia)

---

## 🚀 Características

- **Configuración desacoplada en YAML**: Cambia la física, los hiperparámetros de PPO o el tipo de recompensa sin tocar el código fuente.
- **Entorno Gymnasium Estándar (`Arm2DEnv`)**: Compatible con el ecosistema de RL.
- **Estrategias de Recompensa Modulares**:
  - `distance_only`: Penalización básica por distancia euclidiana.
  - `smooth_precision`: Recompensa de proximidad y penalización de frenado cerca del objetivo para evitar sobreimpulsos.
  - `energy_efficient`: Penalización por esfuerzo y uso de torques.
- **Registro Automático de Metadatos**: Cada ejecución guarda un `experiment_meta.json` con todos los parámetros usados.
- **Renderizado Interactivo**: Ventana a 60 FPS con Pygame para mover el objetivo con el mouse o la barra espaciadora.

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
│   ├── exp1_baseline.yaml         # Configuración base
│   └── exp2_smooth_precision.yaml # Configuración optimizada para precisión y suavidad
├── arm2d_env.py                   # Entorno Gymnasium configurable
├── train.py                       # Script de entrenamiento con soporte para YAML
├── interactive_render.py          # Simulador interactivo en tiempo real (Pygame)
├── plot_results.py                # Generador de gráficas de entrenamiento
├── compare_experiments.py         # Comparador y tabla resumen de experimentos
├── test_env.py                    # Validación con Gymnasium check_env
├── requirements.txt               # Dependencias del proyecto
├── .gitignore                     # Archivos ignorados por Git
└── README.md                      # Documentación completa
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
Crea tu propio archivo en `configs/` (por ejemplo: `configs/exp_tomi.yaml` o `configs/exp_uli.yaml`):
```yaml
experiment:
  name: "exp2_smooth_precision"
  author: "Uli"

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
Ejecuta el entrenamiento pasando tu archivo de configuración:
```powershell
.venv\Scripts\python train.py --config configs/exp2_smooth_precision.yaml
```
Los modelos se guardan en `models/<nombre_experimento>/` junto con su `experiment_meta.json`.

### 3. Simulación Interactiva en Tiempo Real
Prueba cualquier modelo generado en el simulador:
```powershell
.venv\Scripts\python interactive_render.py models/exp2_smooth_precision/best_model/best_model.zip
```
**Controles:**
- 🖱️ **Click Izquierdo**: Mueve el objetivo (Target) al puntero del mouse.
- ⌨️ **Barra Espaciadora**: Genera un nuevo objetivo aleatorio.
- 🔄 **Tecla 'R'**: Reinicia los ángulos del brazo.
- ❌ **Tecla 'ESC'**: Cierra el simulador.

### 4. Monitoreo con TensorBoard
Para comparar las curvas de varios experimentos al mismo tiempo:
```powershell
.venv\Scripts\tensorboard --logdir logs
```
Abre tu navegador en `http://localhost:6006`.

### 5. Comparar Experimentos y Resumen
Para ver una tabla con los parámetros y estados de todos los modelos entrenados por el equipo:
```powershell
.venv\Scripts\python compare_experiments.py
```

---

## 🤝 Buenas Prácticas para Trabajo Colaborativo

1. **No modificar código central para cambiar parámetros**: Agrega o modifica archivos en la carpeta `configs/`.
2. **Archivos `.zip` y logs locales**: No se suben a Git (están en `.gitignore`). Si deseas compartir un modelo `.zip` específico con tu compañero, puedes adjuntarlo en un **GitHub Release** o carpeta compartida de Google Drive.
3. **Flujo de ramas**:
   - Crea tu rama para nuevas funciones: `git checkout -b feature/mi-mejora`
   - Sube tu configuración: `git add configs/mi_exp.yaml && git commit -m "add: mi config"`
   - Haz Pull Request a `main`.

---

## 📄 Licencia
Este proyecto está bajo la Licencia MIT.
