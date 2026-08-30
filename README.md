# 🤖 Aprendizaje por Refuerzo (PPO) - Brazo Robótico 2D (2 DOF)

Simulación y entrenamiento de un brazo robótico planar de 2 Grados de Libertad (2 DOF) mediante **Aprendizaje por Refuerzo Profundo** usando el algoritmo **PPO (Proximal Policy Optimization)** con **Gymnasium**, **Stable-Baselines3** y renderizado interactivo en tiempo real con **Pygame**.

---

## 📌 Tabla de Contenidos

- [Características](#-características)
- [Fundamento Teórico y Cinemática Directa](#-fundamento-teórico-y-cinemática-directa)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso y Ejecución](#-uso-y-ejecución)
  - [1. Verificación del Entorno](#1-verificación-del-entorno)
  - [2. Entrenamiento con PPO](#2-entrenamiento-con-ppo)
  - [3. Simulación Interactiva en Tiempo Real](#3-simulación-interactiva-en-tiempo-real)
  - [4. Monitoreo con TensorBoard](#4-monitoreo-con-tensorboard)
  - [5. Generar Gráficas de Rendimiento Offline](#5-generar-gráficas-de-rendimiento-offline)
- [Personalización y Parámetros](#-personalización-y-parámetros)
- [Licencia](#-licencia)

---

## 🚀 Características

- **Entorno Gymnasium Estándar (`Arm2DEnv`)**: Cumple con todas las especificaciones y APIs de Gymnasium.
- **Espacio de Acciones Continuo**: Control directo de velocidades angulares $(\dot{\theta}_1, \dot{\theta}_2)$ normalizadas en $[-1.0, 1.0]$.
- **Espacio de Observaciones Continuo**: Vector de 10 dimensiones sin discontinuidades angulares:
  $$[\cos\theta_1, \sin\theta_1, \cos\theta_2, \sin\theta_2, \dot{\theta}_1, \dot{\theta}_2, x_{target}, y_{target}, d_x, d_y]$$

$[\cos\theta_1, \sin\theta_1, \cos\theta_2, \sin\theta_2, \dot{\theta}_1, \dot{\theta}_2, x_{target}, y_{target}, d_x, d_y]$

- **Función de Recompensa Densa**: Incentiva la aproximación al objetivo, penaliza el consumo excesivo de energía/torque y otorga una bonificación de éxito al alcanzar la meta con precisión.
- **Renderizado Interactivo**: Ventana en tiempo real a 60 FPS con Pygame para interactuar con el brazo usando el mouse.
- **Análisis de Métricas**: Compatible con TensorBoard y generación automática de reportes gráficos.

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
├── arm2d_env.py           # Entorno Gymnasium con cinemática directa y renderizado Pygame
├── train.py               # Script de entrenamiento PPO con paralelización y callbacks
├── interactive_render.py  # Visualizador interactivo en tiempo real (control con mouse)
├── plot_results.py        # Generador de gráficas de métricas a partir de los logs
├── test_env.py            # Validación del entorno con Gymnasium check_env
├── requirements.txt       # Lista de dependencias fijadas y compatibles
├── .gitignore             # Archivos y carpetas ignorados por Git
└── README.md              # Documentación completa del proyecto
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

> [!TIP]
> Si no activas el entorno virtual previamente en PowerShell, puedes ejecutar los scripts anteponiendo `.venv\Scripts\` a los comandos (por ejemplo: `.venv\Scripts\python train.py`).

---

## 🎮 Uso y Ejecución

### 1. Verificación del Entorno

Valida que el entorno cumpla con todas las especificaciones de Gymnasium:

```bash
# Con entorno activado:
python test_env.py

# Sin activar el entorno (ruta directa en Windows):
.venv\Scripts\python test_env.py
```

### 2. Entrenamiento con PPO

Inicia el entrenamiento del agente con PPO (150,000 pasos en 4 entornos paralelos):

```bash
# Con entorno activado:
python train.py

# Sin activar el entorno (ruta directa en Windows):
.venv\Scripts\python train.py
```

Los modelos y mejores checkpoints se guardan automáticamente en `models/PPO/` y los logs en `logs/`.

### 3. Simulación Interactiva en Tiempo Real

Ejecuta la visualización interactiva a 60 FPS con el agente entrenado:

```bash
# Con entorno activado:
python interactive_render.py

# Sin activar el entorno (ruta directa en Windows):
.venv\Scripts\python interactive_render.py
```

**Controles del Simulador:**

- 🖱️ **Click Izquierdo**: Mueve el objetivo (Target) a la posición del cursor del mouse.
- ⌨️ **Barra Espaciadora**: Genera un objetivo aleatorio dentro del espacio alcanzable.
- 🔄 **Tecla 'R'**: Reinicia la pose y ángulos del brazo.
- ❌ **Tecla 'ESC'**: Cierra la simulación.

### 4. Monitoreo con TensorBoard

Para inspeccionar las curvas de recompensa media, pérdidas y entropía en tiempo real:

```powershell
# Con entorno activado:
tensorboard --logdir logs

# Sin activar el entorno (ruta directa en Windows):
.venv\Scripts\tensorboard --logdir logs
```

Abre tu navegador en `http://localhost:6006`.

### 5. Generar Gráficas de Rendimiento Offline

Si deseas exportar una imagen en alta definición con el análisis de curvas de aprendizaje sin abrir TensorBoard:

```bash
# Con entorno activado:
python plot_results.py

# Sin activar el entorno (ruta directa en Windows):
.venv\Scripts\python plot_results.py
```

La gráfica se guardará automáticamente en `analysis/training_curves.png`.

---

## 🔧 Personalización y Parámetros

- **Modificar la física o dimensiones del brazo**: Edita `arm2d_env.py` (`self.l1`, `self.l2`, `self.max_speed`, `self.target_radius`).
- **Ajustar la función de recompensa**: Modifica la fórmula en el método `step()` de `arm2d_env.py`.
- **Hiperparámetros de PPO**: Ajusta `learning_rate`, `n_steps`, `batch_size` o `ent_coef` en `train.py`.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.
