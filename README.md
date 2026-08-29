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
  - [2. Entrenamiento](#2-entrenamiento)
  - [3. Simulación Interactiva en Tiempo Real](#3-simulación-interactiva-en-tiempo-real)
  - [4. Monitoreo con TensorBoard](#4-monitoreo-con-tensorboard)
- [Licencia](#-licencia)

---

## 🚀 Características
- **Entorno Gymnasium Estándar (`Arm2DEnv`)**: Compatible con el ecosistema moderno de Reinforcement Learning.
- **Espacio de Acciones Continuo**: Control directo de velocidades angulares $(\dot{\theta}_1, \dot{\theta}_2)$ normalizadas en $[-1.0, 1.0]$.
- **Espacio de Observaciones Continuo**: Vector de 10 dimensiones evitando discontinuidades angulares:
  $$[\cos\theta_1, \sin\theta_1, \cos\theta_2, \sin\theta_2, \dot{\theta}_1, \dot{\theta}_2, x_{target}, y_{target}, d_x, d_y]$$
- **Función de Recompensa Densa**: Incentiva la aproximación al objetivo, penaliza el consumo excesivo de energía y otorga una bonificación sustancial al alcanzar la meta con precisión.
- **Renderizado Interactivo**: Ventana en tiempo real a 60 FPS desarrollada en Pygame donde puedes interactuar con el brazo usando el mouse.

---

## 📐 Fundamento Teórico y Cinemática Directa

El brazo robótico se modela con dos eslabones de longitudes $L_1$ y $L_2$, anclado a la base $(0, 0)$. Los ángulos de las articulaciones son $\theta_1$ (ángulo absoluto del primer eslabón) y $\theta_2$ (ángulo relativo del segundo eslabón respecto al primero).

### Ecuaciones de Cinemática Directa:
1. **Posición del Codo (Articulación 1)**:
   $$x_1 = L_1 \cdot \cos(\theta_1)$$
   $$y_1 = L_1 \cdot \sin(\theta_1)$$

2. **Posición del Efector Final (Articulación 2)**:
   $$x = x_1 + L_2 \cdot \cos(\theta_1 + \theta_2)$$
   $$y = y_1 + L_2 \cdot \sin(\theta_1 + \theta_2)$$

---

## 📂 Estructura del Proyecto

```text
├── arm2d_env.py           # Clase del entorno Gymnasium con física y render Pygame
├── train.py               # Script de entrenamiento PPO con paralelización y callbacks
├── interactive_render.py  # Visualizador interactivo en tiempo real con soporte para mouse
├── test_env.py            # Validación del entorno con Gymnasium check_env
├── requirements.txt       # Lista de dependencias fijadas y compatibles
├── .gitignore             # Archivos y carpetas excluidas de Git
└── README.md              # Documentación completa del proyecto
```

---

## ⚙️ Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
   cd TU_REPOSITORIO
   ```

2. **Crear y activar el entorno virtual (Python 3.11 recomendado):**
   ```bash
   # En Windows PowerShell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Uso y Ejecución

### 1. Verificación del Entorno
Valida que el entorno cumpla con las especificaciones de Gymnasium:
```bash
python test_env.py
```

### 2. Entrenamiento
Inicia el entrenamiento del agente mediante PPO (por defecto 150,000 pasos en 4 entornos vectorizados):
```bash
python train.py
```
Los modelos y mejores checkpoints se guardarán automáticamente en `models/PPO/`.

### 3. Simulación Interactiva en Tiempo Real
Ejecuta la visualización interactiva:
```bash
python interactive_render.py
```
**Controles del Simulador:**
- 🖱️ **Click Izquierdo**: Mueve el objetivo (Target) a la posición del puntero.
- ⌨️ **Barra Espaciadora**: Genera un objetivo aleatorio dentro del rango de alcance.
- 🔄 **Tecla 'R'**: Reinicia los ángulos del brazo.
- ❌ **Tecla 'ESC'**: Cierra la simulación.

### 4. Monitoreo con TensorBoard
Para inspeccionar las curvas de recompensa, pérdida de la política y longitud de episodios en tiempo real:
```bash
tensorboard --logdir logs
```
Abre tu navegador en `http://localhost:6006`.

---

## 📄 Licencia
Este proyecto está bajo la Licencia MIT.
