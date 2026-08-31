# 🤖 Aprendizaje por Refuerzo (PPO / SAC) - Brazo Robótico 2D (2 DOF)

Simulación y entrenamiento de un brazo robótico planar de 2 Grados de Libertad (2 DOF) mediante **Aprendizaje por Refuerzo Profundo** usando algoritmos continuos **PPO (Proximal Policy Optimization)** y **SAC (Soft Actor-Critic)** con **Gymnasium**, **Stable-Baselines3** y renderizado interactivo en tiempo real con **Pygame**.

---

## 📌 Tabla de Contenidos

- [Características](#-características)
- [Fundamento Teórico y Cinemática Directa](#-fundamento-teórico-y-cinemática-directa)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Guía de Uso y Comandos Rápidos](#-guía-de-uso-y-comandos-rápidos)
  - [1. Verificación del Entorno](#1-verificación-del-entorno)
  - [2. Entrenar un Experimento (PPO o SAC)](#2-entrenar-un-experimento-ppo-o-sac)
  - [3. Simulación Interactiva en Tiempo Real](#3-simulación-interactiva-en-tiempo-real)
  - [4. Monitoreo y Análisis con TensorBoard](#4-monitoreo-y-análisis-con-tensorboard)
  - [5. Comparación y Resumen de Experimentos](#5-comparación-y-resumen-de-experimentos)
- [Buenas Prácticas para Trabajo Colaborativo](#-buenas-prácticas-para-trabajo-colaborativo)
- [Licencia](#-licencia)

---

## 🚀 Características

- **Soporte Multialgoritmo (PPO y SAC)**: Elige entre PPO (*On-Policy*) o SAC (*Off-Policy con Replay Buffer*) simplemente cambiando el campo `algorithm` en el archivo YAML.
- **Configuración desacoplada en YAML**: Ajusta la física, hiperparámetros de RL o funciones de recompensa sin modificar el código fuente.
- **Detección Automática de Modelos en el Simulador**: `interactive_render.py` detecta si el modelo es PPO o SAC y carga la física con la que fue entrenado para evitar discrepancias de observación.
- **Entorno Gymnasium Estándar (`Arm2DEnv`)**: Implementación compatible con espacios continuos de acción y observación (10 dimensiones).
- **Estrategias de Recompensa Modulares**:
  - `distance_only`: Penalización base por distancia euclidiana.
  - `smooth_precision`: Recompensa de proximidad fina y penalización de velocidad cerca de la meta para posarse con precisión sin oscilaciones ni sobreimpulsos.
  - `energy_efficient`: Minimización de esfuerzo y torques angulares.
- **Registro Automático de Metadatos**: Cada ejecución almacena un `.json` en `reports/runs/` para compartir por Git sin subir modelos pesados.
- **Generación de Reportes Automática**: `compare_experiments.py` consolida todos los modelos del equipo en `reports/EXPERIMENTS_SUMMARY.md`.
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
│   ├── exp1_baseline.yaml              # Configuración base PPO (150k pasos)
│   ├── exp2_smooth_precision.yaml      # Configuración PPO de alta precisión y frenado suave
│   ├── exp_sac_baseline.yaml           # Configuración base SAC (Soft Actor-Critic)
│   └── template_custom_experiment.yaml # Plantilla comentada para nuevos experimentos
├── reports/
│   ├── runs/                           # Metadatos livianos de cada entrenamiento para compartir
│   └── EXPERIMENTS_SUMMARY.md          # Tabla consolidada con todos los experimentos del equipo
├── arm2d_env.py                        # Entorno Gymnasium configurable con Pygame
├── train.py                            # Script de entrenamiento multialgoritmo (PPO / SAC)
├── interactive_render.py               # Simulador interactivo (detección automática PPO/SAC)
├── compare_experiments.py              # Consolidador y generador de reportes comparativos
├── test_env.py                         # Test de validación con Gymnasium check_env
├── requirements.txt                    # Dependencias del proyecto
├── .gitignore                          # Archivos y carpetas ignorados por Git
└── README.md                           # Documentación completa
```

---

## ⚙️ Instalación

1. **Clonar el repositorio:**
   ```powershell
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
   ```powershell
   pip install -r requirements.txt
   ```

---

## 🎮 Guía de Uso y Comandos Rápidos

### 1. Verificación del Entorno
Comprueba que el entorno físico cumpla con los estándares de la API de Gymnasium:

```powershell
.venv\Scripts\python test_env.py
```

---

### 2. Entrenar un Experimento (PPO o SAC)

**Comando rápido (plantilla):**
```powershell
.venv\Scripts\python train.py --config configs/<tu_archivo>.yaml
```

**Ejemplos listos para ejecutar:**
- **Entrenar con PPO (Baseline):**
  ```powershell
  .venv\Scripts\python train.py --config configs/exp1_baseline.yaml
  ```
- **Entrenar con PPO (Precisión y Suavidad):**
  ```powershell
  .venv\Scripts\python train.py --config configs/exp2_smooth_precision.yaml
  ```
- **Entrenar con SAC (Soft Actor-Critic):**
  ```powershell
  .venv\Scripts\python train.py --config configs/exp_sac_baseline.yaml
  ```

---

### 3. Simulación Interactiva en Tiempo Real

**Comando rápido (plantilla):**
```powershell
.venv\Scripts\python interactive_render.py models/<nombre_experimento>/best_model/best_model.zip
```

**Ejemplos listos para ejecutar:**
- **Probar modelo SAC:**
  ```powershell
  .venv\Scripts\python interactive_render.py models/exp_sac_baseline/best_model/best_model.zip
  ```
- **Probar modelo PPO de precisión:**
  ```powershell
  .venv\Scripts\python interactive_render.py models/exp2_smooth_precision/best_model/best_model.zip
  ```

**Controles del Simulador:**
- 🖱️ **Click Izquierdo**: Mueve el objetivo (Target) a la posición del cursor del mouse.
- ⌨️ **Barra Espaciadora**: Genera un objetivo aleatorio dentro del área alcanzable.
- 🔄 **Tecla 'R'**: Reinicia los ángulos del brazo.
- ❌ **Tecla 'ESC'**: Cierra el simulador.

---

### 4. Monitoreo y Análisis con TensorBoard
Para inspeccionar las curvas de aprendizaje en vivo o comparar múltiples ejecuciones superpuestas:

```powershell
.venv\Scripts\tensorboard --logdir logs
```
Abre tu navegador en: `http://localhost:6006`

#### 📊 Guía de Secciones Principales en TensorBoard:
- **`eval/` (Evaluación Determinística)**:
  - `eval/mean_reward`: **Métrica reina**. Mide la recompensa media en modo examen (sin ruido aleatorio). Si sube de forma continua, el agente aprendió exitosamente.
  - `eval/success_rate`: Porcentaje de éxito tocando el objetivo (0.0 a 1.0).
- **`rollout/` (Fase de Exploración en Vivo)**:
  - `rollout/ep_rew_mean`: Recompensa promedio mientras el robot explora caminos nuevos.
  - `rollout/ep_len_mean`: Cuántos pasos tarda en terminar el episodio. Debe disminuir a medida que el robot aprende a llegar más rápido.
- **`train/` (Salud de la Red Neuronal y Optimización)**:
  - `train/entropy_loss` (PPO) / `train/ent_coef` (SAC): Mide la exploración/creatividad.
  - `train/actor_loss` / `train/critic_loss` (SAC): Estabilidad del actor y el crítico.
  - `train/approx_kl` (PPO): Estabilidad de los pasos de optimización de PPO.

---

### 5. Comparación y Resumen de Experimentos

**Comando rápido (por defecto: mejor a peor recompensa):**
```powershell
.venv\Scripts\python compare_experiments.py --sort reward
```

**Opciones adicionales de ordenamiento:**
```powershell
.venv\Scripts\python compare_experiments.py --sort author
```
```powershell
.venv\Scripts\python compare_experiments.py --sort date
```
```powershell
.venv\Scripts\python compare_experiments.py --sort steps
```

El reporte se actualiza automáticamente en [`reports/EXPERIMENTS_SUMMARY.md`](reports/EXPERIMENTS_SUMMARY.md).

---

## 🤝 Buenas Prácticas para Trabajo Colaborativo

1. **Trabajar en ramas separadas**:
   ```powershell
   git checkout -b feature/tomi-experimento-sac
   ```
2. **Entrenar y registrar metadatos**:
   Al entrenar, se genera automáticamente un archivo liviano en `reports/runs/exp_nombre_Tomi.json`.
3. **Compartir resultados sin subir modelos pesados**:
   ```powershell
   git add configs/ reports/runs/
   git commit -m "feat: add nuevo experimento con SAC"
   git push origin feature/tomi-experimento-sac
   ```
4. **Sincronizar y actualizar el reporte de todo el equipo**:
   Al hacer merge en GitHub o `git pull origin main`, ejecutan:
   ```powershell
   .venv\Scripts\python compare_experiments.py
   ```
   Y la tabla [`reports/EXPERIMENTS_SUMMARY.md`](reports/EXPERIMENTS_SUMMARY.md) sumará automáticamente los experimentos de Tomi y Gaby sin conflictos.

---

## 📄 Licencia
Este proyecto está bajo la Licencia MIT.
