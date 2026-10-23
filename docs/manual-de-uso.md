# Manual de Operación y Uso — Manipulador Planar 2-DOF (Fase D: MuJoCo Reacher-v5)

Guía operativa para la instalación, entrenamiento, evaluación mecatrónica, simulación interactiva 3D y ejecución de pruebas unitarias sobre el manipulador planar 2-DOF accionado por torque. Para el fundamento teórico y la formulación matemática, consulte [docs/metodologia.md](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/docs/metodologia.md).

---

## 1. Configuración del Entorno de Ejecución

### 1.1 Requisitos del Sistema
* Python 3.11 (o superior compatible con MuJoCo).
* Entorno virtual activo con acceso a librerías gráficas de OpenGL.

### 1.2 Activación del Entorno Virtual e Instalación de Dependencias
```powershell
# 1. Activación del entorno virtual (Windows PowerShell)
..\.venv\Scripts\Activate.ps1

# 2. Instalación de paquetes requeridos
pip install -r requirements.txt
```

Las dependencias principales contemplan `gymnasium[mujoco]>=1.0.0`, `mujoco>=2.3.3`, `imageio>=2.14.1`, `stable-baselines3>=2.0.0`, `torch>=2.0.0`, `matplotlib>=3.8.0`, `pandas>=2.1.0` y `pytest>=8.0.0`.

---

## 2. Entrenamiento de Agentes (`train.py`)

El script en la raíz [train.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/train.py) centraliza la inicialización, vectorización y entrenamiento de los agentes bajo el paradigma MDP continuo:
* **Resolución de Hiperparámetros (SSOT):** Fusiona automáticamente los parámetros físicos invariantes de [configs/environment.yaml](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/configs/environment.yaml) con la configuración específica del algoritmo.
* **Selección de Algoritmo:** Detecta si la configuración define hiperparámetros para PPO (`ppo:`) o SAC (`sac:`).
* **Normalización Acoplada:** Utiliza `VecNormalize` sobre observaciones físicas crudas preservando las escalas originales de recompensa (`norm_reward: false`).

### 2.1 Sintaxis del Comando
```powershell
python train.py --config <ruta_yaml> [--seed <int>] [--device <str>] [--total_timesteps <int>] [--checkpoint_freq <int>]
```

### 2.2 Comandos de Ejecución Estándar
```powershell
# Entrenamiento completo de PPO_1 (300.000 timesteps, seed 42)
python train.py --config configs/phase_D/prueba/PPO_1.yaml --seed 42

# Entrenamiento completo de SAC_1 (300.000 timesteps, seed 42)
python train.py --config configs/phase_D/prueba/SAC_1.yaml --seed 42

# Pruebas de integración rápida (Smoke tests, 256 timesteps)
python train.py --config configs/phase_D/smoke/PPO_smoke.yaml --total_timesteps 256
python train.py --config configs/phase_D/smoke/SAC_smoke.yaml --total_timesteps 256
```

### 2.3 Estructura de Artefactos de Salida
Cada sesión de entrenamiento genera un directorio estructurado en `experiments/<fase>/<experimento>/seed_<semilla>/`:
* `models/<algoritmo>_arm2d_final.zip`: Red neuronal con los pesos finales entrenados.
* `models/best_model/`: Checkpoint con el mayor retorno obtenido en validación periódica (`best_model.zip` junto a `vecnormalize_stats.pkl`).
* `checkpoints/ckpt_<paso>/`: Puntos de control intermedios acoplados para análisis de convergencia.
* `logs/`: Registros de telemetría continuos para TensorBoard (`tensorboard --logdir experiments/`).

---

## 3. Evaluación Mecatrónica y Benchmarking (`evaluate.py`)

El script [evaluate.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/evaluate.py) ejecuta la evaluación determinista de políticas entrenadas o del controlador clásico sobre conjuntos congelados de datos, cuantificando la estabilización terminal y el costo LQR sobre coordenadas físicas crudas.

### 3.1 Sintaxis del Comando
```powershell
python evaluate.py [--agent <ppo|sac|c0>] [--model-path <ruta_modelo.zip>] [--c0] [--dataset <ruta_dataset.json>] [--output-dir <directorio_salida>]
```

### 3.2 Ejemplos de Evaluación
```powershell
# 1. Evaluación del baseline analítico C0 sobre el dataset de validación (200 episodios)
python evaluate.py --c0 --dataset data/datasets/validation_set_200.json

# 2. Evaluación de un modelo PPO sobre el conjunto de evaluación final (300 episodios)
python evaluate.py --agent ppo --model-path experiments/phase_D_prueba/PPO_1/seed_42/models/ppo_arm2d_final.zip --dataset data/datasets/final_eval_set_300.json

# 3. Evaluación de un modelo SAC con estadísticas explícitas de normalización
python evaluate.py --agent sac --model-path experiments/phase_D_prueba/SAC_1/seed_42/models/best_model/best_model.zip --vecnormalize experiments/phase_D_prueba/SAC_1/seed_42/models/best_model/vecnormalize_stats.pkl --dataset data/datasets/final_eval_set_300.json
```

Los resultados se almacenan en `reports/data/evaluation/`:
* `<agente>_episodes.csv`: Telemetría paso a paso por episodio ($e_{\text{ss}}, v_{\text{ss}}, t_s, J_u, J_{\text{lqr}}$).
* `<agente>_summary.json`: Métricas estadísticas agregadas e intervalos de confianza de Wilson.

---

## 4. Generación de Gráficas Comparativas (`plot.py`)

El script [plot.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/plot.py) procesa de forma desacoplada los archivos CSV generados por `evaluate.py` para construir las figuras de análisis mecatrónico sin requerir dependencias del simulador físico:

```powershell
python plot.py --data-dir reports/data/evaluation --out-dir reports/figures
```

Figuras generadas en `reports/figures/`:
* `fig1_success_rate_comparison.png`: Comparativa de tasa de estabilización con barras de error binomial (Wilson al 95%).
* `fig2_steady_state_precision.png`: Diagramas de caja y distribuciones de error estacionario ($e_{\text{ss}}$) y velocidad angular residual ($v_{\text{ss}}$).
* `fig3_settling_time_and_effort.png`: Dispersión y correlación entre tiempo de establecimiento ($t_s$) y esfuerzo acumulado de torque ($J_u$).

---

## 5. Simulación Visual en Tiempo Real (`interactive_render.py`)

El visualizador [interactive_render.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/interactive_render.py) interactúa directamente con la ventana nativa OpenGL de MuJoCo en 3D:

```powershell
# Visualización de un modelo entrenado de RL
python interactive_render.py experiments/phase_D_prueba/PPO_1/seed_42/models/best_model/best_model.zip

# Visualización del controlador cartesiano analítico C0
python interactive_render.py --c0 --episodes 5
```

En la ventana de MuJoCo es posible manipular la cámara orbital mediante el ratón (rotación con clic izquierdo, traslación con clic derecho y zoom con la rueda). La consola imprime periódicamente la telemetría del efector final, torques aplicados y recompensas instantáneas.

---

## 6. Ejecución de la Suite de Pruebas Unitarias

La suite de validación asegura la coherencia dimensional, cinemática y algorítmica del proyecto:

```powershell
python -m pytest tests/
```

Módulos de prueba verificados:
* [tests/test_config.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/tests/test_config.py): Carga desacoplada de configuraciones y verificación de parámetros SSOT de MuJoCo Reacher-v5.
* [tests/test_kinematics.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/tests/test_kinematics.py): Validación de cinemática directa y matriz Jacobiana contrastada contra las funciones internas de MuJoCo.
* [tests/test_normalization.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/tests/test_normalization.py): Normalización vectorial `VecNormalize`, cálculo de recompensa cuadrática continua LQR y truncamiento de horizonte.