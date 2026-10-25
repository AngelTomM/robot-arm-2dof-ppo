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
python train.py --config <ruta_yaml> [--seed <int|lista>] [--device <str>] [--total_timesteps <int>] [--checkpoint_freq <int>]
```

### 2.2 Comandos de Ejecución Estándar

```powershell
# Entrenamiento con una única semilla (300.000 timesteps, seed 42)
python train.py --config configs/entorno/PPO_1.yaml --seed 42

# Entrenamiento multisemilla secuencial en una sola instrucción (semillas 42, 100 y 2026)
python train.py --config configs/entorno/PPO_1.yaml --seed "42, 100, 2026"

# Entrenamiento completo de SAC_1 (300.000 timesteps, seed 42)
python train.py --config configs/entorno/SAC_1.yaml --seed 42

# Pruebas de integración rápida (Smoke tests, 256 timesteps)
python train.py --config configs/smoke/PPO_smoke.yaml --total_timesteps 256
python train.py --config configs/smoke/SAC_smoke.yaml --total_timesteps 256
```

### 2.3 Estructura de Artefactos de Salida

Cada sesión de entrenamiento genera un directorio estructurado en `experiments/<fase>/<experimento>/seed_<semilla>/`:

* `models/<algoritmo>_arm2d_final.zip`: Red neuronal con los pesos finales entrenados.
* `models/best_model/`: Checkpoint con el mayor retorno obtenido en validación periódica (`best_model.zip` junto a `vecnormalize_stats.pkl`).
* `checkpoints/ckpt_<paso>/`: Puntos de control intermedios acoplados para análisis de convergencia.
* `logs/`: Registros de telemetría continuos para TensorBoard (`tensorboard --logdir experiments/`) y archivo de validación periódica `evaluations.npz`.

---

## 3. Evaluación Mecatrónica y Benchmarking (`evaluate.py`)

El script [evaluate.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/evaluate.py) ejecuta la evaluación determinista de políticas de aprendizaje por refuerzo o del controlador cinemático C0 sobre conjuntos congelados de datos. Los resultados cuantitativos se organizan en subcarpetas exclusivas dentro de `reports/data/evaluation/<label>/`.

El parámetro `--label` define el identificador asignado al modelo o controlador para denominar su directorio de datos (`reports/data/evaluation/<label>/`) y su directorio de figuras (`reports/figures/<label>/`). En caso de omitirse, el script infiere automáticamente dicho nombre a partir de la ruta del modelo especificada.

### 3.1 Sintaxis del Comando

```powershell
python evaluate.py [--model-path <ruta_modelo.zip>] [--label <identificador>] [--c0] [--vecnormalize <ruta_stats.pkl>] [--dataset <ruta_dataset.json>] [--output-dir <directorio_salida>] [--plot]
```

### 3.2 Ejemplos de Evaluación

```powershell
# 1. Evaluación de modelo PPO indicando etiqueta de identificación
python evaluate.py --model-path experiments/phase_D_prueba/PPO_2/seed_42/models/best_model/best_model.zip --label PPO_2

# 2. Evaluación de modelo SAC con estadísticas explícitas de normalización y generación inmediata de figuras
python evaluate.py --model-path experiments/phase_D_prueba/SAC_1/seed_42/models/best_model/best_model.zip --label SAC_1 --vecnormalize experiments/phase_D_prueba/SAC_1/seed_42/models/best_model/vecnormalize_stats.pkl --plot

# 3. Evaluación del baseline analítico C0 sobre el dataset de validación
python evaluate.py --c0 --label C0 --dataset data/datasets/validation_set_200.json
```

### 3.3 Estructura de Datos de Salida

Cada evaluación almacena los registros tabulares en `reports/data/evaluation/<label>/`:

* `episodes.csv`: Telemetría por episodio con las métricas de estabilización (error cartesiano terminal e_ss, velocidad residual v_ss, tiempo de establecimiento t_s, esfuerzo de torque J_u), coordenadas de meta (target_x, target_y) y el desglose de costos cuadráticos (j_pos, j_vel, j_ctrl, j_lqr).
* `summary.json`: Métricas estadísticas agregadas (media, desviación estándar, percentiles 50 y 90) e intervalo de confianza binomial de Wilson al 95%.
* `sample_trajectories.csv`: Registro paso a paso de las trayectorias temporales (posiciones articulares, velocidades, errores cartesianos y torques aplicados) en episodios de muestra.

---

## 4. Generación Modular de Gráficas (`plot.py`)

El script [plot.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/plot.py) procesa de forma desacoplada los archivos tabulares generados en `reports/data/evaluation/` y los registros de eventos de TensorBoard. Las gráficas se generan de manera independiente en imágenes individuales y se clasifican en subdirectorios dedicados por modelo, además de directorios para combinaciones comparativas específicas.

### 4.1 Sintaxis del Comando

```powershell
python plot.py [--model <nombre_modelo>] [--compare <modelo1,modelo2,...>] [--eval-log <ruta_evaluations.npz>] [--data-dir <directorio_datos>] [--out-dir <directorio_salida>]
```

### 4.2 Ejemplos de Ejecución

```powershell
# 1. Generación completa para todos los modelos evaluados (la comparativa se guarda en reports/figures/C0_PPO_2/ si existen C0 y PPO_2)
python plot.py

# 2. Generación exclusiva para un modelo específico (ej. PPO_2)
python plot.py --model PPO_2

# 3. Comparativa combinada exclusiva entre modelos específicos (ej. C0 y PPO_2 o PPO_1, PPO_2 y SAC_1)
python plot.py --compare C0,PPO_2
python plot.py --compare PPO_1,PPO_2,SAC_1

python plot.py --files reports/data/evaluation/PPO_2_episodes.csv,reports/data/evaluation/C0_episodes.csv
```

Tanto la ejecución por defecto como el uso explícito de `--compare` nombran el directorio comparativo a partir de la combinación de los modelos contrastados (por ejemplo, `reports/figures/C0_PPO_2/` o `reports/figures/PPO_1_PPO_2_SAC_1/`), suprimiendo carpetas genéricas redundantes.

### 4.3 Organización de las Figuras Generadas

#### Subcarpeta por Modelo (`reports/figures/<modelo>/`)

* **Métricas Mecatrónicas de Estabilización y Control:**
  * `success_rate.png`: Tasa de estabilización terminal con intervalo de confianza de Wilson al 95%.
  * `steady_state_error.png`: Distribución del error cartesiano terminal e_ss con tolerancia r_target = 20 mm.
  * `residual_velocity.png`: Distribución de la velocidad angular residual v_ss con cota v_tol = 0.10 rad/s.
  * `settling_time.png`: Distribución del tiempo de establecimiento transitorio t_s.
  * `control_effort.png`: Gasto energético acumulado por esfuerzo de torque J_u.
  * `reward_components.png`: Evolución de las tres curvas de costo cuadrático LQR (posición J_pos, velocidad J_vel y control J_ctrl).
  * `temporal_trajectory_episode.png`: Respuesta temporal de 3 paneles (error cartesiano ||e(t)||, variables articulares q(t) y torques de control tau(t) frente a límites de saturación +-0.05 N*m).
  * `pareto_tradeoff.png`: Diagrama de dispersión bi-objetivo (e_ss vs J_u) con la frontera empírica de Pareto.
  * `workspace_spatial_error.png`: Distribución espacial del error cartesiano en el plano de trabajo (x, y) con límites cinemáticos del manipulador.

* **Curvas de Aprendizaje y Optimización:**
  * `learning_curve_return.png`: Retorno episódico medio a lo largo del entrenamiento (con media y banda de dispersión si existen varias semillas).
  * `learning_curve_success_rate.png`: Tasa de éxito durante el entrenamiento.
  * `multiseed_learning_curves.png`: Panel comparativo multisemilla con bandas de desviación estadística.
  * `optimizer_total_loss.png`: Curva estática de la pérdida total del optimizador.
  * `optimizer_policy_loss.png`: Pérdida del gradiente de la política (clipped loss).
  * `optimizer_value_loss.png`: Pérdida cuadrática de la función de valor (crítico).
  * `optimizer_entropy.png`: Evolución temporal de la entropía de la política.
  * `optimizer_approx_kl.png`: Divergencia de Kullback-Leibler aproximada.
  * `optimizer_explained_variance.png`: Varianza explicada de la función de valor.
  * `optimizer_convergence_summary.png`: Panel resumen 2x3 con todas las curvas de pérdida del optimizador.

#### Subcarpeta Comparativa (`reports/figures/<modelo1>_<modelo2>_.../`)

* `comparative_success_rate.png`: Comparación de la tasa de éxito entre los modelos contrastados.
* `comparative_steady_state_error.png`: Comparativa de precisión cartesiana terminal e_ss.
* `comparative_residual_velocity.png`: Comparativa de disipación de velocidad angular residual v_ss.
* `comparative_settling_time.png`: Comparativa de rapidez de respuesta temporal t_s.
* `comparative_control_effort.png`: Comparativa del gasto de control acumulado J_u.
* `comparative_reward_components.png`: Comparativa del desglose de costos cuadráticos en barras agrupadas.
* `comparative_pareto_tradeoff.png`: Dispersión de episodios y centroides en el espacio bi-objetivo e_ss vs J_u.
* `comparative_spatial_error.png`: Subplots espaciales comparativos del error en el espacio de trabajo.
* `comparative_temporal_trajectories.png`: Superposición de respuestas temporales de los modelos sobre un mismo episodio.

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

---

## 7. Generación de la Matriz de Benchmarking (`generate_benchmark_matrix.py`)

Para consolidar en una única tabla todos los hiperparámetros del entorno físico, algoritmos de control y las métricas cinemáticas y dinámicas calculadas durante las evaluaciones, se provee el script [generate_benchmark_matrix.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/generate_benchmark_matrix.py):

```powershell
python generate_benchmark_matrix.py
```

El script explora recursivamente todos los archivos `.yaml` en `configs/`, extrae los parámetros de preprocesamiento, redes y optimización, y vincula automáticamente las métricas de `reports/data/evaluation/<modelo>/summary.json`. La matriz resultante se exporta como archivo delimitado por comas en [reports/data/benchmark_matrix.csv](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/reports/data/benchmark_matrix.csv).

