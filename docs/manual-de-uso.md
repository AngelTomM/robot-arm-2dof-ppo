# Manual de Operación y Uso — Manipulador Planar 2-DOF (Fase D: MuJoCo Reacher-v5)

Guía operativa para la instalación, entrenamiento, evaluación mecatrónica, generación de gráficos y simulación interactiva 3D sobre el manipulador planar 2-DOF accionado por torque. Para el fundamento teórico y la formulación matemática de los costos cuadráticos y la cinemática, remítase a [docs/metodologia.md](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/docs/metodologia.md).

---

## 1. Configuración del Entorno de Ejecución

### 1.1 Requisitos del Sistema

* Python 3.11 o versión superior compatible con librerías de MuJoCo.
* Entorno virtual activo con acceso a controladores de aceleración gráfica OpenGL.

### 1.2 Activación del Entorno Virtual e Instalación de Dependencias

```powershell
# 1. Activación del entorno virtual (Windows PowerShell)
..\.venv\Scripts\Activate.ps1

# 2. Instalación de paquetes requeridos
pip install -r requirements.txt
```

Las dependencias principales contemplan `gymnasium[mujoco]>=1.0.0`, `mujoco>=2.3.3`, `imageio>=2.14.1`, `stable-baselines3>=2.0.0`, `torch>=2.0.0`, `matplotlib>=3.8.0`, `pandas>=2.1.0` y `pytest>=8.0.0`.

---

## 2. Taxonomía Modular y Fuente Única de Verdad (SSOT)

El repositorio articula sus etapas bajo una taxonomía homogénea y desacoplada estructurada en carpetas temáticas o de presupuesto computacional (secciones):

```text
configs/<seccion>/<modelo>.yaml
experiments/<seccion>/<modelo>/seed_<semilla>/
reports/data/evaluation/<eval_set>/<seccion>/<modelo>/
reports/<fig_partition>/<seccion>/<modelo>/
```

### 2.1 Secciones Canónicas

* `entorno/`: Modelos de calibración base de física y baseline cinemático C0 (`C0`, `PPO_1` a `PPO_7`, `SAC_1`, `SAC_2`).
* `entorno_V2/`: Variaciones paramétricas de recompensa y dinámicas del entorno (`PPO_7.2` a `PPO_7.12`, `PPO_7.11_500`).
* `NN/`: Ablación de arquitecturas de red neuronal y funciones de activación (`PPO_S`, `PPO_L`, `PPO_M_RELU`, `PPO_M_TANH`, `PPO_M_LEAKYRELU`).
* `PPO_eval_300k/`: Evaluaciones PPO con horizonte estándar de 300 000 timesteps (`PPO_NOM300`, `PPO_MYO`, `PPO_EXP_H`, `PPO_EXP_L`, `PPO_FAR`, `PPO_STG`, `PPO_AGR`).
* `PPO_eval_500k/`: Evaluaciones PPO con horizonte intermedio de 500 000 timesteps (`PPO_NOM`, `PPO_NOM_LRelu`, `PPO_15`, `PPO_17`, `PPO_17_E`, `PPO_10_todo` a `PPO_18`).
* `PPO_eval_1M/`: Evaluaciones PPO de alta convergencia a 1 000 000 de pasos (`PPO_NOM_1M`, `PPO_15_1M`, `PPO_17_1M`, `PPO_17_2M`).
* `SAC_eval_300k/`: Evaluaciones SAC a 300 000 timesteps (`SAC_NOM`, `SAC_4`, `SAC_4s`, `SAC_BUF_L`, `SAC_BUF_S`, `SAC_EXP_LOW`, `SAC_EXP_H`, `SAC_NOM_NREWARD`).
* `SAC_eval_500k/`: Configuraciones SAC de horizonte extendido.
* `smoke/`: Pruebas de integración rápida de bajo presupuesto temporal (`PPO_smoke`, `SAC_smoke`).

En los archivos YAML, el nombre del modelo está dado por el nombre del archivo sin extensión (`.stem`) y la sección por el nombre de la carpeta contenedora (`.parent.name`). El bloque `experiment:` contiene exclusivamente `description:`.

---

## 3. Entrenamiento de Agentes (`train.py`)

El script en la raíz [train.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/train.py) centraliza la inicialización, vectorización y entrenamiento de los agentes bajo el proceso de decisión de Markov continuo (MDP):

* **Resolución de Hiperparámetros (SSOT):** Fusiona automáticamente los parámetros físicos invariantes de [configs/environment.yaml](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/configs/environment.yaml) con la configuración específica del algoritmo de control.
* **Selección de Algoritmo:** Detecta automáticamente si la configuración define hiperparámetros para PPO (`ppo:`) o SAC (`sac:`).
* **Normalización Acoplada:** Utiliza `VecNormalize` sobre observaciones cinemáticas y dinámicas crudas preservando las escalas de la función de costo LQR continua (`norm_reward: false`).

### 3.1 Sintaxis del Comando y Parámetros

```powershell
python train.py --config <ruta_yaml> [--seed <int|lista>] [--device <str>] [--total_timesteps <int>] [--checkpoint_freq <int>] [--resume [ruta_o_auto]] [--env-config <ruta_yaml>]
```

Comportamiento de los argumentos y valores por defecto:

* `--config` (requerido): Ruta al archivo YAML del experimento (por ejemplo, `configs/PPO_eval_500k/PPO_17.yaml`).
* `--seed` (por defecto: `42`): Semilla pseudoaleatoria. Admite enteros individuales o listas separadas por comas (ej. `"42, 100, 2026"`), ejecutando entrenamientos secuenciales de todas las semillas en una única orden.
* `--device` (por defecto: `"auto"`): Dispositivo de cómputo para los tensores de PyTorch (`cuda` o `cpu`).
* `--total_timesteps` (por defecto: `None`): Presupuesto total de pasos de interacción en el entorno. Si se omite, adopta el valor de `training.total_timesteps` en el YAML.
* `--checkpoint_freq` (por defecto: `None`): Frecuencia de serialización periódica de puntos de control. Si se omite, adopta `training.checkpoint_freq` del YAML (típicamente 50 000 timesteps).
* `--resume` (opcional, por defecto: `None`): Reanuda el entrenamiento desde un punto de control. Sin argumentos o con `"auto"`, localiza de manera autónoma el último punto de control (`ckpt_<paso>`) en `experiments/<seccion>/<modelo>/seed_<semilla>/checkpoints/`. Admite una ruta explícita a un directorio de checkpoint o a su `model.zip`.
* `--env-config` (por defecto: `None`): Ruta al archivo de configuración del entorno físico común. Por defecto, carga [configs/environment.yaml](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/configs/environment.yaml).

### 3.2 Comandos de Ejecución Estándar

```powershell
# Entrenamiento estándar (seed 42)
python train.py --config configs/PPO_eval_500k/PPO_17.yaml --seed 42

# Reanudación automática desde el último checkpoint registrado
python train.py --config configs/PPO_eval_500k/PPO_17.yaml --resume

# Reanudación manual especificando la ruta al checkpoint deseado
python train.py --config configs/PPO_eval_500k/PPO_17.yaml --resume experiments/PPO_eval_500k/PPO_17/seed_42/checkpoints/ckpt_250000

# Extensión desacoplada de presupuesto (Warm-Start de 500k a 1M pasos)
python train.py --config configs/PPO_eval_1M/PPO_17_1M.yaml --resume experiments/PPO_eval_500k/PPO_17/seed_42/checkpoints/ckpt_500000

# Entrenamiento multisemilla secuencial en una sola instrucción
python train.py --config configs/PPO_eval_500k/PPO_NOM.yaml --seed "42, 100, 2026"

# Entrenamiento SAC (seed 42)
python train.py --config configs/SAC_eval_300k/SAC_NOM.yaml --seed 42

# Ejecución secuencial de múltiples entrenamientos encadenados en una sola línea mediante el operador ';'
python train.py --config configs/SAC_eval_300k/SAC_NOM.yaml --seed 42; python train.py --config configs/SAC_eval_300k/SAC_BUF_S.yaml --seed 42

# Pruebas de integración rápida (Smoke tests, 256 timesteps)
python train.py --config configs/smoke/PPO_smoke.yaml --total_timesteps 256
python train.py --config configs/smoke/SAC_smoke.yaml --total_timesteps 256
```

### 3.3 Estructura de Artefactos de Salida de Entrenamiento

Cada sesión de entrenamiento genera su estructura en `experiments/<seccion>/<modelo>/seed_<semilla>/`:

* `models/<algoritmo>_arm2d_final.zip`: Red neuronal con los pesos correspondientes al último paso de optimización.
* `models/best_model/`: Checkpoint con el mayor retorno medio alcanzado durante las evaluaciones periódicas (`best_model.zip` junto a `vecnormalize_stats.pkl`).
* `checkpoints/ckpt_<paso>/`: Puntos de control periódicos.
* `logs/`: Registros de eventos para TensorBoard y archivo de validación periódica `evaluations.npz`.

### 3.4 Monitoreo con TensorBoard

```powershell
# Iniciar servidor web de monitoreo apuntando al directorio raíz de experimentos
tensorboard --logdir experiments/

# Especificar puerto alternativo
tensorboard --logdir experiments/ --port 6007
```

---

## 4. Evaluación Mecatrónica y Benchmarking (`evaluate.py`)

El script [evaluate.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/evaluate.py) ejecuta la evaluación determinista de políticas de aprendizaje por refuerzo o del controlador cinemático analítico C0 sobre bancos congelados de trayectorias.

### 4.1 Conjuntos de Datos Disponibles (Datasets Congelados)

* **Conjunto de Validación Intermedia ([datasets/validation_set_200.json](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/datasets/validation_set_200.json)):** Contiene 200 episodios generados con semilla pseudoaleatoria 202601. Se emplea para comparar y calibrar hiperparámetros durante el desarrollo.
* **Conjunto de Evaluación Final de Referencia ([datasets/final_eval_set_300.json](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/datasets/final_eval_set_300.json)):** Contiene 300 episodios generados con semilla pseudoaleatoria 202602. Actúa como conjunto de prueba desacoplado (*test set*) para certificar formalmente las métricas definitivas del benchmark mecatrónico.

### 4.2 Sintaxis del Comando y Parámetros

```powershell
python evaluate.py [--model-path <ruta_modelo.zip>] [--section <nombre_seccion>] [--label <identificador>] [--c0] [--vecnormalize <ruta_stats.pkl>] [--dataset <200|300|all|ruta_json>] [--output-dir <directorio_salida>] [--mode <deterministic|stochastic>] [--plot]
```

Comportamiento de los argumentos y valores por defecto:

* `--model-path` (por defecto: `None`): Ruta al archivo `.zip` del modelo entrenado.
* `--section`, `--phase` (opcional, por defecto: `None`): Subcarpeta que clasifica al modelo (ej. `PPO_eval_500k`, `entorno`). Si se omite, se deduce automáticamente a partir de la ruta del modelo o de su definición en `configs/`.
* `--label` (por defecto: automático): Nombre identificatorio del modelo. Si se omite, se deduce automáticamente analizando la ruta del modelo: al detectar la carpeta `seed_<semilla>`, asigna por defecto el formato estandarizado `<modelo>_seed_<semilla>` (por ejemplo, `PPO_NOM_seed_42`), manteniendo concordancia con la carpeta de experimentos y evitando sobreescrituras en las evaluaciones. Solo es necesario especificar `--label` cuando se desee forzar un identificador personalizado.
* `--c0` (por defecto: `False`): Bandera para evaluar el controlador cinemático analítico con transpuesta de Jacobiano amortiguada. Admite modificar ganancias mediante `--kp` (por defecto: `50.0`), `--kd` (por defecto: `0.05`) y `--lambda-damping` (por defecto: `0.01`).
* `--dataset` (por defecto: `"datasets/validation_set_200.json"`): Especificación del conjunto de prueba. Admite `"all"` (ejecuta consecutivamente 200 y 300), `"200"` o `"300"`.
* `--vecnormalize` (por defecto: `None`): Ruta a `vecnormalize_stats.pkl`. Si se omite, se localiza automáticamente en el directorio del modelo.
* `--output-dir` (por defecto: `"reports/data/evaluation"`): Directorio base de evaluación.
* `--mode` (por defecto: `"deterministic"`): Modo de inferencia de la política (`deterministic` o `stochastic`).
* `--plot` (por defecto: `False`): Dispara de forma automática el generador gráfico tras concluir la evaluación numérica.

### 4.3 Ejemplos de Evaluación

```powershell
# 1. Evaluación sobre benchmark definitivo de 300 episodios (etiqueta automática: PPO_17_seed_42)
python evaluate.py --model-path experiments/PPO_eval_500k/PPO_17/seed_42/models/best_model/best_model.zip --dataset 300

# 2. Evaluación encadenada de múltiples semillas en PowerShell (asigna automáticamente PPO_NOM_seed_42, PPO_NOM_seed_100, etc.)
42, 100, 2026 | ForEach-Object {
    python evaluate.py --model-path "experiments/PPO_eval_500k/PPO_NOM/seed_$_/models/best_model/best_model.zip" --dataset 300
}

# 3. Forzar explícitamente un identificador personalizado con --label
python evaluate.py --model-path experiments/PPO_eval_500k/PPO_NOM/seed_42/models/best_model/best_model.zip --label PPO_NOM_nominal --dataset 300

# 4. Evaluación del controlador analítico C0 sobre 300 episodios (etiqueta automática: C0)
python evaluate.py --c0 --dataset 300

# 5. Evaluación secuencial completa sobre 200 y 300 episodios con generación inmediata de figuras
python evaluate.py --model-path experiments/PPO_eval_500k/PPO_NOM/seed_42/models/best_model/best_model.zip --dataset all --plot

# 6. Evaluación encadenada de múltiples modelos independientes en una sola línea mediante ';'
python evaluate.py --model-path experiments/SAC_eval_300k/SAC_NOM/seed_42/models/best_model/best_model.zip --dataset 300 ; python evaluate.py --model-path experiments/SAC_eval_300k/SAC_BUF_S/seed_42/models/best_model/best_model.zip --dataset 300
```

### 4.4 Estructura de Salida Unificada (SSOT)

Cada evaluación almacena sus resultados en un destino único correspondiente a la sección y al identificador provisto en `--label` o deducido, eliminando redundancias en la raíz:

```text
reports/data/evaluation/<eval_set>/<seccion>/<modelo_o_semilla>/
├── episodes.csv              # Métricas de estabilización cartesiana por episodio
├── summary.json              # Consolidado estadístico e intervalos de Wilson
└── sample_trajectories.csv   # Trayectorias temporales paso a paso (si aplica)
```

Por ejemplo, la evaluación de tres semillas para `PPO_NOM` genera independientemente:

* `reports/data/evaluation/eval_300/PPO_eval_500k/PPO_NOM_seed_42/`
* `reports/data/evaluation/eval_300/PPO_eval_500k/PPO_NOM_seed_100/`
* `reports/data/evaluation/eval_300/PPO_eval_500k/PPO_NOM_seed_2026/`

---

## 5. Generación Modular de Gráficas (`plot.py`)

El script [plot.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/plot.py) procesa de forma desacoplada los datos tabulares evaluados en `reports/data/evaluation/` y los registros de telemetría de TensorBoard.

* **Filtrado Automático de Sesiones:** Detecta si existen múltiples subcarpetas de eventos para una misma semilla (ej. ejecuciones abortadas o reiniciadas) y selecciona la sesión canónica correspondiente a la corrida completa y evaluada.
* **Agregación Multisemilla:** Si un modelo cuenta con múltiples semillas, grafica la media muestral $\mu$ con una banda sombreada de dispersión a una desviación estándar ($\pm 1\sigma$).

### 5.1 Sintaxis del Comando y Parámetros

```powershell
python plot.py [--model <[seccion/]nombre_modelo>] [--section <seccion>] [--compare <m1,m2,...>] [--dataset <200|300|all>] [--metric <metrica1,metrica2,...>] [--data-dir <directorio_datos>] [--out-dir <directorio_salida>] [--strict]
```

Comportamiento de los argumentos principales:

* `--data-dir` (por defecto: `"reports/data/evaluation"`): Directorio base que contiene los datos tabulares. Explora recursivamente subdirectorios jerárquicos (`eval_*/<seccion>/<modelo>/`).
* `--out-dir` (por defecto: `"reports"`): Directorio base para la salida gráfica. Se enruta automáticamente a `reports/figures_valid_200/` y `reports/figures_final_300/`.
* `--dataset` (por defecto: `None`): Filtro de partición (`200`, `300`, `eval_200` o `eval_300`). Si se omite, procesa las particiones disponibles.
* `--model` (por defecto: `None`): Genera figuras para el modelo indicado. Admite rutas calificadas por sección (`--model <seccion>/<modelo>`, ej. `--model PPO_eval_500k/PPO_7`) o nombres simples (`--model PPO_7`).
* `--section` (por defecto: `None`): Especifica explícitamente la sección o fase del modelo para desambiguar homónimos entre distintas carpetas (ej. `--section PPO_eval_500k`).
* `--compare` (por defecto: `None`): Lista de modelos separados por comas para generar figuras comparativas consolidadas. Admite modelos simples o calificados por sección (ej. `--compare entorno/PPO_7,PPO_eval_500k/PPO_7`).
* `--metric`, `--metrics` (por defecto: `None`): Selecciona métricas específicas o macros:
  * **Optimizador (TensorBoard):** `explained_variance` (`ev`), `approx_kl` (`kl`), `value_loss` (`vl`), `policy_loss` (`pl`), `entropy`, `loss`, `optimizer_summary`.
  * **Curvas de Aprendizaje:** `learning_return` (`return`), `learning_success_rate` (`learning_sr`), `learning_curves`.
  * **Benchmark de Evaluación:** `success_rate`, `steady_state_error` (`e_ss`), `residual_velocity` (`v_ss`), `settling_time` (`t_s`), `control_effort` (`j_u`), `reward_components`, `pareto_tradeoff`, `spatial_error`, `temporal_trajectories`.
  * **Macros:** `optimizer` (todas las del optimizador), `eval` (las 9 de benchmark), `all` (suite completa).
* `--strict` (por defecto: `False`): Exige consistencia muestral homogénea al comparar modelos entre particiones distintas.

### 5.2 Organización de las Figuras de Salida

Las figuras se organizan en correspondencia con el árbol de secciones:

1. **Figuras individuales por modelo:**
   `reports/<particion_figuras>/<seccion>/<modelo>/`
   *(Ejemplo: `reports/figures_final_300/PPO_eval_500k/PPO_17/`)*

2. **Figuras comparativas intra-sección (modelos de la misma categoría):**
   `reports/<particion_figuras>/<seccion>/<modelo1>_<modelo2>/`
   *(Ejemplo: `reports/figures_final_300/PPO_eval_500k/PPO_17_PPO_15/`)*

3. **Figuras comparativas inter-sección (modelos de categorías distintas o calificados):**
   `reports/<particion_figuras>/comparisons/<modelo1>_<modelo2>/`
   *(Ejemplo: `reports/figures_final_300/comparisons/entorno_PPO_7_PPO_eval_500k_PPO_17/`)*

### 5.3 Ejemplos de Ejecución Gráfica

```powershell
# 1. Generación de figuras individuales para un modelo único
python plot.py --dataset 300 --model PPO_17

# 2. Generación con calificación por sección ante posibles homónimos
python plot.py --dataset 300 --model PPO_eval_500k/PPO_7 --metric explained_variance
python plot.py --dataset 300 --model PPO_7 --section entorno --metric steady_state_error

# 3. Comparativa intra-sección (salida en reports/figures_final_300/PPO_eval_500k/PPO_17_PPO_15/)
python plot.py --dataset 300 --compare PPO_17,PPO_15 --metric steady_state_error,pareto_tradeoff

# 4. Comparativa inter-sección con desambiguación calificada
python plot.py --dataset 300 --compare entorno/PPO_7,PPO_eval_500k/PPO_17 --metric steady_state_error

# 5. Comparativa de homónimos entre distintas secciones
python plot.py --dataset 300 --compare entorno/PPO_7,PPO_eval_500k/PPO_7 --metric e_ss,control_effort

# 6. Comparativa de curvas del optimizador (panel 2x3)
python plot.py --dataset 300 --compare PPO_17,PPO_15 --metric optimizer_summary

# 7. Comparativa entre evaluaciones de semillas individuales de un mismo modelo
python plot.py --dataset 300 --compare PPO_eval_500k/PPO_NOM_seed_42,PPO_eval_500k/PPO_NOM_seed_100 --metric steady_state_error

# 8. Flujo integrado continuo (Evaluación -> Generación gráfica) encadenado mediante ';'
python evaluate.py --model-path experiments/PPO_eval_500k/PPO_NOM/seed_42/models/best_model/best_model.zip --dataset 300 ; python plot.py --dataset 300 --model PPO_eval_500k/PPO_NOM_seed_42 --metric eval
```

---

## 6. Simulación Visual Interactiva 3D (`interactive_render.py`)

El visualizador [interactive_render.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/interactive_render.py) interactúa con la ventana nativa OpenGL de MuJoCo en 3D para inspección cualitativa del movimiento del manipulador:

```powershell
# Visualización de un modelo entrenado de RL
python interactive_render.py experiments/PPO_eval_500k/PPO_17/seed_42/models/best_model/best_model.zip

# Visualización del controlador cartesiano analítico C0 durante 10 episodios
python interactive_render.py --c0 --episodes 10
```

---

## 7. Ejecución de Pruebas Unitarias

La suite de validación asegura la coherencia dimensional, cinemática y algorítmica del sistema mediante `pytest`:

```powershell
python -m pytest tests/
```

Módulos de prueba verificados:

* [tests/test_config.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/tests/test_config.py): Carga desacoplada de configuraciones y verificación de parámetros SSOT de MuJoCo Reacher-v5.
* [tests/test_kinematics.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/tests/test_kinematics.py): Validación de cinemática directa y matriz Jacobiana contrastada contra las librerías cinemáticas de MuJoCo.
* [tests/test_normalization.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/tests/test_normalization.py): Normalización vectorial `VecNormalize`, cálculo de recompensa continua LQR y truncamiento del horizonte de simulación.
* [tests/test_plot_metrics.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/tests/test_plot_metrics.py): Resolución canónica y alias de métricas modulares para graficación.

---

## 8. Generación de la Matriz de Benchmarking (`generate_benchmark_matrix.py`)

Para consolidar en una tabla estructurada todos los hiperparámetros del entorno físico, los algoritmos de control y las métricas cinemáticas y dinámicas calculadas durante las evaluaciones:

```powershell
python generate_benchmark_matrix.py
```

El script explora recursivamente todos los archivos `.yaml` en `configs/`, extrae los parámetros de preprocesamiento, redes y optimización, y vincula automáticamente las métricas de evaluación almacenadas en `reports/data/evaluation/eval_300/` (prioritario) o `eval_200/`. La matriz resultante se exporta a [reports/data/benchmark_matrix.csv](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/reports/data/benchmark_matrix.csv).
