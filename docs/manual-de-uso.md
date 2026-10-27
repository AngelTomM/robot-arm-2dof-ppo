# Manual de Operación y Uso — Manipulador Planar 2-DOF (Fase D: MuJoCo Reacher-v5)

Guía operativa para la instalación, entrenamiento, evaluación mecatrónica, simulación interactiva 3D y ejecución de pruebas unitarias sobre el manipulador planar 2-DOF accionado por torque. Para el fundamento teórico y la formulación matemática de los costos cuadráticos y la cinemática, remítase a [docs/metodologia.md](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/docs/metodologia.md).

---

## 1. Configuración del Entorno de Ejecución

### 1.1 Requisitos del Sistema

* Python 3.11 (o versión superior compatible con librerías de MuJoCo).
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

## 2. Entrenamiento de Agentes (`train.py`)

El script en la raíz [train.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/train.py) centraliza la inicialización, vectorización y entrenamiento de los agentes bajo el proceso de decisión de Markov continuo (MDP):

* **Resolución de Hiperparámetros (SSOT):** Fusiona automáticamente los parámetros físicos invariantes de [configs/environment.yaml](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/configs/environment.yaml) con la configuración específica del algoritmo de control.
* **Selección de Algoritmo:** Detecta automáticamente si la configuración define hiperparámetros para PPO (`ppo:`) o SAC (`sac:`).
* **Normalización Acoplada:** Utiliza `VecNormalize` sobre observaciones cinemáticas y dinámicas crudas preservando las escalas de la función de costo LQR continua (`norm_reward: false`).

### 2.1 Sintaxis del Comando y Parámetros

```powershell
python train.py --config <ruta_yaml> [--seed <int|lista>] [--device <str>] [--total_timesteps <int>] [--checkpoint_freq <int>] [--resume [ruta_o_auto]] [--env-config <ruta_yaml>]
```

Comportamiento de los argumentos y valores por defecto:

* `--config` (requerido): Ruta al archivo YAML del experimento (por ejemplo, `configs/entorno/PPO_1.yaml`). Determina el algoritmo, la arquitectura de la red y los pesos de recompensa.
* `--seed` (por defecto: `42`): Semilla para los generadores pseudoaleatorios de NumPy, PyTorch y Gym. Admite una lista de enteros separados por comas (por ejemplo, `"42, 100, 2026"`), ejecutando sesiones de entrenamiento secuenciales bajo idénticas condiciones.
* `--device` (por defecto: `"auto"`): Dispositivo de cómputo para los tensores de PyTorch. Selecciona aceleración por GPU (`cuda`) si está disponible; de lo contrario, asigna `cpu`.
* `--total_timesteps` (por defecto: `None`): Presupuesto total de pasos de interacción en el entorno. Si se omite, adopta el valor especificado en el YAML bajo la clave `train.total_timesteps` (fijado típicamente en 300.000 timesteps).
* `--checkpoint_freq` (por defecto: `None`): Intervalo de pasos para serializar puntos de control periódicos. Si se omite, adopta el valor de `train.checkpoint_freq` definido en el archivo YAML (típicamente cada 50.000 timesteps).
* `--resume` (opcional, por defecto: `None`): Reanuda el entrenamiento de la política desde un punto de control (*checkpoint*). Si se invoca sin argumentos adicionales (`--resume`) o con el valor `"auto"`, detecta de manera autónoma el último punto de control válido (`ckpt_<paso>`) disponible en el directorio `experiments/<fase>/<exp_id>/seed_<seed>/checkpoints/`. Alternativamente, admite una ruta explícita a una carpeta de punto de control o a su archivo `model.zip`. Restablece los pesos de la red neuronal, las estadísticas acumuladas de normalización de observaciones (`vecnormalize_stats.pkl`), la telemetría acumulada en TensorBoard y calcula los timesteps remanentes para completar el presupuesto fijado.
* `--env-config` (por defecto: `None`): Ruta al archivo de configuración del entorno físico común. Si se omite, utiliza la referencia interna indicada en el archivo YAML o carga directamente [configs/environment.yaml](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/configs/environment.yaml).

### 2.2 Comandos de Ejecución Estándar

```powershell
# Entrenamiento con una única semilla (300.000 timesteps, seed 42)
python train.py --config configs/entorno/PPO_1.yaml --seed 42

# Reanudación automática desde el último checkpoint registrado (detecta automáticamente ckpt_<paso>)
python train.py --config configs/entorno/PPO_1.yaml --resume

# Reanudación manual especificando la ruta al checkpoint deseado
python train.py --config configs/entorno/PPO_1.yaml --resume experiments/entorno/PPO_1/seed_42/checkpoints/ckpt_50000

# Extensión desacoplada de presupuesto (Warm-Start): Reanudar desde el checkpoint de un experimento previo
# hacia una nueva configuración con mayor presupuesto (ej. de 500.000 a 1.000.000 pasos) preservando ambos modelos
python train.py --config configs/PPO_eval/PPO_17_1M.yaml --resume experiments/PPO_eval/PPO_17/seed_42/checkpoints/ckpt_500000

# Entrenamiento multisemilla secuencial en una sola instrucción (semillas 42, 100 y 2026)
python train.py --config configs/entorno/PPO_1.yaml --seed "42, 100, 2026"

# Entrenamiento completo de SAC_1 (300.000 timesteps, seed 42)
python train.py --config configs/entorno/SAC_1.yaml --seed 42

# Pruebas de integración rápida (Smoke tests, 256 timesteps)
python train.py --config configs/smoke/PPO_smoke.yaml --total_timesteps 256
python train.py --config configs/smoke/SAC_smoke.yaml --total_timesteps 256
```

### 2.3 Extensión Desacoplada de Presupuesto (*Warm-Start*)

Cuando se requiere evaluar el impacto de un mayor horizonte de optimización (por ejemplo, extender el entrenamiento de 500.000 a 1.000.000 de pasos) sin incurrir nuevamente en el costo temporal de las iteraciones iniciales, se emplea la técnica de reanudación desacoplada:

1. **Configuración del nuevo experimento**: Se crea un archivo YAML derivado (por ejemplo, `configs/PPO_eval/PPO_17_1M.yaml`) especificando un nuevo identificador en `experiment.name: PPO_17_1M` y fijando el presupuesto objetivo en `training.total_timesteps: 1000000`.
2. **Ejecución con enlace al punto de control**: Se invoca [train.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/train.py) suministrando la ruta explícita al punto de control del modelo base:
   ```powershell
   python train.py --config configs/PPO_eval/PPO_17_1M.yaml --resume experiments/PPO_eval/PPO_17/seed_42/checkpoints/ckpt_500000
   ```
3. **Mecanismo de preservación**: El script carga los pesos y las estadísticas de `VecNormalize` acumuladas en el paso 500.000, calcula automáticamente las iteraciones remanentes ($1.000.000 - 500.000 = 500.000$ pasos) y deposita los nuevos puntos de control y telemetría en `experiments/PPO_eval/PPO_17_1M/seed_42/`. El modelo original (`PPO_17`) permanece inalterado, permitiendo ejecutar comparativas directas mediante:
   ```powershell
   python plot.py --dataset 300 --compare PPO_17,PPO_17_1M --metric steady_state_error,success_rate,learning_curves
   ```

### 2.4 Estructura de Artefactos de Salida

Cada sesión de entrenamiento genera un directorio estructurado en `experiments/<fase>/<experimento>/seed_<semilla>/`:

* `models/<algoritmo>_arm2d_final.zip`: Red neuronal con los pesos correspondientes al último paso de optimización.
* `models/best_model/`: Checkpoint con el mayor retorno medio alcanzado durante las evaluaciones periódicas (`best_model.zip` junto al archivo serializado de normalización `vecnormalize_stats.pkl`).
* `checkpoints/ckpt_<paso>/`: Puntos de control intermedios para auditoría de estabilidad temporal.
* `logs/`: Registros continuos de eventos para TensorBoard y archivo de validación periódica `evaluations.npz`.

### 2.5 Monitoreo de Telemetría en Tiempo Real con TensorBoard

El entrenamiento genera eventos de telemetría continuos procesables en tiempo real mediante la interfaz web de TensorBoard:

```powershell
# Iniciar servidor web de monitoreo apuntando al directorio raíz de experimentos
tensorboard --logdir experiments/

# Especificar puerto alternativo si el puerto por defecto (6006) estuviese ocupado
tensorboard --logdir experiments/ --port 6007
```

Una vez iniciado el servicio, acceda desde el navegador a `http://localhost:6006/`. La interfaz organiza las siguientes series temporales:

* **Métricas de Interacción (Rollout):**
  * `rollout/ep_rew_mean`: Retorno escalar acumulado medio por episodio.
  * `rollout/ep_len_mean`: Longitud media de los episodios en pasos de simulación.
* **Métricas de Validación Periódica (EvalCallback):**
  * `eval/mean_reward`: Retorno medio obtenido en episodios de validación periódicos con política determinista.
  * `eval/success_rate`: Tasa empírica de estabilización alcanzada en validación intermedia.
* **Métricas de Optimización (Train):**
  * `train/loss`: Función de pérdida total agregada.
  * `train/policy_gradient_loss`: Pérdida de la política (*clipped surrogate loss* en PPO / *actor loss* en SAC).
  * `train/value_loss` o `train/critic_loss`: Pérdida cuadrática de la función de valor del crítico.
  * `train/entropy_loss`: Nivel de exploración estocástica de la distribución gaussiana de acciones.
  * `train/approx_kl`: Divergencia de Kullback-Leibler aproximada entre actualizaciones sucesivas de la política.
  * `train/explained_variance`: Proporción de la varianza del retorno explicada por el crítico ($1 - \text{Var}(y - V)/\text{Var}(y)$).
  * `train/ent_coef` (exclusivo SAC): Coeficiente de temperatura dinámico $\alpha$ que pondera el término de entropía.

Para contrastar el progreso entre múltiples semillas o algoritmos, emplee la caja de filtrado por expresiones regulares en el panel izquierdo de la interfaz (por ejemplo, filtrando por `PPO_` o `SAC_`).

---

## 3. Evaluación Mecatrónica y Benchmarking (`evaluate.py`)

El script [evaluate.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/evaluate.py) ejecuta la evaluación determinista de políticas de aprendizaje por refuerzo o del controlador cinemático analítico C0 sobre conjuntos congelados y reproducibles de trayectorias.

### 3.1 Conjuntos de Datos Disponibles (Datasets Congelados)

En [src/datasets.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/src/datasets.py) se generan dos conjuntos independientes con idéntica distribución cinemática en el espacio de trabajo:

* **Conjunto de Validación Intermedia ([datasets/validation_set_200.json](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/datasets/validation_set_200.json)):** Contiene 200 episodios generados con semilla pseudoaleatoria 202601. Se emplea para comparar y descartar variantes de sintonización de hiperparámetros (pesos LQR, arquitecturas de red o funciones de activación) durante la fase de desarrollo.
* **Conjunto de Evaluación Final de Referencia ([datasets/final_eval_set_300.json](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/datasets/final_eval_set_300.json)):** Contiene 300 episodios generados con semilla pseudoaleatoria 202602. Actúa como conjunto de prueba desacoplado (*test set*) para certificar formalmente las métricas definitivas del benchmark mecatrónico sin sesgo de optimización retrospectivo.

### 3.2 Sintaxis del Comando y Parámetros

```powershell
python evaluate.py [--model-path <ruta_modelo.zip>] [--label <identificador>] [--c0] [--vecnormalize <ruta_stats.pkl>] [--dataset <ruta_dataset.json>] [--output-dir <directorio_salida>] [--mode <deterministic|stochastic>] [--plot]
```

Comportamiento de los argumentos y valores por defecto:

* `--dataset` (por defecto: `"datasets/validation_set_200.json"`): Especificación del conjunto de prueba. Admite:
  * `"all"`: Ejecuta consecutivamente la evaluación sobre el conjunto de 200 episodios y sobre el de 300 episodios en una sola instrucción.
  * `"200"` o ruta `datasets/validation_set_200.json`: Evaluación de validación intermedia.
  * `"300"` o ruta `datasets/final_eval_set_300.json`: Evaluación del benchmark final de 300 episodios.
* `--model-path` (por defecto: `None`): Ruta al archivo `.zip` del modelo entrenado. Requerido para agentes RL a menos que se invoque el flag `--c0`.
* `--label` (por defecto: `None`): Etiqueta asignada al modelo para estructurar las carpetas de salida. Si se omite, se infiere automáticamente a partir del nombre del archivo o del directorio del modelo (o se asigna `"C0"` si se evalúa el controlador analítico).
* `--c0` (por defecto: `False`): Bandera para evaluar el controlador cinemático analítico con transpuesta de Jacobiano amortiguada. Admite modificar sus ganancias mediante `--kp` (por defecto: `50.0`), `--kd` (por defecto: `0.05`) y `--lambda-damping` (por defecto: `0.01`).
* `--vecnormalize` (por defecto: `None`): Ruta al archivo `vecnormalize_stats.pkl`. Si se omite, el evaluador busca automáticamente el archivo en el mismo directorio del modelo o en sus directorios padre.
* `--output-dir` (por defecto: `"reports/data/evaluation"`): Directorio raíz donde se almacenan los resultados tabulares.
* `--mode` (por defecto: `"deterministic"`): Modo de inferencia de la política en Gym. En modo determinista toma la media de la distribución gaussiana sin ruido exploratorio.
* `--plot` (por defecto: `False`): Si se activa, invoca de manera automática al generador gráfico tras concluir la evaluación numérica, actualizando las figuras correspondientes.

### 3.3 Ejemplos de Evaluación

#### Evaluación Completa sobre Ambos Datasets (Opción `all`: 200 y 300 Episodios)

```powershell
# Ejecuta la evaluación secuencial sobre 200 y 300 episodios generando figuras en ambas particiones
python evaluate.py --model-path experiments/entorno/PPO_7/seed_42/models/best_model/best_model.zip --dataset all --plot
```

#### Evaluaciones sobre el Dataset Definitivo de 300 Episodios (Benchmark Final)

```powershell
# 1. Evaluación del modelo definitivo PPO_7 sobre el conjunto de 300 episodios
python evaluate.py --model-path experiments/entorno/PPO_7/seed_42/models/best_model/best_model.zip --label PPO_7 --dataset 300

# 2. Evaluación de SAC_2 sobre 300 episodios con generación inmediata de figuras
python evaluate.py --model-path experiments/entorno/SAC_2/seed_42/models/best_model/best_model.zip --label SAC_2 --dataset 300 --plot

# 3. Evaluación del controlador cinemático analítico C0 sobre 300 episodios
python evaluate.py --c0 --label C0 --dataset 300
```

#### Evaluaciones sobre el Dataset de Validación de 200 Episodios (Sintonización y Ablación)

```powershell
# 1. Evaluación de modelo intermedio PPO_2 (emplea por defecto el dataset de 200 episodios)
python evaluate.py --model-path experiments/entorno/PPO_2/seed_42/models/best_model/best_model.zip --label PPO_2

# 2. Evaluación explícita indicando --dataset 200
python evaluate.py --model-path experiments/NN/PPO_S/seed_42/models/best_model/best_model.zip --label PPO_S --dataset 200
python evaluate.py --model-path experiments/NN/PPO_L/seed_42/models/best_model/best_model.zip --label PPO_L --dataset 200
```

### 3.4 Estructura y Segregación de Datos de Salida

Para evitar sobreescrituras entre fases de validación intermedia y evaluación final, los resultados tabulares se almacenan en subdirectorios particionados según el tamaño del conjunto evaluado:

* `reports/data/evaluation/eval_200/<label>/`: Evaluaciones sobre el conjunto de 200 episodios.
* `reports/data/evaluation/eval_300/<label>/`: Evaluaciones sobre el conjunto de 300 episodios.

En cada subdirectorio se generan tres archivos estructurados:

* `episodes.csv`: Telemetría por episodio con las métricas de estabilización (error cartesiano terminal e_ss, velocidad residual v_ss, tiempo de establecimiento t_s, esfuerzo de torque J_u), coordenadas del objetivo cartesiano (target_x, target_y) y el desglose de costos cuadráticos (j_pos, j_vel, j_ctrl, j_lqr).
* `summary.json`: Métricas estadísticas consolidadas (media, desviación estándar, percentiles 50 y 90) e intervalo de confianza binomial de Wilson al 95%.
* `sample_trajectories.csv`: Registro paso a paso de las trayectorias temporales (posiciones articulares q, velocidades dq, errores cartesianos y torques aplicados tau) en episodios de muestra.

---

## 4. Generación Modular de Gráficas (`plot.py`)

El script [plot.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/plot.py) procesa de forma desacoplada los archivos tabulares generados en `reports/data/evaluation/` y los registros de convergencia. Las gráficas se generan de manera independiente en imágenes individuales de alta resolución clasificadas en subdirectorios dedicados por modelo o por combinaciones comparativas específicas.

### 4.1 Sintaxis del Comando y Parámetros

```powershell
python plot.py [--model <nombre_modelo>] [--compare <modelo1,modelo2,...>] [--dataset <200|300|all>] [--metric <metrica1,metrica2,...>] [--data-dir <directorio_datos>] [--out-dir <directorio_salida>] [--eval-log <ruta_evaluations.npz>] [--strict]
```

Comportamiento de los argumentos y valores por defecto:

* `--data-dir` (por defecto: `"reports/data/evaluation"`): Directorio base que contiene los datos tabulares evaluados. Explora recursivamente subcarpetas particionadas (`eval_200`, `eval_300`) y carpetas directas.
* `--out-dir` (por defecto: `"reports"`): Directorio base para la salida gráfica. Se rutea automáticamente a `reports/figures_valid_200/` y `reports/figures_final_300/` según la partición evaluada.
* `--dataset` (por defecto: `None`): Filtro de partición muestral. Admite `200`, `300`, `eval_200` o `eval_300`. Si se omite, procesa automáticamente todas las evaluaciones presentes en las particiones disponibles.
* `--metric`, `--metrics` (por defecto: `None`): Selecciona de forma modular y paramétrica una métrica específica o lista separada por comas para generar exclusivamente las comparativas deseadas, omitiendo el renderizado innecesario del resto de la suite. Admite:
  * **Escalares de Optimizador (TensorBoard):** `explained_variance` (alias: `var_exp`, `ev`), `approx_kl` (alias: `kl`), `value_loss` (alias: `critic_loss`, `vl`), `policy_loss` (alias: `pl`), `entropy`, `loss`, `optimizer_summary`.
  * **Curvas de Aprendizaje:** `learning_return` (alias: `return`), `learning_success_rate` (alias: `train_success`, `learning_sr`), `learning_curves`.
  * **Métricas de Benchmark (Evaluación):** `success_rate`, `steady_state_error` (alias: `e_ss`), `residual_velocity` (alias: `v_ss`), `settling_time` (alias: `t_s`), `control_effort` (alias: `j_u`), `reward_components`, `pareto_tradeoff`, `spatial_error`, `temporal_trajectories`.
  * **Macros consolidadas:** `optimizer` (todas las del optimizador), `eval` (las 9 de benchmark), `all` (suite íntegra de evaluación y optimización).
* `--model` (por defecto: `None`): Genera las figuras individuales para el modelo indicado (por ejemplo, `--model PPO_7`). Si se combina con `--metric`, genera únicamente las métricas solicitadas para ese modelo.
* `--compare` (por defecto: `None`): Lista de modelos separados por comas (sin espacios) para generar un panel gráfico comparativo consolidado (por ejemplo, `--compare PPO_7,PPO_S,PPO_L`).
* `--eval-log` (por defecto: `None`): Ruta opcional a un archivo `evaluations.npz` registrado durante el entrenamiento para graficar curvas de convergencia de optimización.
* `--strict` (por defecto: `False`): Exige consistencia homogénea estricta en el tamaño muestral al comparar modelos. Si se activa y los modelos provienen de particiones distintas (200 vs 300), interrumpe la ejecución con una excepción. Si se omite (`False`), emite una advertencia metodológica en consola, anota el tamaño muestral (N) en cada gráfica y deposita la salida en `reports/figures_mixed/`.

### 4.2 Ejemplos de Ejecución Gráfica

#### Comparativas Personalizadas y Paramétricas por Métrica (--metric)

```powershell
# 1. Comparar varianza explicada (explained_variance) entre PPO_7, PPO_S y PPO_L en dataset 300
python plot.py --dataset 300 --compare PPO_7,PPO_S,PPO_L --metric explained_variance

# 2. Comparar divergencia KL aproximada (approx_kl) entre modelos en dataset 300
python plot.py --dataset 300 --compare PPO_7,PPO_S,PPO_L --metric approx_kl

# 3. Comparativa combinada de múltiples métricas de optimización
python plot.py --dataset 300 --compare PPO_7,PPO_S,PPO_L --metric value_loss,policy_loss

# 4. Panel consolidado 2x3 con todas las pérdidas del optimizador comparadas
python plot.py --dataset 300 --compare PPO_7,PPO_S,PPO_L --metric optimizer_summary

# 5. Comparativa de curvas de aprendizaje durante entrenamiento (retorno y tasa de estabilización)
python plot.py --dataset 300 --compare PPO_7,PPO_S,PPO_L --metric learning_curves

# 6. Comparativa paramétrica de precisión y esfuerzo de control (e_ss y J_u)
python plot.py --dataset 300 --compare PPO_7,PPO_S,PPO_L --metric steady_state_error,control_effort

# 7. Generación de una única métrica para un modelo individual (renderizado instantáneo)
python plot.py --dataset 300 --model PPO_7 --metric explained_variance
```

#### Comparativas sobre el Conjunto Definitivo de 300 Episodios (Suite Completa)

```powershell
# 1. Comparativa de modelos definitivos (Benchmark C0 vs PPO_7 vs SAC_2 en reports/figures_final_300/C0_PPO_7_SAC_2/)
python plot.py --dataset 300 --compare C0,PPO_7,SAC_2

# 2. Comparativa de impacto de capacidad de red neuronal sobre 300 episodios (salida en reports/figures_final_300/PPO_7_PPO_S_PPO_L/)
python plot.py --dataset 300 --compare PPO_7,PPO_S,PPO_L

# 3. Generación de figuras individuales para todos los modelos evaluados en 300 episodios
python plot.py --dataset 300
```

#### Comparativas sobre el Conjunto de Validación de 200 Episodios

```powershell
# 1. Comparativa de funciones de activación sobre 200 episodios (salida en reports/figures_valid_200/PPO_S_PPO_M_TANH_PPO_M_LEAKYRELU/)
python plot.py --dataset 200 --compare PPO_S,PPO_M_TANH,PPO_M_LEAKYRELU

# 2. Generación exclusiva para un único modelo evaluado en 200 episodios
python plot.py --dataset 200 --model PPO_2
```

#### Comparativas Exploratorias Mixtas y Validación Estricta

```powershell
# 1. Comparativa exploratoria con advertencia metodológica no bloqueante (salida en reports/figures_mixed/PPO_7_PPO_M_TANH_PPO_M_LEAKYRELU/)
python plot.py --compare PPO_7,PPO_M_TANH,PPO_M_LEAKYRELU

# 2. Forzar consistencia metodológica estricta para el informe final (arroja excepción si los tamaños de muestra difieren)
python plot.py --compare PPO_7,PPO_M_TANH,PPO_M_LEAKYRELU --strict
```

### 4.3 Organización de las Figuras Generadas

Las figuras se almacenan de forma desacoplada y directa en `reports/figures_valid_200/`, `reports/figures_final_300/` y `reports/figures_mixed/`:

#### Subcarpetas por Modelo (`reports/figures_<N>/<modelo>/`)

* **Métricas Mecatrónicas de Estabilización y Control:**
  * `success_rate.png`: Tasa de estabilización terminal con intervalo de confianza de Wilson al 95%.
  * `steady_state_error.png`: Distribución del error cartesiano terminal e_ss con tolerancia de radio r_target = 20 mm.
  * `residual_velocity.png`: Distribución de la velocidad angular residual v_ss con cota v_tol = 0.10 rad/s.
  * `settling_time.png`: Distribución del tiempo de establecimiento transitorio t_s al 5%.
  * `control_effort.png`: Gasto energético acumulado por esfuerzo de torque J_u.
  * `reward_components.png`: Evolución de las tres curvas de costo cuadrático LQR (posición J_pos, velocidad J_vel y control J_ctrl).
  * `temporal_trajectory_episode.png`: Respuesta temporal de 3 paneles (error cartesiano ||e(t)||, variables articulares q(t) y torques de control tau(t) frente a límites de saturación +-0.05 N*m).
  * `pareto_tradeoff.png`: Diagrama de dispersión bi-objetivo (e_ss vs J_u) con la frontera empírica de Pareto.
  * `workspace_spatial_error.png`: Distribución espacial del error cartesiano en el plano de trabajo (x, y) con límites cinemáticos del manipulador.

* **Curvas de Aprendizaje y Optimización:**
  * `learning_curve_return.png`: Retorno episódico medio a lo largo del entrenamiento (con banda de dispersión si existen varias semillas).
  * `learning_curve_success_rate.png`: Tasa de éxito durante el entrenamiento.
  * `multiseed_learning_curves.png`: Panel comparativo multisemilla con bandas de desviación estadística.
  * `optimizer_total_loss.png`: Curva de la pérdida total del optimizador.
  * `optimizer_policy_loss.png`: Pérdida del gradiente de la política (clipped surrogate loss).
  * `optimizer_value_loss.png`: Pérdida cuadrática de la función de valor (crítico).
  * `optimizer_entropy.png`: Evolución temporal de la entropía de la política.
  * `optimizer_approx_kl.png`: Divergencia de Kullback-Leibler aproximada por época.
  * `optimizer_explained_variance.png`: Varianza explicada de la función de valor.
  * `optimizer_convergence_summary.png`: Panel resumen 2x3 con todas las pérdidas del optimizador.

#### Subcarpetas Comparativas (`reports/figures_<dataset>/<modelo1>_<modelo2>_.../`)

* **Métricas Mecatrónicas de Estabilización y Control:**
  * `comparative_success_rate.png`: Comparación de la tasa de éxito entre los modelos contrastados con intervalos de Wilson al 95%.
  * `comparative_steady_state_error.png`: Comparativa de precisión cartesiana terminal e_ss frente a tolerancia r_target = 20 mm.
  * `comparative_residual_velocity.png`: Comparativa de disipación de velocidad angular residual v_ss frente a cota v_tol = 0.10 rad/s.
  * `comparative_settling_time.png`: Comparativa de rapidez de respuesta temporal t_s al 5%.
  * `comparative_control_effort.png`: Comparativa del gasto de control acumulado J_u.
  * `comparative_reward_components.png`: Comparativa del desglose de costos cuadráticos LQR en barras agrupadas.
  * `comparative_pareto_tradeoff.png`: Dispersión de episodios y centroides en el espacio bi-objetivo e_ss vs J_u con frontera de Pareto.
  * `comparative_spatial_error.png`: Subplots espaciales comparativos del error cartesiano en el espacio de trabajo.
  * `comparative_temporal_trajectories.png`: Superposición de respuestas temporales de los modelos sobre un mismo episodio.
* **Telemetría de Entrenamiento y Convergencia de Optimización:**
  * `comparative_explained_variance.png`: Varianza explicada de la función de valor del crítico frente a referencia óptima 1.0.
  * `comparative_approx_kl.png`: Divergencia KL aproximada entre políticas consecutivas con umbral de guía 0.015.
  * `comparative_value_loss.png`: Pérdida cuadrática de la función de valor (MSE del crítico).
  * `comparative_policy_loss.png`: Pérdida de la política por gradiente truncado (clipped surrogate).
  * `comparative_entropy.png`: Evolución temporal de la entropía de la distribución de acciones.
  * `comparative_loss.png`: Pérdida escalar total del optimizador.
  * `comparative_optimizer_summary.png`: Panel 2x3 consolidando las 6 curvas de telemetría del optimizador comparadas.
  * `comparative_learning_return.png`: Retorno episódico acumulado durante el entrenamiento (con bandas de dispersión +-1 sigma si existen múltiples semillas).
  * `comparative_learning_success_rate.png`: Tasa de estabilización durante el entrenamiento evaluada periódicamente.

---

## 5. Simulación Visual en Tiempo Real (`interactive_render.py`)

El visualizador [interactive_render.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/interactive_render.py) interactúa directamente con la ventana nativa OpenGL de MuJoCo en 3D para inspección cualitativa del movimiento del manipulador:

### 5.1 Sintaxis del Comando y Parámetros

```powershell
python interactive_render.py [<ruta_modelo.zip>] [--c0] [--config <ruta_yaml>] [--episodes <int>]
```

Comportamiento de los argumentos y valores por defecto:

* `model_path` (posicional, por defecto: `None`): Ruta al archivo `.zip` del modelo entrenado. Requerido si no se utiliza `--c0`. Localiza automáticamente las estadísticas de normalización adyacentes.
* `--c0` (por defecto: `False`): Activa la simulación del controlador analítico cinemático en lugar de una política de RL.
* `--config` / `-c` (por defecto: `None`): Ruta al archivo YAML con los parámetros del entorno. Si se omite, carga [configs/environment.yaml](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/configs/environment.yaml).
* `--episodes` / `-e` (por defecto: `5`): Número de episodios a simular en la sesión visual.

### 5.2 Ejemplos de Visualización

```powershell
# Visualización de un modelo entrenado de RL (5 episodios por defecto)
python interactive_render.py experiments/entorno/PPO_7/seed_42/models/best_model/best_model.zip

# Visualización del controlador cartesiano analítico C0 durante 10 episodios
python interactive_render.py --c0 --episodes 10
```

En la ventana de MuJoCo es posible manipular la cámara orbital mediante el ratón (rotación con clic izquierdo, traslación con clic derecho y zoom con la rueda). La consola imprime en cada paso la distancia instantánea, velocidades articulares y torques ejercidos.

---

## 6. Ejecución de la Suite de Pruebas Unitarias

La suite de validación asegura la coherencia dimensional, cinemática y algorítmica del sistema mediante `pytest`:

```powershell
python -m pytest tests/
```

Módulos de prueba verificados:

* [tests/test_config.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/tests/test_config.py): Carga desacoplada de configuraciones y verificación de parámetros SSOT de MuJoCo Reacher-v5.
* [tests/test_kinematics.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/tests/test_kinematics.py): Validación de cinemática directa y matriz Jacobiana contrastada contra las librerías cinemáticas de MuJoCo.
* [tests/test_normalization.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/tests/test_normalization.py): Normalización vectorial `VecNormalize`, cálculo de recompensa cuadrática continua LQR y truncamiento del horizonte de simulación.

---

## 7. Generación de la Matriz de Benchmarking (`generate_benchmark_matrix.py`)

Para consolidar en una única tabla todos los hiperparámetros del entorno físico, los algoritmos de control y las métricas cinemáticas y dinámicas calculadas durante las evaluaciones, se provee el script [generate_benchmark_matrix.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/generate_benchmark_matrix.py):

```powershell
python generate_benchmark_matrix.py
```

El script explora recursivamente todos los archivos `.yaml` ubicados en `configs/`, extrae los parámetros de preprocesamiento, redes y optimización, y vincula automáticamente las métricas de evaluación almacenadas en `reports/data/evaluation/eval_300/` (prioritario) o `reports/data/evaluation/eval_200/`. La matriz resultante se exporta como archivo delimitado por comas en [reports/data/benchmark_matrix.csv](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/reports/data/benchmark_matrix.csv).
