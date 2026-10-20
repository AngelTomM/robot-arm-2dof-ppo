# 📘 Manual de Uso — Brazo Robótico 2 DOF

Manual operativo del proyecto: instalación, entrenamiento unificado, simulación interactiva, evaluación formal y ejecución de la suite de validación.
Para el fundamento técnico (cinemática, criterio de éxito y diseño de recompensas), consulte [`docs/metodologia.md`](metodologia.md).

---

## 1. Instalación y Configuración

### 1.1 Clonar el repositorio
```bash
git clone https://github.com/AngelTomM/robot-arm-2dof-ppo.git
cd robot-arm-2dof-ppo
```

### 1.2 Entorno virtual (Python 3.11)
- **Windows (PowerShell):**
  ```powershell
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  python3.11 -m venv .venv
  source .venv/bin/activate
  ```


### 1.3 Dependencias
```bash
pip install -r requirements.txt
```

---

## 2. Entrenamiento de Agentes (`train.py`)

Todo el entrenamiento se ejecuta desde el **punto de entrada único** en la raíz. Dos principios de diseño:

1. **Configuración como fuente única de verdad (SSOT):** la fase y el identificador del experimento se derivan de la **ruta** del archivo YAML de configuración, no de su contenido.
2. **Soporte dual de algoritmos:** si el YAML contiene la sección `sac:` se instancia SAC; si contiene `ppo:`, PPO. La presencia de ambas secciones, o de ninguna, produce un error de configuración.

### 2.1 Sintaxis general
```powershell
python train.py --config <ruta_del_yaml> [--seed <n>] [--device <s>] [--total_timesteps <n>] [--checkpoint_freq <n>]
```

### 2.2 Argumentos

| Argumento | Valor por defecto | Descripción |
| :--- | :---: | :--- |
| `--config` | *(requerido)* | Ruta al YAML del experimento bajo `configs/`. Define fase, experimento y algoritmo. |
| `--seed` | `42` | Semilla aleatoria (global: Python, NumPy, torch, CUDA). |
| `--device` | `auto` | Dispositivo de cómputo: `auto` (CUDA si está disponible; si no, CPU), `cpu` o `cuda`. |
| `--total_timesteps` | del YAML | Sustituye el presupuesto de pasos de `training.total_timesteps`. |
| `--checkpoint_freq` | del YAML | Sustituye la frecuencia de guardado de `training.checkpoint_freq`. |

### 2.3 Derivación de la fase (SSOT)

| Ruta del config | Fase derivada | Directorio de salida |
| :--- | :---: | :--- |
| `configs/phase_a/X3.yaml` | `phase_a` | `experiments/phase_a/X3/seed_<n>/` |
| `configs/phase_b/X3.yaml` | `phase_b` | `experiments/phase_b/X3/seed_<n>/` |
| `configs/phase_b/extension/X4.yaml` | `phase_b_ext` | `experiments/phase_b_ext/X4/seed_<n>/` |
| `configs/phase_c/S1.yaml` | `phase_c` | `experiments/phase_c/S1/seed_<n>/` |

De forma genérica: la fase es la ruta del config relativa a `configs/` aplanada con `_`, y la salida se escribe en `experiments/<fase>/<exp_id>/seed_<n>/`.

### 2.4 Ejemplos por fase

```powershell
# Fase A — diagnóstico (PPO, 300.000 pasos)
python train.py --config configs/phase_a/X3.yaml --seed 42

# Fase B — presupuesto de 1.000.000 de pasos (PPO)
python train.py --config configs/phase_b/X3.yaml --seed 42

# Extensión de Fase B — fase_b_ext (PPO, 655.360 pasos)
python train.py --config configs/phase_b/extension/X4.yaml --seed 42

# Fase C — agentes SAC (655.360 pasos)
python train.py --config configs/phase_c/S1.yaml --seed 42
```

### 2.5 Artefactos de salida

Cada ejecución escribe exclusivamente en `experiments/<fase>/<exp_id>/seed_<n>/`:

| Ruta | Contenido |
| :--- | :--- |
| `models/<algoritmo>_arm2d_final.zip` | Pesos del modelo final. |
| `models/best_model/` | Mejor modelo según la evaluación periódica (`best_model.zip` + `vecnormalize_stats.pkl`). |
| `models/vecnormalize_stats.pkl` | Estadísticas de `VecNormalize` del entorno de entrenamiento, guardadas de forma acoplada al modelo final. |
| `checkpoints/ckpt_<paso>/` | Checkpoint periódico: `model.zip` **y** `vecnormalize_stats.pkl` correspondientes al mismo paso (deben cargarse siempre juntos). |
| `logs/` | Registros de TensorBoard (`tensorboard --logdir experiments/<fase>/<exp_id>/seed_<n>/logs`). |
| `experiment_meta.json` | Metadatos: config fuente, fase, algoritmo, semilla, dispositivo, y parámetros completos. |
| `train_metrics.json` | Tiempo de entrenamiento, rendimiento medio (FPS) y pasos de los checkpoints guardados. |

### 2.6 Callbacks y telemetría

- `PhaseATrainingCallback`: registra en TensorBoard la tasa de éxito y la longitud de permanencia (*hold*) en ventanas de 100 episodios.
- `PhaseAEvalCallback`: evaluación periódica en línea (activa únicamente si el YAML define `training.eval_freq > 0`), con sincronización de las estadísticas de `VecNormalize` al entorno de evaluación en modo inferencia.
- `PhaseBCheckpointCallback`: guardado periódico del par modelo + estadísticas de normalización en el mismo paso.

Una ejecución de referencia rápida para verificar la instalación:
```powershell
python train.py --config configs/phase_c/S1_smooth_200.yaml --seed 42 --total_timesteps 4000
```

---

## 3. Simulación Interactiva en Tiempo Real

Vista del agente entrenado a 60 FPS. Carga automáticamente el vector de normalización acoplado `vecnormalize_stats.pkl` asociado al checkpoint (los artefactos `model.zip` y `vecnormalize_stats.pkl` son locales y no se versionan; reproduzca primero el entrenamiento con `train.py` si es necesario):

```powershell
# Agente X4 (checkpoint de la carpeta ckpt_<paso>/: model.zip + vecnormalize_stats.pkl)
python interactive_render.py experiments/phase_b/extension/X4/seed_42/checkpoints/ckpt_450560/model.zip --config configs/phase_b/extension/X4.yaml

# Ejemplo con un checkpoint recién entrenado por train.py (nuevo layout phase_b_ext)
python interactive_render.py experiments/phase_b_ext/X4/seed_42/checkpoints/ckpt_<paso>/model.zip --config configs/phase_b/extension/X4.yaml
```

| Control | Función |
| :--- | :--- |
| 🖱️ Clic izquierdo (mantener y arrastrar) | Reubica el objetivo hacia la posición del cursor. |
| ⌨️ Barra espaciadora | Genera un objetivo aleatorio dentro del espacio de trabajo alcanzable. |
| ⌨️ Tecla `R` | Reinicia la configuración articular del robot. |
| ⌨️ Tecla `ESC` | Cierra la ventana del simulador. |

---

## 4. Evaluación y Benchmarking Formal

Replicación de la evaluación multi-semilla confirmada ($n=500$ episodios por semilla, dataset congelado criptográficamente, intervalos de Wilson al 95%):

```powershell
python scripts/phase_b_ext_run_b12_final_eval.py
```

El resultado se registra de forma reproducible en `reports/data/phase_b/extension/final_b2_selection_verdict.json`.

Otros recursos de análisis en `scripts/`:
- Evaluación de checkpoints: punto único `evaluate_checkpoints.py` sobre el módulo `src/evaluation/checkpoints.py` (docstring como fuente de parámetros):

```powershell
python scripts/evaluate_checkpoints.py --help

# Fase B: todos los checkpoints de una corrida sobre 'validation' (200 episodios)
python scripts/evaluate_checkpoints.py --phase b --exp_dir experiments/<fase>/<exp>/seed_<seed> --config configs/<fase>/<exp>.yaml [--output_csv <ruta>]

# Fase C: S1 (SAC) + X3 (PPO) + referencia C0 sobre 'validation_c' (500 episodios)
python scripts/evaluate_checkpoints.py --phase c [--dataset <json>] [--output-dir <dir>] [--mode deterministic|stochastic]
```

La extensión de Fase B se orquesta con `phase_b_ext_batch_eval_validation.py` (barrido de todos los checkpoints sobre `validation_b2`) y `phase_b_ext_run_b12_final_eval.py` (evaluación terminal de la selección de la Regla 4.1).

- `phase_b_ext_apply_selection.py`: regla formal de selección (suavizado y desempate) con cálculo del intervalo de indistinguibilidad $U_{equiv}$.
- Generación de figuras: punto único `generate_figures.py` sobre el paquete `src/visualization/` (sección 5).
- Prueba de humo con entrenamiento real corto (no toca datos históricos): los configs de `configs/smoke/` derivan la fase desde su ruta hacia `experiments/smoke/`:

```powershell
python train.py --config configs/smoke/x3_smoke.yaml --total_timesteps 8192 --checkpoint_freq 4096   # PPO
python train.py --config configs/smoke/s1_smoke.yaml --total_timesteps 8192 --checkpoint_freq 4096   # SAC
python scripts/evaluate_checkpoints.py --phase b --exp_dir experiments/smoke/X3/seed_42 --config configs/smoke/x3_smoke.yaml --output_csv <ruta>
```

---

## 5. Generación de Figuras

Todos los generadores se invocan a través de un único punto de entrada:

```powershell
python scripts/generate_figures.py --help              # referencia completa
python scripts/generate_figures.py --phase <fase> [--out-dir <ruta>]
```

Las fases `b` y `c` admiten además **banderas de selección** para dibujar modelos
nuevos sin tocar código: `--experiments` (lista de modelos, p. ej. `X3,S7`),
`--seeds`, `--exps-dir` (raíz alternativa de corridas para el TensorBoard),
`--data-dir` (CSVs de evaluación alternativos, mismo esquema histórico de
columnas) y `--figures` (subconjunto, p. ej. `1,3`). Sin banderas, la salida es
idéntica al estudio histórico. Las fases monográficas (`a`, `a_agg`,
`trajectories`, `b_ext`, `c_s3`) no las aceptan.

Receta para un modelo recién entrenado (ejemplo con la config Fase C
`S1_smooth_200` y presupuesto corto):

```powershell
# 1) Entrenar (la fase y el experimento se derivan de la ruta del config)
python train.py --config configs/phase_c/S1_smooth_200.yaml --total_timesteps 8192 --checkpoint_freq 4096

# 2) Evaluar los checkpoints -> CSV (fase c: bastón biblioteca sobre
#    src.evaluation.checkpoints.run_phase_c_evaluation; fase b por corrida:
#    python scripts/evaluate_checkpoints.py --phase b --exp_dir ... --config ...)

# 3) Generar las figuras solo del modelo nuevo
python scripts/generate_figures.py --phase c --data-dir <csvs> --experiments S1_smooth_200 --exps-dir experiments/phase_c/S1_smooth_200 --out-dir <figs>
python scripts/generate_figures.py --phase b --data-dir <csvs> --experiments S1_smooth_200 --seeds 42 --figures 2,3 --out-dir <figs>
```

Los experimentos sin color canónico en la paleta (`EXP_COLORS`/`AGENT_COLORS`)
reciben un color del ciclo neutro por orden de aparición
(`src/visualization/common.color_resolver`).

| Fase | Contenido | Insumos | Salida por defecto |
| :--- | :--- | :--- | :--- |
| `a` | Figuras históricas de la Fase A por experimento (fig1–fig4) | `reports/data/all_final_*.csv` | `reports/figures/` |
| `a_agg` | Auditoría agregada multi-semilla de Fase A + validaciones V01–V18 | CSV consolidados + TensorBoard de `experiments/phase_a` | `reports/data/aggregated/` y `reports/figures/aggregated/` |
| `trajectories` | Trayectorias cartesianas reales (4 casos canónicos) | Modelos PPO en `experiments/phase_a/*/seed_42/models/` | `reports/figures/` |
| `b` | 8 figuras de la Fase B con JSON de trazabilidad | TensorBoard de `experiments/phase_b` + `reports/data/phase_b/` | `reports/figures/phase_b/` |
| `b_ext` | Figuras de la Extensión (curvas, Pareto, selección) | `reports/data/phase_b/extension/*.csv\|json` | `reports/figures/phase_b/extension/` |
| `c` | 6 figuras de la Fase C (SAC) | `reports/data/phase_c/*.csv` + `experiments/phase_c/S1` (opcional) | `reports/figures/phase_c/` |
| `c_s3` | 4 figuras del análisis de cierre S3_combined | `reports/phase_c_s2_analysis/` y `phase_c_s3_analysis/` | `reports/phase_c_s3_analysis/figures/` |

### 5.1 Parámetros configurables de cada generador

La fuente de verdad de los parámetros son los docstrings de cada módulo
(`src/visualization/<modulo>.py`, consultables con
`python -c "help('src.visualization.phase_b')"` o leyendo el archivo).
Resumen orientativo:

- **`phase_a`**: `output_dir`; `exp_order` y `colors` (definidos dentro de `generate_all_plots`).
- **`phase_a_agg`**: `DATA_DIR` (lecturas); `run(out_dir)` redirige las escrituras (`data_aggregated/`, `figures/aggregated/`, `ppo_diagnostics/`); `EXP_COLORS`, `REPLICATED_RL_EXPS`, `SEEDS`.
- **`trajectories`**: `output_dir`; rutas de modelos (`x5_model_path` ... `x6_norm_path`); episodios de los casos canónicos (`final_dataset[0/5/10/12]`); horizonte de 200 pasos.
- **`phase_b`**: `FIG_B_DIR`/`DATA_B_DIR`; `EXP_COLORS`, `CAT_COLORS`/`CATEGORIES` (mantener tonos históricos: los JSON publicados los asocian a las figuras ya publicadas); figs 2/3/8 parametrizables (`experiments`, `seeds`, `exps_dir`, `selected_ckpts` solo por biblioteca); figs 1 y 4-7 monográficas (se omiten con aviso si faltan sus CSV).
- **`phase_b_ext`**: `out_dir`; `AGENT_COLORS`; listas `main_agents` y `pilot_agents`.
- **`phase_c`**: `data_dir`/`figures_dir`; `AGENT_COLORS`; `experiments`/`sac_exp_dir`/`figures` (figs 1-3 y 5 parametrizables); `step_filter` (655360 por defecto; 245760 para el presupuesto del estudio S3; `None` sin filtro); `time_s1_sec`/`time_x3_sec` (4600.0 y 720.0 segundos medidos en GPU).
- **`phase_c_s3`**: `out_dir`; `STEPS`, `SEEDS`, `COLORS`; `train_times`/`fps_vals` (tiempos de entrenamiento en minutos y FPS medios por modelo, medidos en GPU, de la fig4).

### 5.2 Notas de reproducción

- **Aviso controlado**: si falta el insumo principal, los generadores `a` y `trajectories` avisan y terminan sin error ("no encontrado aún", "Modelos aún no listos"); el resto aborta con la excepción correspondiente. En la fase `b`, las figuras monográficas 1 y 4-7 se omiten con aviso ("[OMITIDA]") cuando sus CSV no existen (p. ej. con un `--data-dir` alternativo).
- **Opcionales**: en la fase `c`, `fig2` (diagnósticos SAC) se genera solo si existe la corrida indicada (`--exps-dir`; por defecto `experiments/phase_c/S1/`); las figs 4-5 solo si existe `step_data_val_geometry.csv` en el `--data-dir`.
- **`a_agg`**: las tablas y figuras se generan íntegras; la suite V01–V18 exige los logs históricos de TensorBoard (`experiments/phase_a/<exp>/seed_<seed>/logs/`); si no están presentes, la validación V12 se interrumpe tras la generación de artefactos.
- **Trazabilidad JSON**: cada figura con metadatos escribe un `<figura>.json` junto al PNG. El campo `script_path` apunta ahora al módulo migrado en `src/visualization/` (procedencia real); los esquemas de claves son idénticos a los publicados.
- La invocación directa de cada módulo (`python -m src.visualization.phase_b`) es equivalente al despachador.

---

## 6. Suite de Validación

31 pruebas unitarias: cinemática directa, Jacobiano analítico, contador de *hold*, conservación de suavidad, checkpointing acoplado a `VecNormalize`, integridad de los datasets congelados y API de Gymnasium.

```powershell
python -m pytest tests/ -q
```

---

## 7. Estructura del Repositorio

```text
robot-arm-2dof-ppo/
├── train.py                                 # Punto de entrada único de entrenamiento (PPO/SAC)
├── arm2d_env.py                             # Entorno Gymnasium (física, cinemática directa, hold, jacobiano)
├── interactive_render.py                    # Simulador interactivo Pygame con auto-carga de normalización
├── configs/
│   ├── phase_a/                             # Configuraciones de diagnóstico de la Fase A (X1a a X6)
│   ├── phase_b/                             # Configuraciones a 1M pasos (X3, X5_near)
│   │   └── extension/                       # Configuraciones de la Extensión (X1a, X2, X3, X4, X5, X6) -> fase phase_b_ext
│   ├── phase_c/                             # Configuraciones SAC de la Fase C (S1, S1_smooth_200, S2_freq, S2_wa, S3_combined)
│   └── smoke/                               # Fixtures de humo con entrenamiento corto -> experiments/smoke/ (sección 4)
├── experiments/
│   └── <phase>/<exp_id>/seed_<seed>/        # Salida estricta del entrenamiento (ver sección 2.5)
├── src/
│   ├── controllers/
│   │   └── jacobian_controller.py           # Controlador cinemático diferencial analítico C0 (Jacobiano DLS)
│   ├── evaluation/
│   │   ├── datasets.py                      # Generación y congelamiento criptográfico de datasets (n=200, 300, 500)
│   │   ├── evaluator.py                     # Motor formal de evaluación multi-episodio (determinista/estocástico)
│   │   ├── checkpoints.py                   # Evaluación de checkpoints (núcleo de Fases B/B-ext/C, esquemas 'b' y 'c')
│   │   └── metrics.py                       # Métricas operativas (éxito estricto, Wilson 95%, etc.)
│   ├── training/
│   │   ├── callbacks.py                     # Callbacks de telemetría y evaluación en línea (sincronización VecNormalize)
│   │   └── phase_b_callbacks.py             # Checkpointing acoplado (model.zip + vecnormalize_stats.pkl)
│   └── visualization/                       # Paquete de generación de figuras por fase (sección 5)
│       ├── common.py                        # Estilo, guardado con metadatos JSON y clasificación de desenlaces
│       ├── phase_a.py, phase_a_agg.py, trajectories.py, phase_b.py
│       └── phase_b_ext.py, phase_c.py, phase_c_s3.py
├── scripts/
│   ├── evaluate_checkpoints.py              # Punto único de evaluación de checkpoints (--phase b|c)
│   ├── generate_figures.py                  # Punto único de generación de figuras (--phase a|a_agg|trajectories|b|b_ext|c|c_s3)
│   └── ...                                  # runners batch, selección formal y análisis
├── reports/                                 # Informes por fase, datos agregados y figuras exportadas
├── docs/
│   ├── manual-de-uso.md                     # Este manual
│   └── metodologia.md                       # Fundamento técnico y protocolo experimental
├── tests/                                   # Suite de 31 pruebas unitarias
├── requirements.txt                         # Dependencias fijadas (Python 3.11)
├── .dvc/ + *.dvc                            # Versionado de datos con DVC (sección 8)
├── train.py                                 # raíz — ver arriba
└── README.md                                # Página de talón (resumen, hallazgos y arranque rápido)
```

---

## 8. Trabajo en Equipo con DVC (modelos y figuras compartidas)

Los binarios generados (modelos `.zip`, `vecnormalize_stats.pkl`, TensorBoard,
CSV/JSON de `experiments/` y las figuras PNG+JSON de `reports/figures/` y
`reports/phase_c_s3_analysis/figures/`) **no viven en git**: se versionan con
[DVC](https://dvc.org) y se sincronizan vía Google Drive. Git conserva el
código, `configs/`, `docs/`, los CSV pequeños de `reports/data/` y los
informes `.md`.

### 8.1 Remote y primer arranque (ya configurado en el repo)

El remoto ya está fijado en `.dvc/config` (sin secretos; cada cuenta se
autentica aparte la primera vez que hace push/pull):

| Remoto | URL | Carpeta compartida |
| :--- | :--- | :--- |
| `storage` (por defecto) | `gdrive://1IEO1rKzjEdylb-TTwW7NywV8oNclvjnX` | <https://drive.google.com/drive/folders/1IEO1rKzjEdylb-TTwW7NywV8oNclvjnX> |

Google bloquea el acceso desde apps "anónimas", así que se usa un **Client ID
OAuth propio** del grupo (ya configurado en `.dvc/config`; la pantalla de
consentimiento debe estar en modo *Testing* con las 3 cuentas como *Test
users* — al ver "acceso bloqueado": *Advanced → Go to app*). El
`client_secret` vive SOLO en `.dvc/config.local` (gitignored): si haces pull
en una máquina nueva, el compañero de infra te lo pasa y lo añades a mano:

```powershell
dvc remote modify --local storage gdrive_client_secret <SECRETO>
```

Además, en el proyecto de Google Cloud (`339127141205`) debe estar
**habilitada la Google Drive API**
(<https://console.developers.google.com/apis/api/drive.googleapis.com/overview?project=339127141205>);
tras crear o cambiar el cliente OAuth, Google tarda entre 5 minutos y un par
de horas en propagar el cambio.

Tras `git pull`, instala la dependencia y baja los datos versionados:

```powershell
pip install -r requirements.txt   # incluye dvc[gdrive]==3.67.1
dvc pull                          # 1ª vez: abre el navegador para tu cuenta de Google
```

Para subir artefactos nuevos que hayas generado tú:

```powershell
dvc add <ruta>        # p. ej. experiments/phase_c/S2_freq, reports/figures
dvc push              # sube los binarios a la carpeta de Drive compartida
git add *.dvc && git commit -m "dvc: <descripcion>" && git push
```

Los `.dvc` son los "punteros" versionados: quien hace pull de git recibe el
hash de los datos y `dvc pull` los materializa en el sitio correcto.

### 8.2 Qué se gestiona con cada herramienta

| Contenido | Vía |
| :--- | :--- |
| Código, `configs/`, `docs/`, `scripts/`, `tests/` | git |
| `experiments/` completo (modelos, VecNormalize, TensorBoard, evals) | DVC (`experiments.dvc`) |
| `reports/figures/` (PNG 300 dpi + JSON trazabilidad) | DVC (`reports/figures.dvc`) |
| `reports/phase_c_s3_analysis/figures/` | DVC (`reports/phase_c_s3_analysis/figures.dvc`) |
| `reports/data/` (CSV de evaluación, pequeños) e informes `.md` | git (diferenciables en diffs) |

### 8.3 Flujo diario recomendado

1. `git pull && dvc pull` al empezar el día (trae modelos/figuras de los compañeros).
2. Entrenar/evaluar/generar figuras normalmente (secciones 2-5).
3. `dvc add <lo nuevo>` → `dvc push` → `git add` los `.dvc` → `git commit` → `git push`.
4. Evitar editar los mismos `.dvc` a la vez entre varias personas: quien entrena
   es quien hace `add`; los demás solo hacen `pull`.

### 8.4 Caveats conocidos

- En carpeta de Drive "compartida normal" la cuota de cada archivo subido
  cuenta contra quien lo sube: a escala actual (decenas de MB) es irrelevante.
- DVC sube los tuyos localmente a `.dvc/cache/`: NO sincronizar el repo con
  el cliente de Google Drive Desktop (duplicaría la caché).
- La fase `a_agg` de figuras necesita el TensorBoard histórico de
  `experiments/phase_a/*/seed_*/logs`; si tu máquina no tiene los logs,
  `dvc pull` (de `experiments.dvc`) los trae y la fase vuelve a ser
  reproducible. Sin logs, las tablas pierden las columnas de entrenamiento.
- Si `dvc status` muestra cambios que no reconoces, `git status` de los
  `.dvc` te dirá qué "puntero" cambió y quién lo subió.
