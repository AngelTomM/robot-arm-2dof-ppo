# 📘 Manual de Uso — Brazo Robótico 2 DOF (Fase D: MDP Continuo con Regulación LQR)

Manual operativo del proyecto: instalación, entrenamiento continuo, simulación interactiva, evaluación dual desacoplada y ejecución de la suite de validación.
Para el fundamento técnico (cinemática, MDP, regulación LQR y formulación matemática), consulte [`docs/metodologia.md`](metodologia.md).

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

Todo el entrenamiento se ejecuta desde el punto de entrada unificado en la raíz.
Principios de diseño:
1. **Configuración como fuente de verdad (SSOT):** La fase y el identificador del experimento se derivan de la ruta del archivo YAML bajo `configs/`.
2. **Soporte de algoritmos:** Si el YAML contiene `ppo:`, se instancia PPO; si contiene `sac:`, se instancia SAC.
3. **Formulación MDP pura:** Entrenado con `reward_type: lyapunov_lqr` y `success_mode: none`.

### 2.1 Sintaxis general
```powershell
python train.py --config <ruta_del_yaml> [--seed <n>] [--device <s>] [--total_timesteps <n>] [--checkpoint_freq <n>]
```

### 2.2 Configuraciones vigentes de Fase D
```powershell
# Entrenamiento PPO_1 (PPO continuo con ventaja GAE)
python train.py --config configs/phase_X/prueba/PPO_1.yaml --seed 42

# Entrenamiento SAC_1 (Soft Actor-Critic continuo)
python train.py --config configs/phase_X/prueba/SAC_1.yaml --seed 42

# Pruebas de humo rápidas (1.000 timesteps):
python train.py --config configs/phase_X/smoke/x3_smoke.yaml --total_timesteps 1000
python train.py --config configs/phase_X/smoke/s1_smoke.yaml --total_timesteps 1000
```

### 2.3 Artefactos generados
Cada ejecución genera en `experiments/<fase>/<exp_id>/seed_<seed>/`:
- `models/<algoritmo>_arm2d_final.zip`: Pesos del modelo final.
- `models/best_model/`: Mejor checkpoint (`best_model.zip` + `vecnormalize_stats.pkl`).
- `checkpoints/ckpt_<paso>/`: Guardado periódico acoplado (`model.zip` y `vecnormalize_stats.pkl`).
- `logs/`: Telemetría continua de TensorBoard (`tensorboard --logdir experiments/`).

---

## 3. Simulación Interactiva en Tiempo Real (`interactive_render.py`)

Simulador visual a 60 FPS con Pygame. Detecta automáticamente algoritmos PPO y SAC y carga de forma acoplada las estadísticas de normalización `vecnormalize_stats.pkl`:

```powershell
# Visualizar agente PPO_1
python interactive_render.py experiments/phase_X/prueba/PPO_1/seed_42/models/best_model/best_model.zip --config configs/phase_X/prueba/PPO_1.yaml

# Visualizar agente SAC_1
python interactive_render.py experiments/phase_X/prueba/SAC_1/seed_42/models/best_model/best_model.zip --config configs/phase_X/prueba/SAC_1.yaml
```

| Control | Acción |
| :--- | :--- |
| 🖱️ Clic izquierdo y arrastrar | Mueve interactivamente el objetivo cartesiano. |
| ⌨️ Barra espaciadora | Genera un objetivo aleatorio en el espacio de trabajo. |
| ⌨️ Tecla `R` | Reinicia la postura articular inicial. |
| ⌨️ Tecla `ESC` | Cierra el simulador. |

El HUD en pantalla muestra en tiempo real: distancia al objetivo, velocidad articular instantánea ($\|\dot{\boldsymbol{\theta}}\|_2$), recompensa instantánea y desglose por componentes.

---

## 4. Evaluación y Benchmarking Formal

La evaluación en Fase D desacopla la métrica de éxito de la función de recompensa del agente, evaluando sobre la cinemática cruda:
- **Opción 2 (Métrica Principal):** Estabilización en los últimos $k=20$ pasos ($\bar{d}_{20} \le 0.06\text{ m}$, $\bar{v}_{20} \le 0.10\text{ rad/s}$).
- **Opción 3 (Métrica Canónica):** Costo integral continuo LQR ($J_{\text{LQR}} = -\sum [1.0\cdot d_t^2 + 0.05\cdot v_t^2 + 0.01\cdot \|a_t\|^2]$).

### 4.1 Evaluación de Checkpoints
```powershell
# Evaluar una corrida individual
python scripts/evaluate_checkpoints.py --exp_dir experiments/phase_X/prueba/PPO_1/seed_42 --config configs/phase_X/prueba/PPO_1.yaml

# Evaluar suite completa Fase D (PPO_1, SAC_1, C0) sobre validation_set_200:
python scripts/evaluate_checkpoints.py --phase d
```

### 4.2 Ajuste y Evaluación del Controlador Cinemático Clásico (`C0`)
```powershell
python scripts/tune_and_eval_controller.py
```
Ajusta la ganancia $K_p$ y el amortiguamiento Levenberg-Marquardt $\lambda$ sobre el conjunto de validación ($n=200$) y evalúa el controlador congelado sobre el conjunto final ($n=300$) reportando Opción 2 y Opción 3.

---

## 5. Generación de Figuras

```powershell
# Comparativa libre entre agentes (PPO_1 vs SAC_1):
python scripts/generate_figures.py --phase custom --configs configs/phase_X/prueba/PPO_1.yaml,configs/phase_X/prueba/SAC_1.yaml

# Visualización de trayectorias con ventana de estabilización destacada:
python scripts/generate_figures.py --phase trajectories
```

---

## 6. Suite de Pruebas Unitarias

Ejecuta la suite completa de pruebas unitarias que verifica cinemática directa, Jacobiano analítico, cálculo exacto de recompensas `lyapunov_lqr` y `distance_only`, rechazo de modos obsoletos, truncamiento estricto a 200 pasos y normalización:

```powershell
python -m unittest discover tests
```

---

## 7. Estructura del Repositorio (Fase D)

```text
robot-arm-2dof-ppo/
├── arm2d_env.py                             # Entorno Gymnasium (MDP puro, lyapunov_lqr, 10D cinemático)
├── train.py                                 # Pipeline de entrenamiento PPO y SAC
├── interactive_render.py                    # Visualizador interactivo Pygame con telemetría LQR
├── configs/
│   └── phase_X/
│       ├── prueba/                          # PPO_1.yaml y SAC_1.yaml (agentes vigentes)
│       └── smoke/                           # s1_smoke.yaml y x3_smoke.yaml (fixtures de humo)
├── data/
│   └── datasets/                            # Conjuntos congelados de validación (200) y final (300)
├── src/
│   ├── controllers/
│   │   └── jacobian_controller.py           # Controlador cinemático C0 (Jacobiano amortiguado)
│   ├── evaluation/
│   │   ├── checkpoints.py                   # Runner de evaluación de checkpoints
│   │   ├── datasets.py                      # Carga y hashes de datasets congelados
│   │   ├── evaluator.py                     # Evaluador dual desacoplado (Opción 2 y Opción 3)
│   │   └── metrics.py                       # Constantes canónicas y funciones de métricas LQR
│   ├── training/
│   │   ├── callbacks.py                     # Callbacks de telemetría y sincronización
│   │   └── phase_b_callbacks.py             # Checkpointing periódico acoplado
│   └── visualization/
│       ├── common.py                        # Estilo y trazabilidad JSON
│       ├── phase_custom.py                  # Curvas de entrenamiento y comparativa entre agentes
│       └── trajectories.py                  # Trayectorias cartesianas y ventana de estabilización
├── scripts/
│   ├── evaluate_checkpoints.py              # CLI para evaluación de modelos
│   ├── generate_figures.py                  # CLI para figuras de reporte
│   └── tune_and_eval_controller.py          # Ajuste y benchmarking de C0
├── reports/                                 # Datos y figuras generadas
├── docs/                                    # Manual de uso y metodología
├── tests/                                   # Pruebas unitarias (test_kinematics, test_normalization)
└── requirements.txt                         # Dependencias fijadas
```