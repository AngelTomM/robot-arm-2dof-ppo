# Reporte Técnico de Implementación y Arquitectura del Sistema (Fase D)

Este informe detalla las decisiones de diseño, la reestructuración física del repositorio, la formulación matemática adoptada y la implementación técnica de la **Fase D (Proceso de Decisión de Markov Continuo con Regulación LQR)** sobre la planta física dinámica **MuJoCo Reacher-v5**. El propósito es servir como documento de referencia y alineación técnica para todos los integrantes del equipo de trabajo.

---

## 1. Diagnóstico del Estado Inicial y Objetivos de la Refactorización

Previo a la intervención, el repositorio presentaba deficiencias arquitectónicas y metodológicas que obstaculizaban la reproducibilidad, el mantenimiento y la validez formal del benchmarking:

1. **Parcial Observabilidad en la Función de Éxito (POMDP):** En versiones previas, la tarea de posicionamiento utilizaba un contador de permanencia (*hold counter*) de 20 pasos consecutivos dentro de una tolerancia para gatillar un bono terminal positivo y truncar el episodio. Dicho contador no formaba parte del vector de observación suministrado al agente, violando la propiedad markoviana y provocando oscilaciones e inestabilidades de convergencia en el crítico de las redes neuronales.
2. **Duplicidad de Archivos y Ruptura del Principio SSOT:** Existían archivos redundantes y desincronizados entre la raíz y los subdirectorios (por ejemplo, `arm2d_env.py` en la raíz como módulo puente y en `src/environment/arm2d_env.py`), así como réplicas inconsistentes de archivos de configuración (`configs/phase_D/environment.yaml` frente a `configs/environment.yaml`).
3. **Sobrecarga de Jerarquía de Carpetas:** Existían múltiples subdirectorios unitarios dentro de `src/` que contenían un único archivo (`controllers/`, `environment/`, `training/`, `utils/`, `visualization/`), así como directorios auxiliares como `scripts/` con responsabilidades superpuestas.
4. **Simulación Cinemática Simplificada:** El entorno se basaba en integración cinemática simplificada en Pygame con control por velocidad angular directa en lugar de simular la dinámica física del manipulador accionado por torque.

---

## 2. Desacoplamiento de Configuraciones y Fuente Única de Verdad (SSOT)

Se implementó una estricta separación de responsabilidades a nivel paramétrico:

* **Planta Física y Formulación MDP Común:** El archivo [configs/environment.yaml](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/configs/environment.yaml) es la Fuente Única de Verdad (SSOT). Define de manera centralizada la geometría de los eslabones, las masas inerciales, los límites de giro de las juntas, el paso de integración temporal, la escala del actuador de torque y los pesos del funcional cuadrático de recompensa. Ningún parámetro físico se encuentra codificado de forma estática en el código fuente.
* **Hiperparámetros de Agentes y Controladores:** Las configuraciones específicas de algoritmos ([configs/phase_D/prueba/](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/configs/phase_D/prueba/)) declaran exclusivamente la tasa de aprendizaje, arquitectura de redes neuronales, tamaño de lote, coeficientes de entropía o ganancias del controlador analítico.
* **Cargador Jerárquico Desacoplado:** El módulo [src/config.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/src/config.py) resuelve automáticamente las referencias relativas y fusiona la sección `env` en tiempo de carga, asegurando que cualquier cambio en la física del sistema se propague de manera instantánea y transparente a todos los experimentos.

---

## 3. Reorganización y Aplanamiento Modular del Paquete `src/`

Se consolidó una arquitectura aplanada sin anidamientos innecesarios, eliminando carpetas unitarias y scripts auxiliares. El paquete [src/](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/src/) agrupa directamente los ocho módulos del sistema, exponiendo su interfaz pública unificada en [src/\_\_init\_\_.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/src/__init__.py):

1. [src/arm2d_env.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/src/arm2d_env.py): Adaptador canónico de Gymnasium sobre `Reacher-v5` de MuJoCo con cálculo integrado de la recompensa continua cuadrática.
2. [src/jacobian_controller.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/src/jacobian_controller.py): Controlador cartesiano analítico PD con transposición del Jacobiano para generar comandos de torque normalizados en $[-1.0, 1.0]$.
3. [src/callbacks.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/src/callbacks.py): Callbacks especializados para registrar en TensorBoard el retorno medio, la distancia terminal cartesiana y sincronizar estadísticas de `VecNormalize` en validaciones periódicas.
4. [src/config.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/src/config.py): Resolvedor y cargador de configuraciones YAML desacopladas.
5. [src/datasets.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/src/datasets.py): Generador determinista de episodios de validación ($n = 200$) y evaluación final ($n = 300$) confinados al espacio de trabajo accesible del robot.
6. [src/evaluator.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/src/evaluator.py): Evaluador agnóstico de trayectorias que procesa datos físicos crudos para políticas RL o controladores clásicos.
7. [src/metrics.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/src/metrics.py): Cálculo formal de métricas mecatrónicas de régimen permanente ($e_{\text{ss}}, v_{\text{ss}}$), tiempo de establecimiento ($t_s$), costo LQR canónico e intervalos de Wilson.
8. [src/\_\_init\_\_.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/src/__init__.py): Punto de entrada que exporta todas las clases y funciones públicas hacia los scripts de la raíz y los bancos de prueba.

---

## 4. Migración al Motor Dinámico MuJoCo (`Reacher-v5`)

La planta física se migró al motor MuJoCo mediante el entorno de referencia `Reacher-v5` de Gymnasium:

* **Cinemática y Dimensiones:** Eslabón proximal $L_1 = 0.10\text{ m}$ y eslabón distal $L_2 = 0.11\text{ m}$ (incluyendo el cuerpo del efector final). El radio de alcance máximo es $0.21\text{ m}$.
* **Dinámica de Cuerpos Rígidos y Actuación:** La planta se acciona por torque continuo con una relación de transmisión $\text{gear} = 200.0\text{ N}\cdot\text{m}$. La acción emitida por los agentes $\mathbf{a}_t \in [-1.0, 1.0]^2$ excita directamente la dinámica articular de segundo orden.
* **Paso Temporal:** Paso de integración temporal de MuJoCo $\Delta t = 0.01\text{ s}$ con $\text{frame\_skip} = 2$, generando un período de muestreo efectivo de $\Delta t_{\text{eff}} = 0.02\text{ s}$ ($50\text{ Hz}$) y un horizonte continuo de 200 pasos ($T = 4.0\text{ s}$).
* **Cinemática Directa y Jacobiano Contrastados:** La posición cartesiana analítica y la matriz Jacobiana $J(\boldsymbol{\theta}) \in \mathbb{R}^{2 \times 2}$ fueron contrastadas unitariamente contra `mj_jacBody` y las matrices de transformación homogéneas de MuJoCo, obteniendo discrepancias numéricas nulas a nivel de precisión de máquina ($< 10^{-5}$).
* **Visualización 3D Nativa:** Se implementó soporte de renderizado OpenGL nativo mediante `render_mode="human"` en [interactive_render.py](file:///c:/Users/ulica/Documents/AprendizajeAuo2026/Fase_D_V3/interactive_render.py) y captura de frames mediante `render_mode="rgb_array"`.

---

## 5. Formulación Matemática de la Recompensa Continua Cuadrática

Se implementó una formulación inspirada en el Regulador Lineal Cuadrático (LQR) y en funciones de Lyapunov para control de posición:
$$r_t = - \left(w_{\text{pos}} \|\mathbf{x}_{ee} - \mathbf{x}_{\text{tgt}}\|^2 + w_{\text{ctrl}} \|\mathbf{a}_t\|^2 + w_{\text{vel}} \|\dot{\boldsymbol{\theta}}_t\|^2\right)$$
donde los parámetros de penalización son:

* $w_{\text{pos}} = 1.0$: Atracción cartesiana continua en todo el espacio de trabajo.
* $w_{\text{ctrl}} = 0.01$: Minimización del esfuerzo de torque aplicado para evitar saturación de actuadores.
* $w_{\text{vel}} = 0.05$: Disipación de energía cinética para forzar el frenado y estabilización terminal.

El entorno no posee condiciones de parada anticipada por proximidad (`success_mode: none`), manteniendo invariablemente `terminated = False` y truncando el episodio únicamente al agotarse los 200 pasos (`truncated = True`), garantizando la preservación estricta de la propiedad markoviana.

---

## 6. Rediseño del Baseline Analítico (`C0`)

El controlador clásico C0, previamente basado en control cinemático por velocidad, fue reformulado a un esquema por torque en espacio operacional mediante control proporcional-derivativo (PD) con transposición del Jacobiano:
$$\tau = J(\boldsymbol{\theta})^T \left(K_p \, (\mathbf{x}_{\text{tgt}} - \mathbf{x}_{ee})\right) - K_d \, \dot{\boldsymbol{\theta}}$$
$$\mathbf{a}_t = \text{clip}(\tau, -1.0, 1.0)$$
con ganancias $K_p = 50.0$ y $K_d = 1.5$. La saturación en $[-1.0, 1.0]$ somete al controlador analítico a las mismas restricciones físicas de saturación de torque que los agentes PPO y SAC.

---

## 7. Protocolo de Evaluación Mecatrónica y Métricas

La evaluación de modelos se ejecuta siempre sobre coordenadas físicas crudas sin normalizar:

1. **Métrica Principal — Estabilización Terminal (Últimos 20 Pasos):**
   $$\bar{e}_{\text{ss}} = \frac{1}{20} \sum_{t=181}^{200} \|\mathbf{x}_{ee}(t) - \mathbf{x}_{\text{tgt}}\| \le 0.02\text{ m} \quad \wedge \quad \bar{v}_{\text{ss}} = \frac{1}{20} \sum_{t=181}^{200} \|\dot{\boldsymbol{\theta}}(t)\| \le 0.10\text{ rad/s}$$
   El éxito se reporta junto a su intervalo de confianza binomial de Wilson al 95%.
2. **Métrica Comparativa — Costo Integral LQR Canónico:**
   $$J_{\text{LQR}} = -\sum_{t=1}^{200} \left(1.0 \cdot \|\mathbf{x}_{ee}(t) - \mathbf{x}_{\text{tgt}}\|^2 + 0.01 \cdot \|\mathbf{a}(t)\|^2 + 0.05 \cdot \|\dot{\boldsymbol{\theta}}(t)\|^2\right)$$
3. **Métricas Dinámicas y Energéticas:** Tiempo de establecimiento continuo ($t_s$), esfuerzo acumulado de torque ($J_u = \sum \|\mathbf{a}_t\|^2 \Delta t$) y variación cuadrática de acción ($J_{\Delta u}$).

---

## 8. Guía Operativa de Ejecución para el Equipo

Todos los scripts de flujo de trabajo se operan desde la raíz del proyecto:

```powershell
# 1. Asegurar activación del entorno virtual
..\.venv\Scripts\Activate.ps1
.\.venv\Scripts\Activate.ps1

# 2. Entrenamiento de Agentes (PPO y SAC)
python train.py --config configs/phase_D/prueba/PPO_1.yaml --seed 42
python train.py --config configs/phase_D/prueba/SAC_1.yaml --seed 42

# 3. Evaluación de Políticas Entrenadas y Baseline Analítico C0
python evaluate.py --c0 --dataset data/datasets/validation_set_200.json
python evaluate.py --agent ppo --model-path experiments/phase_D_prueba/PPO_1/seed_42/models/ppo_arm2d_final.zip --dataset data/datasets/final_eval_set_300.json

# 4. Generación de Figuras Comparativas
python plot.py --data-dir reports/data/evaluation --out-dir reports/figures

# 5. Visualizador Interactivo OpenGL 3D
python interactive_render.py --c0 --episodes 5
python interactive_render.py experiments/phase_D_prueba/PPO_1/seed_42/models/best_model/best_model.zip

# 6. Ejecución de la Suite de Pruebas Unitarias
python -m pytest tests/
```

Todas las 14 pruebas unitarias en `tests/` se encuentran operativas y superadas, asegurando la consistencia dimensional, paramétrica y algorítmica del proyecto.
