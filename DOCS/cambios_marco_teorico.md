# Registro de cambios — Marco Teórico (`Informe Trabajo Final.md`)

Fecha: 19-oct-2026 · Base: la versión que estaba en Google Docs.

Leyenda de cada ítem:
- **ANTES** = texto original que se eliminó o se modificó (lo que vas a marcar como *sugerencia de borrado* en Docs).
- **DESPUÉS** = texto nuevo que se agregó (lo que vas a insertar).
- Los ítems sin "ANTES" son solo inserciones.

---

## 1. Referencias cruzadas base

**DESDE LA PRIMERA PÁGINA (sección 1, primer párrafo):**

DESPUÉS: se agregó la cita \[9\] a la primera oración.

> Un manipulador robótico **\[9\]** se define como una cadena cinemática abierta...

---

## 2. Encabezado de sección 1 (ancla rota)

ANTES (el anchor del título apuntaba erróneamente a la sección 2):

> #### **1. Fundamentos de Manipuladores Robóticos y Cinemática Directa** {#2.-proceso-de-decisión-de-markov-(mdp)-y-aprendizaje-por-refuerzo-profundo-(drl)}

DESPUÉS:

> #### **1. Fundamentos de Manipuladores Robóticos y Cinemática Directa** {#1.-fundamentos-de-manipuladores-robóticos-y-cinemática-directa}

(En Google Docs este ancla no existe; solo aplica al .md/exportación).

---

## 3. Bloque duplicado ELIMINADO en sección 1 (después de la Figura "Espacio cartesiano de trabajo", antes de 1.2)

Se borró TODO este bloque (era una repetición de la definición de q, θ1 y θ2):

> ---
>
> Donde $Q\ \subset {ℝ}^{n}{\ }$denota el **espacio articular** (*joint space*). Para una articulación rotacional (revoluta), ${q}_{i}$ representa el ángulo de giro ${\theta }_{i}\in [-\pi ,\pi ]$, mientras que para una prismática define un desplazamiento lineal ${d}_{i}$.
>
> El sistema bajo estudio consiste en un brazo robótico planar de cadena cinemática abierta formado por dos eslabones rígidos conectados mediante dos pares cinemáticos de rotación pura (2 GDL). El estado de configuración del mecanismo se describe en el **espacio articular** $Q\subset {ℝ}^{2}$ por el vector:
>
> ![][image5]
>
> Donde:
>
> * ${\theta }_{1}\in [-\pi ,\pi )$ denota el ángulo de orientación absoluto del primer eslabón respecto al semieje positivo $+X$ del marco inercial fijo $\{{F}_{0}\}$ situado en la base del robot.
> * ${\theta }_{2}\in [-\pi ,\pi )$ representa el ángulo articular relativo del segundo eslabón respecto a la prolongación del primer eslabón.
>
> Los eslabones poseen longitudes constantes ${L}_{1}$ y ${L}_{2}$.
>
> ---

⚠️ Nota: incluye la imagen "image5" que solo existía en este bloque duplicado; si en el Google Doc esa imagen es única, movela junto al bloque 1.1 en lugar de borrarla.

---

## 4. Sección 2.1 — señal de recompensa

ANTES:

> 3. Un paso de tiempo después, el entorno transiciona a un nuevo estado ${S}_{t+1}\in S$ y devuelve una señal escalar de recompensa ${R}_{t+1}\in R\subset R$.

DESPUÉS:

> 3. Un paso de tiempo después, el entorno transiciona a un nuevo estado ${S}_{t+1}\in S$ y devuelve una señal escalar de recompensa ${R}_{t+1}\in \mathbb{R}$.

---

## 5. Sección 2.2 — tupla del MDP

ANTES:

> * $P:S\times A\times S\rightarrow$ es la función de probabilidad de transición de estados condicional, caracterizada por la densidad de probabilidad $p(s',r|s,a)$.
>
> * $R:S\times A\times S\rightarrow R$ denota la función escalar de recompensa inmediata que evalúa el desempeño de cada transición.

DESPUÉS:

> * $P$ denota la función de probabilidad de transición de estados, caracterizada por la densidad de probabilidad condicional $p(s',r\mid s,a)$ que asigna a cada par estado-acción una distribución sobre el estado sucesor y la recompensa.
>
> * $R:S\times A\times S\rightarrow \mathbb{R}$ denota la función escalar de recompensa inmediata que evalúa el desempeño de cada transición.

---

## 6. Sección 2.2 — T_max genérico

ANTES:

> ...estructura episódica de horizonte finito con un límite de pasos temporales ${T}_{\max}=200$. El objetivo consiste en encontrar los parámetros óptimos ${\Phi }^{*}$ de una política estocástica ${\pi }_{\Phi }=({a}_{t}|{o}_{t})$ que maximicen...

DESPUÉS:

> ...estructura episódica de horizonte finito con un límite de pasos temporales ${T}_{\max}$. El objetivo consiste en encontrar los parámetros óptimos ${\Phi }^{*}$ de una política estocástica ${\pi }_{\Phi }$ que maximicen...

---

## 7. Sección 2.3 — dimensión de observaciones genérica

ANTES:

> ...sino un vector de observaciones sensoriales ${o}_{t}\in O\subset {R}^{10}$. Para evitar discontinuidades numéricas..., **las variables articulares no se ingresan en radianes directamente, sino proyectadas**...

DESPUÉS (dos cambios):

1. La dimensión pasa de 10 a genérica:

> ...sino un vector de observaciones sensoriales ${o}_{t}\in O\subset {\mathbb{R}}^{{n}_{o}}$. Para evitar discontinuidades numéricas..., **es una práctica estándar no ingresar las variables articulares en radianes directamente, sino proyectadas**...

2. Insertar, después de la fórmula de las observaciones (image14), el siguiente párrafo:

> La dimensión exacta de ${o}_{t}$ y la composición definitiva de sus componentes constituyen decisiones de diseño del agente que se detallan en el capítulo de desarrollo.

---

## 8. Sección 2.3 — cierre del análisis de Markovianidad (POMDP)

ANTES (última oración del párrafo):

> ...el proceso opera formalmente como un Proceso de Decisión de Markov Parcialmente Observable (POMDP) con dependencia temporal de orden unitario, requiriendo que la red neuronal aproxime el progreso de estabilización de manera implícita a través de la evolución temporal de las observaciones continuas.

DESPUÉS:

> ...el proceso opera formalmente como un Proceso de Decisión de Markov Parcialmente Observable (POMDP) con dependencia temporal de orden unitario. Frente a esta situación existen dos estrategias generales: ampliar el vector de estados (por ejemplo, apilando observaciones o contadores previos) para restaurar la propiedad de Markov, o directamente aceptar la parcial observabilidad y dejar que la política aproxime el progreso de estabilización de manera implícita a través de la evolución temporal de las observaciones continuas. La alternativa adoptada en este trabajo se detalla en el capítulo de desarrollo.

---

## 9. Sección 2.4 — dt y v_max genéricos + orden de argumentos de P

ANTES:

> Las transiciones de estado en el simulador cinemático operan de forma determinista mediante integración numérica de Euler hacia adelante con un paso temporal $dt=0.05s$:

DESPUÉS:

> ...con un paso temporal $dt$ dado:

ANTES:

> con una cota superior de velocidad de consigna ${v}_{\max}=1.0\,rad/s$. La ausencia de matrices dinámicas acopladas de segundo orden simplifica el operador de transición $P({s}_{t+1},{s}_{t},{a}_{t})$ a una evolución...

DESPUÉS:

> con una cota superior de velocidad de consigna ${v}_{\max}$. La ausencia de matrices dinámicas acopladas de segundo orden simplifica el operador de transición $P({s}_{t+1}\mid {s}_{t},{a}_{t})$ a una evolución...

---

## 10. Sección 2.5 — truncamiento sin valor hard-codeado

ANTES:

> * Truncamiento (truncated): Interrupción forzada del episodio al alcanzar el límite del horizonte de ${T}_{max}=200$ pasos sin haber logrado...

DESPUÉS:

> * Truncamiento (truncated): Interrupción forzada del episodio al alcanzar el límite del horizonte de ${T}_{max}$ pasos sin haber logrado...

---

## 11. Sección 2.5 — párrafo NUEVO de taxonomía de recompensas (inserción)

Insertar inmediatamente después de la fórmula ${R}_{t}={R}_{dist}+{R}_{action}+{R}_{success}$:

> Desde la perspectiva del diseño, las señales de recompensa para tareas de posicionamiento admiten una taxonomía general: (i) recompensas **densas**, que entregan retroalimentación en cada paso a través de términos de distancia de forma lineal $-{c}_{d}\cdot d$ o cuadrática $-{c}_{d}\cdot {d}^{2}$, frente a recompensas **dispersas** que solo consideran el resultado terminal del episodio; (ii) penalizaciones de **esfuerzo** y de **suavidad** del control, basadas en la magnitud del comando o en su variación diferencial $\Delta {a}_{t}$; (iii) **bonificaciones de proximidad**, que densifican la guía geométrica en la cercanía de la zona de tolerancia; y (iv) técnicas de **moldeado de recompensa** (*reward shaping*) \[10\], que agregan términos basados en una función de potencial para acelerar el aprendizaje sin alterar la política óptima resultante. La combinación específica de términos, coeficientes y umbrales de éxito es una decisión de diseño y constituye uno de los objetos de estudio experimental del capítulo de desarrollo.

---

## 12. Sección 2.5 → 2.6 — renumeración del título y dim genérica

ANTES:

> **2.5. Transición a Aprendizaje por Refuerzo Profundo (DRL)**
>
> Dado que el espacio de observaciones $O\subset {R}^{10}$ y el espacio de acciones... mapeando el vector ${o}_{t}\in {ℝ}^{10}$ directamente...

DESPUÉS:

> **2.6. Transición a Aprendizaje por Refuerzo Profundo (DRL)**
>
> Dado que el espacio de observaciones $O\subset {\mathbb{R}}^{{n}_{o}}$ y el espacio de acciones... mapeando el vector ${o}_{t}\in {\mathbb{R}}^{{n}_{o}}$ directamente...

---

## 13. Sección 3 — título con tipeo corregido

ANTES:

> #### **3. Redes neuronales de perceptrón multicapa y marco Actor-Critico**

DESPUÉS:

> #### **3. Redes neuronales de perceptrón multicapa y marco Actor-Crítico**

---

## 14. Sección 3.3 — párrafo NUEVO sobre Adam (inserción)

Insertar después del bullet "Regla Delta de Actualización":
Acción de control..." (end of bullet list)... y Antes de "3.4. Estimación de Funciones de Valor":

> En la práctica moderna del DRL, la regla delta se implementa mediante optimizadores adaptativos, en particular **Adam** (Adaptive Moment Estimation) \[3\], que mantiene estimaciones exponenciales del primer y del segundo momento del gradiente y las emplea para escalar adaptativamente la tasa de aprendizaje de cada parámetro. Su robustez frente a gradientes de magnitudes heterogéneas lo convierte en el optimizador por defecto de los algoritmos de RL profundo; tanto PPO como SAC se entrenan con Adam en este trabajo.

---

## 15. Sección 3.4 — Q(s,a) + Bellman de optimidad (inserciones)

Después de la Ecuación de Expectativa de Bellman (definida solo con V), insertar:

> * Función de Valor de Estado-Acción (${Q}_{\pi }(s,a)$): Esperanza del retorno acumulado descontado al ejecutar la acción $a$ en el estado $s$ y seguir posteriormente la política $\pi$:
>
>   ${Q}_{\pi }(s,a)={E}_{\pi }\ \left[{\ \sum\limits_{k=0}^{\infty }{\gamma }^{k}\ {R}_{t+k+1}\ |\ {S}_{t}=s,\ {A}_{t}=a}\right]$
>
>   Entre ambas funciones de valor se cumple la relación ${V}_{\pi }(s)={E}_{a\sim \pi }{[Q}_{\pi }(s,\ a)]$. En los algoritmos off-policy, como SAC, es precisamente ${Q}_{\pi }(s,\ a)$ —y no el valor de estado ${V}$— la función de valor central, ya que el objetivo de Bellman se construye sobre pares estado-acción almacenados en el *replay buffer*.
>
> * **Ecuaciones de Bellman de optimalidad**: más allá de la consistencia expectativa, el valor óptimo —el máximo retorno esperado alcanzable desde cada estado— satisface el sistema recursivo:
>
>   ${V}^{*}(s)=\max\limits_{a}{E}_{\ }\left[{r+\gamma \ {V}^{*}(s')}\right],\qquad {Q}^{*}(s,\ a)={E}_{\ }\left[{r+\gamma \ \max\limits_{a'}{Q}^{*}(s',\ a')}\right]$
>
>   Ambos operadores de optimalidad son contracciones con un único punto fijo; esta propiedad fundamenta los métodos de programación dinámica clásicos y las aproximaciones neuronales mediante diferenciación temporal, presentes en todos los algoritmos actor-critic considerados.

---

## 16. Sección 3.5 — Actor genérico

ANTES:

> El Actor es la red neuronal parametrizada por el vector de pesos $\theta$ encargada de la toma de decisiones. Procesa la observación ${o}_{t}$ y genera en su capa de salida los comandos continuos de velocidad angular ${\mu }_{\theta }({o}_{t})=[{\mu }_{1},{\mu }_{2}{]}^{T}\in [-1,1{]}^{2}$ para las articulaciones. Al operar en un espacio de acciones continuo $A\subset [-1,1{]}^{2}$, la red del Actor no emite un valor discreto, sino los parámetros de una distribución probabilística Gaussiana multivariable:

DESPUÉS:

> El Actor es la red neuronal parametrizada por el vector de pesos $\theta$ encargada de la toma de decisiones. Procesa la observación ${o}_{t}$ y genera en su capa de salida los parámetros de la distribución de política sobre el espacio de acciones continuo $A\subset {\mathbb{R}}^{\dim (A)}$; la red del Actor no emite un valor discreto de control, sino los parámetros de una distribución probabilística Gaussiana multivariable:

ANTES (bullet "Capa de salida"):

> * **Capa de salida**: Predice el vector de medias angulares ${\mu }_{\theta }({o}_{t})=[{\mu }_{1},{\mu }_{2}{]}^{T}$ para las velocidades de las dos articulaciones, mientras que la desviación estándar $\sigma$ regula el nivel de exploración.

DESPUÉS:

> * **Capa de salida**: Predice el vector de medias ${\mu }_{\theta }({o}_{t})\in {\mathbb{R}}^{\dim (A)}$ que parametriza la distribución de la política, mientras que la desviación estándar $\sigma$ regula el nivel de exploración.

ANTES (Crítico):

> El Crítico es una red neuronal independiente parametrizada por el vector de pesos $\phi$. Recibe la observación del entorno ${o}_{t}\in {R}^{10}$ y estima...

DESPUÉS:

> ...Recibe la observación del entorno ${o}_{t}\in {\mathbb{R}}^{{n}_{o}}$ y estima...

---

## 17. Sección 3.5 — figura duplicada eliminada

ANTES (había dos copias seguidas):

> ![][image19]
> Figura N° x. Arquitectura del marco Actor-Crítico.
>
> ##### ![][image20] Figura N° x. Arquitectura del marco Actor-Crítico.
>
> Fuente: tomado de \[7\].

DESPUÉS (solo queda una):

> ##### ![][image20] Figura N° x. Arquitectura del marco Actor-Crítico.
>
> Fuente: tomado de \[7\].

(En el Google Doc: borrar UNA de las dos copias de la figura y su caption.)

---

## 18. Sección 3.5 — tipeo "valor11"

ANTES:

> ...para reducir el error cuadrático medio de su predicción de valor11:

DESPUÉS:

> ...para reducir el error cuadrático medio de su predicción de valor:

---

## 19. FIN de sección 3 — bloque duplicado ELIMINADO y sección 3.6 NUEVA

Se borró TODO este bloque duplicado (era un segundo borrador resumido de las secciones 3.1-3.5):

> ---
>
> La unidad de procesamiento fundamental es el perceptrón, el cual recibe un vector de entradas, calcula una combinación lineal ponderada mediante pesos $\left({{w}_{i}}\right)$ y un término de sesgo o bias $\left({{w}_{0}}\right)$, y posteriormente aplica una función de activación para generar una salida.
>
> $z={w}_{0}+\sum\limits_{i=1}^{n}{w}_{i}{x}_{i}$
>
> El entrenamiento de una red neuronal consiste en determinar los valores óptimos de estos pesos para minimizar el error de predicción. Esto se logra mediante algoritmos de optimización basados en el descenso del gradiente y la retropropagación.
>
> Las redes neuronales pueden presentarse en múltiples arquitecturas que interconectan las neuronas de diferentes maneras. Una de ellas es el Perceptrón Multicapa (Multilayer Perceptron \- MLP) que es una arquitectura de red neuronal feedforward (alimentación hacia adelante) compuesta por una capa de entrada, una o más capas ocultas de neuronas y una capa de salida.
>
> Esta estructura puede resolver problemas que no sean linealmente separables introduciendo capas con funciones de activación no lineales como la Tangente Hiperbolica (tanh) o la Unidad Lineal Rectificada (RELU).
>
> El marco Actor-Crítico es un paradigma de aprendizaje por refuerzo que divide el aprendizaje del agente en dos módulos funcionales cooperativos:
>
> * El Actor $\left({{\pi }_{\phi }}\right)$: Es el componente encargado de parametrizar la política de control. Recibe la observación o estado del entorno $\left({{o}_{t}}\right)$ y selecciona o propone la acción $\left({{a}_{t}}\right)$ a ejecutar.
> * El Crítico (${V}_{\phi }$ o ${Q}_{\phi }$): Es el módulo que evalúa la calidad de las decisiones tomadas. Su función es estimar la función de valor, prediciendo el retorno acumulado esperado que se obtendrá desde ese estado.
>
> Matemáticamente, el Actor define una distribución sobre el espacio continuo de acciones condicionada a la observación actual:
>
> ${a}_{t}\ ~\ {π}_{θ}\left({{a}_{t}\ |\ {o}_{t}}\right)\ =\ N\left({{μ}_{θ}\left({{o}_{t}}\right),\ σ²}\right){\ }$
>
> mientras que el Crítico aproxima el retorno esperado acumulado mediante la llamada función de valor:
>
> ${V}_{ϕ}\left({{o}_{t}}\right)\ =\ {E}_{π}\ \left[{\sum\limits_{k=0}^{∞}{γ}^{k}\cdot {r}_{t+k+1}\ \ |\ \ {o}_{t}\ }\right]{\ }$
>
> La interacción y sincronización entre ambos módulos se articula a través de la Función de Ventaja:
>
> $A\ \left({{o}_{t},\ aₜ}\right)\ =\ {Q}_{ϕ}\left({{o}_{t},\ aₜ}\right)-{V}_{ϕ}\left({{o}_{t}}\right)\ ≈\ {r}_{t+1}+γ\ ·{\ V}_{ϕ}\left({{o}_{t+1}}\right){-V}_{ϕ}\left({{o}_{t}}\right)\ {\ }$
>
> la cual cuantifica si la acción ejecutada aₜ superó el rendimiento promedio esperado en la observación oₜ, modulando así el gradiente de actualización de los pesos de la red del Actor.
>
> En algoritmos modernos de RL continuo (como PPO o SAC), tanto el actor como el crítico suelen implementarse mediante dos redes de perceptrón multicapa independientes y desacopladas, evitando que ambas funciones compitan por la misma capacidad de representación dentro de la red.

En su lugar, se INSERTÓ la subsección nueva 3.6:

> **3.6. Estimación de la Ventaja: Error de Diferencia Temporal y GAE**
>
> La señal de ventaja $\widehat{A}$ que necesita el Actor puede estimarse mediante dos extremos teóricos inversos:
>
> * El **retorno Monte Carlo** (la suma efectivamente observada de recompensas descontadas a lo largo del episodio completo) es un estimador no sesgado respecto del valor real de la política, pero de alta varianza, pues una única trayectoria puede desviarse significativamente del promedio.
>
> * El **error de diferencia temporal de un paso** (TD error), ${\delta }_{t}={r}_{t+1}+\gamma {V}_{\phi }({o}_{t+1})-{V}_{\phi }({o}_{t})$, incorpora el *bootstrap* del crítico: reduce la varianza a costa de introducir sesgo, ya que depende de la calidad del propio ${V}_{\phi }$.
>
> La **Estimación de Ventaja Generalizada** (Generalized Advantage Estimation, GAE) \[11\] interpola ambos extremos mediante la suma ponderada de errores TD:
>
> $\widehat{A}{\ }^{GAE(\gamma ,\lambda )}_{t}=\sum\limits_{k=0}^{\infty }{(\gamma \lambda )}^{k}{\delta }_{t+k}$
>
> donde $\lambda \in [0,\ 1]$ modula el compromiso sesgo-varianza: con $\lambda =0$ se recupera el estimador de un paso ($\widehat{A}_{t}={\delta }_{t}$), mientras que con $\lambda \rightarrow 1$ el estimador tiende al retorno Monte Carlo.

---

## 20. Sección 4 — reestructurada en subsecciones (va antes de PPO)

El título **4. Algoritmos PPO y SAC** ahora tiene esta secuencia:

**NUEVA 4.1:**

> **4.1. Conceptos preliminares: entropía y divergencia de Kullback-Leibler**
>
> Dos nociones de la teoría de la información aparecen de forma transversal en el diseño de algoritmos de política:
>
> * **Entropía de la política**: Para una distribución discreta, la entropía de Shannon de la distribución de acciones en el estado $s$ se define como $H\left(\pi (\cdot |s)\right)=-\sum\limits_{a}^{}\pi (a|s)\ \ln \pi (a|s)$ (y su análogo integral para distribuciones continuas). La entropía cuantifica la dispersión o incertidumbre de la distribución de acciones: valores elevados reflejan una política exploratoria, mientras que valores próximos a cero corresponden a una política casi determinista.
>
> * **Divergencia de Kullback-Leibler (KL)**: Cuantifica la discrepancia entre dos distribuciones $p$ y $q$ definidas sobre el mismo soporte:
>
> ${D}_{KL}(p\ \parallel \ q)=\sum\limits_{x}^{}p(x)\ \log \frac{p(x)}{q(x)}$
>
>   Es no negativa y se anula únicamente cuando ambas distribuciones coinciden. No es una métrica (no es simétrica ni cumple la desigualdad triangular), pero resulta adecuada para comparar distribuciones de probabilidad de políticas. En el RL continuo, se emplea para medir y acotar la magnitud del cambio entre políticas consecutivas —base de la restricción de región de confianza de TRPO— y en la actualización del actor de SAC, que la minimiza respecto de la distribución del valor Q.

**NUEVA 4.2:**

> **4.2. Familias de Algoritmos: On-Policy vs. Off-Policy**
>
> Según el origen de los datos utilizados para evaluar y mejorar la política, los algoritmos actor-critic se clasifican en dos familias:
>
> * **On-policy**: los datos con los que se computa el gradiente provienen de la misma política que se está optimizando. Tras un número limitado de épocas de optimización sobre el lote de trayectorias recién recolectado, los datos se descartan para evitar sesgo por desactualización. Esta frescura estabiliza el aprendizaje al costo de una baja eficiencia muestral. Es el caso de PPO.
>
> * **Off-policy**: la política de comportamiento que genera los datos queda desacoplada de la política objetivo que se optimiza. Esto permite reutilizar experiencias pasadas almacenadas en una memoria de repetición (*replay buffer*), aumentando la eficiencia muestral, pero introduce el riesgo de la denominada *triada fatal* (*deadly triad*: bootstrap, aproximación de funciones y datos fuera de política empleados simultáneamente) \[2\], que motiva mecanismos de estabilización como las redes objetivo y los críticos dobles. Es el caso de SAC.

**RENUMERADOS:**
- "**Proximal Policy Optimization (PPO)**" → "**4.3. Proximal Policy Optimization (PPO)**"
- "**Algoritmo Soft Actor-Critic (SAC)**" → "**4.4. Algoritmo Soft Actor-Critic (SAC)**"

**Ajustes menores dentro de 4.3 (PPO):**

ANTES:

> Donde ${\widehat{A}}_{t}$ representa la estimación de la función de ventaja mediante el método de Estimación de Ventaja Generalizada (Generalized Advantage Estimation, GAE), y ϵ es el hiperparámetro...

DESPUÉS:

> ...Estimación de Ventaja Generalizada (Generalized Advantage Estimation, GAE), **descrito en la sección 3.6,** y ϵ es el hiperparámetro...

ANTES:

> ...integración del objetivo recortado, la regresión del crítico... y una medida de entropía $S\left[{{\pi }_{\theta }}\right]$ que previene...

DESPUÉS:

> ...y una medida de entropía $S\left[{{\pi }_{\theta }}\right]$ **—conforme a la noción de la sección 4.1—** que previene...

**Ajustes dentro de 4.4 (SAC):**

ANTES:

> ...y $H(\cdot )$ denota la entropía de la distribución (El punto indica que no se especifica una acción concreta). Este principio...

DESPUÉS:

> ...y $H(\cdot )$ denota la entropía de la distribución (El punto indica que no se especifica una acción concreta) **—según la definición formal de la sección 4.1—**. Este principio...

INSERTAR después de "seleccionando el valor mínimo entre ambas estimaciones durante las actualizaciones del Bellman suave.":

> Adicionalmente, los críticos cuentan con **redes objetivo** (*target networks*): copias retardadas de los parámetros, que se actualizan de forma suave mediante interpolación de **Polyak** ${\overline{\theta }}\leftarrow \tau \theta +\left(1-\tau \right)\overline{\theta }$, con $\tau \ll 1$, lo cual amortigua el bootstrap del objetivo de Bellman y estabiliza el entrenamiento.

Y en la oración del actor de SAC:

ANTES:

> Por su parte, la actualización de la red del actor ${\pi }_{\theta }$ minimiza la divergencia Kullback-Leibler respecto a la distribución del valor $Q$.

DESPUÉS:

> Por su parte, la actualización de la red del actor ${\pi }_{\theta }$ minimiza la divergencia Kullback-Leibler **—sección 4.1—** respecto a la distribución del valor $Q$.

---

## 21. Referencias Bibliográficas

ANTES (referencia huérfana, sin número):

> \[\]	J. J. Craig, Introduction to Robotics: Mechanics and Control, 4th ed. Pearson, 2018.

DESPUÉS:

> \[9\]	J. J. Craig, Introduction to Robotics: Mechanics and Control, 4th ed. Pearson, 2018.
>
> \[10\]	A. Y. Ng, D. Harada y S. Russell, "Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping," in Proc. 16th Int. Conf. Machine Learning (ICML), 1999, pp. 278-287.
>
> \[11\]	J. Schulman, P. Moritz, S. Levine, M. Jordan y P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," in Proc. 4th Int. Conf. Learning Representations (ICLR), 2016. \[En línea\]. Disponible: https://arxiv.org/abs/1506.02438

---

## Resumen rápido (checklist para el Google Doc)

| # | Dónde | Acción |
|---|-------|--------|
| 1 | Sec. 1, 1er párrafo | Agregar \[9\] |
| 2 | Bloque post-figura 𝒲 | BORRAR bloque duplicado (q, θ1, θ2) |
| 3 | 2.1 / 2.2 | Corregir ∈ ℝ, P y R del MDP |
| 4 | 2.2 y 2.5 (truncación) | Quitar "=200" de T_max |
| 5 | 2.3 | ℝ¹⁰ → ℝ^{n_o} + párrafo nuevo post-fórmula |
| 6 | 2.3 POMDP | Reescribir oración final |
| 7 | 2.4 | Quitar "dt=0.05s" y "v_max=1.0 rad/s"; orden P |
| 8 | 2.5 | Insertar párrafo de taxonomía tras la fórmula |
| 9 | 2.5 DRL | Renumerar a 2.6; ℝ¹⁰ → ℝ^{n_o} |
| 10 | Sec. 3 título | "Actor-Critico" → "Actor-Crítico" |
| 11 | 3.3 | Insertar párrafo Adam |
| 12 | 3.4 | Insertar Q(s,a) + Bellman de optimidad |
| 13 | 3.5 | Actor genérico; crítico ℝ^{n_o}; borrar figura duplicada; "valor11" |
| 14 | Fin sec. 3 | BORRAR bloque duplicado; INSERTAR 3.6 (GAE) |
| 15 | Sec. 4 | Insertar 4.1 (entropía/KL) y 4.2 (on/off-policy); renumerar PPO=4.3, SAC=4.4; 3 ajustes menores |
| 16 | Referencias | \[\] → \[9\]; agregar \[10\] y \[11\] |
