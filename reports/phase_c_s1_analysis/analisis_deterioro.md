# Cuantificación de la Magnitud y Consistencia del Deterioro — SAC S1

**Fuente de datos:** `reports/phase_c_s1_analysis/metricas_checkpoints.csv`  
**Dataset de evaluación:** `data/datasets/phase_c/validation_c_500.json` (500 episodios deterministas por checkpoint)  
**Semillas analizadas:** 42, 100, 2026 ($N=3$)  
**Puntos comparados:** Checkpoint de máximo síncrono ($327\,680$ pasos) vs Checkpoint final ($655\,360$ pasos).

---

## 1. Métricas Agregadas Síncronas

* **Tasa media de éxito en $327\,680$ pasos:** $52{,}4000\% \pm 6{,}0100\%$ ($\text{ddof}=1$)
* **Tasa media de éxito en $655\,360$ pasos:** $46{,}6000\% \pm 9{,}2065\%$ ($\text{ddof}=1$)
* **Diferencia absoluta en puntos porcentuales:**
  $$\Delta_{\text{abs}} = \mu_{\text{final}} - \mu_{\text{max}} = 46{,}60\% - 52{,}40\% = -5{,}8000\text{ pp}$$
* **Cambio relativo porcentual:**
  $$\Delta_{\text{rel}} = \frac{\mu_{\text{final}} - \mu_{\text{max}}}{\mu_{\text{max}}} \times 100 = \frac{46{,}60 - 52{,}40}{52{,}40} \times 100 = -11{,}0687\% \approx -11{,}07\%$$

*(Nota: Si se toma como referencia alternativa el paso $286\,720$ con $\mu = 52{,}1333\%$, la diferencia absoluta es $-5{,}5333\text{ pp}$ y el cambio relativo es $-10{,}6138\%$).*

---

## 2. Desglose Individual por Semilla ($655\,360$ vs $327\,680$)

| Semilla | Paso $327\,680$ (%) | Paso $655\,360$ (%) | Diferencia Absoluta ($\Delta_{\text{abs}}$) | Cambio Relativo ($\Delta_{\text{rel}}$) |
| :---: | :---: | :---: | :---: | :---: |
| **42** | $59{,}20\%$ | $40{,}60\%$ | **$-18{,}60\text{ pp}$** | **$-31{,}42\%$** |
| **100** | $50{,}20\%$ | $57{,}20\%$ | **$+7{,}00\text{ pp}$** | **$+13{,}94\%$** |
| **2026** | $47{,}80\%$ | $42{,}00\%$ | **$-5{,}80\text{ pp}$** | **$-12{,}13\%$** |

---

## 3. Evolución de la Dispersión ($\sigma$, $\text{ddof}=1$) entre Semillas

| Checkpoint | Pasos Entorno | Media Éxito Síncrono (%) | Desv. Estándar $\sigma$ (%) |
| :--- | :---: | :---: | :---: |
| `ckpt_40960` | $40\,960$ | $28{,}07$ | $7{,}27$ |
| `ckpt_81920` | $81\,920$ | $40{,}20$ | $3{,}70$ |
| `ckpt_122880` | $122\,880$ | $43{,}27$ | $4{,}09$ |
| `ckpt_163840` | $163\,840$ | $47{,}80$ | $6{,}00$ |
| `ckpt_204800` | $204\,800$ | $49{,}13$ | $4{,}09$ |
| `ckpt_245760` | $245\,760$ | $49{,}40$ | $9{,}17$ |
| `ckpt_286720` | $286\,720$ | $52{,}13$ | $8{,}08$ |
| **`ckpt_327680` (Pico Síncrono)** | **$327\,680$** | **$52{,}40$** | **$6{,}01$** |
| `ckpt_368640` | $368\,640$ | $48{,}47$ | $6{,}18$ |
| `ckpt_409600` | $409\,600$ | $49{,}47$ | $4{,}27$ |
| `ckpt_450560` | $450\,560$ | $50{,}27$ | $3{,}24$ |
| `ckpt_491520` | $491\,520$ | $48{,}40$ | $2{,}36$ *(mínima dispersión)* |
| `ckpt_532480` | $532\,480$ | $47{,}73$ | $5{,}03$ |
| `ckpt_573440` | $573\,440$ | $48{,}47$ | $3{,}53$ |
| `ckpt_614400` | $614\,400$ | $45{,}67$ | $4{,}44$ |
| **`ckpt_655360` (Final)** | **$655\,360$** | **$46{,}60$** | **$9{,}21$** *(máxima dispersión)* |

* **Dispersión media a lo largo de los 16 checkpoints:** $\overline{\sigma} = 5{,}42\%$
* **Dispersión en el pico síncrono ($327\,680$):** $\sigma = 6{,}01\%$
* **Dispersión en el cierre ($655\,360$):** $\sigma = 9{,}21\%$

---

## 4. Análisis Crítico: Magnitud y Consistencia

### 4.1. Resultado Descriptivo
La media síncrona disminuye $-5{,}80\text{ pp}$ (un cambio relativo de $-11{,}07\%$). Sin embargo, la magnitud de esta diferencia es **inferior a la desviación estándar entre semillas en el punto final ($9{,}21\%$)** y prácticamente igual a la desviación estándar en el punto máximo ($6{,}01\%$). Asimismo, fluctuaciones intra-corrida de magnitud similar o superior ($> 6\text{ pp}$) ocurren de forma rutinaria entre checkpoints vecinos (por ejemplo, la caída de $4{,}80\text{ pp}$ observada entre los pasos $245\,760$ y $286\,720$ en la semilla 2026, o la fluctuación de más de $15\text{ pp}$ en la semilla 100).

### 4.2. Evidencia de Robustez
La tendencia **no es consistente entre las tres semillas**:
* La semilla 42 experimenta una degradación severa ($-18{,}60\text{ pp}$, $-31{,}42\%$).
* La semilla 2026 experimenta una degradación moderada ($-5{,}80\text{ pp}$, $-12{,}13\%$).
* La semilla 100 sigue la dirección opuesta, incrementando su rendimiento en $+7{,}00\text{ pp}$ ($+13{,}94\%$).

Al presentar comportamientos divergentes de signo opuesto, el deterioro no puede catalogarse como una respuesta sistemática y homogénea del algoritmo en las condiciones evaluadas.

### 4.3. Limitaciones Metodológicas
1. **Tamaño muestral insuficiente para inferencia causal:** Con $N=3$ réplicas independientes, no es metodológicamente válido realizar pruebas de contraste de hipótesis (e.g., t-test o Wilcoxon) ni calcular intervalos de confianza asintóticos.
2. **Dependencia temporal:** Las observaciones de los 16 checkpoints de una misma semilla corresponden a una serie temporal autocorrelacionada (trayectoria de optimización no estacionaria), por lo que tratar los checkpoints sucesivos como observaciones independientes constituiría un error metodológico.
3. **Imposibilidad de aislar la causa:** La coexistencia de un aumento en $\|\Delta \mathbf{a}\|$ y un descenso de la media agregada no demuestra causalidad; podría reflejar dinámicas estocásticas propias del paisaje de optimización de SAC en presencia de una red Q sobreajustada o inestabilidades de gradiente no amortiguadas.
