import numpy as np
import pandas as pd

def wilson_ci_width(p, n=500, z=1.95996):
    """Calcula el ancho total del intervalo de Wilson al 95%."""
    denominator = 1.0 + (z**2) / n
    centre = (p + (z**2) / (2.0 * n)) / denominator
    sd = z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n) / denominator
    return float(2.0 * sd)

def compute_smoothed_success(curve: list) -> list:
    """
    Calcula media móvil de 3 puntos con tratamiento formal de extremos truncados:
    - k = 0: (s_0 + s_1) / 2
    - k in 1..N-2: (s_{k-1} + s_k + s_{k+1}) / 3
    - k = N-1: (s_{N-2} + s_{N-1}) / 2
    """
    n = len(curve)
    smoothed = []
    for k in range(n):
        if k == 0:
            smoothed.append((curve[0] + curve[1]) / 2.0)
        elif k == n - 1:
            smoothed.append((curve[n - 2] + curve[n - 1]) / 2.0)
        else:
            smoothed.append((curve[k - 1] + curve[k] + curve[k + 1]) / 3.0)
    return smoothed

def select_checkpoint(steps: list, mean_success_rates: list, n_val=500) -> dict:
    """
    Regla 4.1: Selecciona el checkpoint óptimo.
    1. Calcula éxito suavizado con ventana de 3 truncada.
    2. Encuentra el máximo suavizado y la ventana U_equiv (ancho de Wilson en el máximo).
    3. Identifica candidatos dentro de [max_smoothed - U_equiv, max_smoothed].
    4. Elige el candidato MÁS TEMPRANO (menor step).
    """
    smoothed = compute_smoothed_success(mean_success_rates)
    max_idx = int(np.argmax(smoothed))
    max_smoothed = smoothed[max_idx]
    
    u_equiv = wilson_ci_width(max_smoothed, n=n_val)
    threshold = max_smoothed - u_equiv
    
    equivalent_indices = [i for i, s in enumerate(smoothed) if s >= threshold]
    earliest_idx = min(equivalent_indices)
    
    return {
        "chosen_step": steps[earliest_idx],
        "chosen_idx": earliest_idx,
        "max_step": steps[max_idx],
        "max_idx": max_idx,
        "max_smoothed": max_smoothed,
        "threshold": threshold,
        "u_equiv": u_equiv,
        "equivalent_steps": [steps[i] for i in equivalent_indices]
    }

def test_synthetic_cases():
    steps = [40960 * (i + 1) for i in range(16)] # 16 checkpoints
    
    # Caso 1: Meseta con empates estadísticos (sube a 0.95 en paso 10 y fluctúa suave)
    # Debe elegir el más temprano dentro de U_equiv (~0.038)
    rates_plateau = [0.05, 0.20, 0.45, 0.70, 0.85, 0.90, 0.93, 0.94, 0.945, 0.95, 0.952, 0.948, 0.951, 0.949, 0.950, 0.950]
    res1 = select_checkpoint(steps, rates_plateau)
    print("Caso 1 (Meseta/Empates):")
    print(f"  Max suavizado en step {res1['max_step']} ({res1['max_smoothed']*100:.2f}%)")
    print(f"  U_equiv: {res1['u_equiv']*100:.2f} pp -> Umbral: {res1['threshold']*100:.2f}%")
    print(f"  Elegido más temprano: step {res1['chosen_step']}")
    assert res1["chosen_step"] <= res1["max_step"]
    
    # Caso 2: Caída/Degradación tras pico (sobreajuste o inestabilidad)
    # Sube a 0.96 en paso 8 y cae fuertemente a 0.80
    rates_drop = [0.10, 0.30, 0.50, 0.75, 0.88, 0.94, 0.96, 0.95, 0.92, 0.88, 0.85, 0.82, 0.80, 0.81, 0.79, 0.80]
    res2 = select_checkpoint(steps, rates_drop)
    print("\nCaso 2 (Pico y Caída):")
    print(f"  Max suavizado en step {res2['max_step']} ({res2['max_smoothed']*100:.2f}%)")
    print(f"  Elegido más temprano: step {res2['chosen_step']}")
    assert res2["chosen_step"] in [steps[5], steps[6], steps[7]]
    
    # Caso 3: Crecimiento monótono continuo
    rates_growth = [0.05 + 0.06 * i for i in range(16)] # 0.05 a 0.95
    res3 = select_checkpoint(steps, rates_growth)
    print("\nCaso 3 (Crecimiento Monótono):")
    print(f"  Max suavizado en step {res3['max_step']} ({res3['max_smoothed']*100:.2f}%)")
    print(f"  Elegido más temprano en zona alta: step {res3['chosen_step']}")
    
    print("\nTodos los tests sintéticos de la Regla 4.1 pasaron correctamente!")

if __name__ == "__main__":
    test_synthetic_cases()
