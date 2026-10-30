"""
Módulo de Métricas de Evaluación Mecatrónica y Control Óptimo (Fase D).

Calcula formalmente:
  1. Error estacionario (e_ss): ||p_target - p_ee|| en régimen permanente.
  2. Velocidad residual (v_ss): ||q_dot|| en régimen permanente.
  3. Tiempo de establecimiento (t_s): tiempo físico continuo en el cual el efector
     ingresa a la banda de tolerancia sin volver a salir.
  4. Esfuerzo / Costo de control (J_u): integral del esfuerzo cuadrático sum(||u_t||^2 * dt).
  5. Costo Canónico LQR (J_lqr): funcional cuadrático de control óptimo continuo.
  6. Tasa de éxito e intervalo de confianza de Wilson (95%).
"""
from typing import List, Dict, Any, Tuple
import math
import numpy as np
import pandas as pd

# Tolerancias Canónicas del Sistema Mecatrónico (MuJoCo Reacher-v5)
K_STAB: int = 20                   # Ventana de evaluación estacionaria final (pasos)
TARGET_RADIUS_EVAL: float = 0.02     # Radio de tolerancia cartesiana (m)
VEL_TOL_EVAL: float = 0.10           # Tolerancia de velocidad articular residual (rad/s)

# Coeficientes Canónicos del Funcional LQR
W_POS_EVAL: float = 1.0
W_VEL_EVAL: float = 0.05
W_CTRL_EVAL: float = 0.01


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calcula el intervalo de confianza de Wilson al 95% para proporciones binomiales."""
    if total <= 0:
        return 0.0, 0.0, 0.0

    z = 1.95996  # Nivel bilateral 95%
    p_hat = successes / total
    z2 = z ** 2

    denominator = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denominator
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * (total ** 2)))) / denominator

    low = max(0.0, center - spread)
    high = min(1.0, center + spread)
    return p_hat, low, high


def format_ci(successes: int, total: int) -> str:
    """Formatea la proporción y el intervalo de confianza de Wilson: 'XX.X% [YY.Y%, ZZ.Z%]'."""
    p_hat, low, high = wilson_score_interval(successes, total)
    return f"{p_hat * 100:.1f}% [{low * 100:.1f}%, {high * 100:.1f}%] ({successes}/{total})"


def compute_settling_time(distances: List[float], target_radius: float = TARGET_RADIUS_EVAL,
                          dt: float = 0.02) -> Tuple[float, int, bool]:
    """
    Calcula el tiempo de establecimiento t_s según la definición formal de control:
    Instante t_s a partir del cual el sistema ingresa a la banda |e| <= target_radius
    y permanece confinado en ella hasta el término del horizonte T.

    Retorna: (t_s_segundos, k_paso_establecimiento, is_settled)
    """
    n = len(distances)
    if n == 0:
        return 0.0, 0, False

    # Buscar desde el final hacia el inicio el último instante que violó la tolerancia
    last_exit_idx = -1
    for i in range(n - 1, -1, -1):
        if distances[i] > target_radius:
            last_exit_idx = i
            break

    if last_exit_idx == -1:
        # Estuvo dentro de la banda durante todo el episodio
        return 0.0, 0, True
    elif last_exit_idx == n - 1:
        # El episodio finalizó fuera de la banda de tolerancia
        return float(n * dt), n, False
    else:
        # Ingresó de manera sostenida en last_exit_idx + 1
        k_settle = last_exit_idx + 1
        t_s = float(k_settle * dt)
        return t_s, k_settle, True


def compute_control_effort(actions: List[np.ndarray], dt: float = 0.02) -> float:
    """
    Calcula el esfuerzo de control acumulado en el episodio:
      J_u = sum(||u_t||^2 * dt)
    """
    if len(actions) == 0:
        return 0.0
    acts = np.array(actions, dtype=np.float64)
    effort_sq = np.sum(acts ** 2, axis=-1)
    return float(np.sum(effort_sq * dt))


def compute_lqr_canonical_cost(distances: List[float], joint_speeds: List[float],
                               actions: List[np.ndarray]) -> Dict[str, float]:
    """Calcula el costo cuadrático acumulado LQR sobre la trayectoria física continua."""
    dists = np.array(distances, dtype=np.float64)
    speeds = np.array(joint_speeds, dtype=np.float64)
    acts = np.array(actions, dtype=np.float64)

    j_pos = float(np.sum(W_POS_EVAL * (dists ** 2)))
    j_vel = float(np.sum(W_VEL_EVAL * (speeds ** 2)))
    j_ctrl = float(np.sum(W_CTRL_EVAL * np.sum(acts ** 2, axis=-1)))
    j_lqr = float(-(j_pos + j_vel + j_ctrl))

    return {
        "j_lqr": j_lqr,
        "j_pos": j_pos,
        "j_vel": j_vel,
        "j_ctrl": j_ctrl
    }


def compute_stabilization_metrics(distances: List[float], joint_speeds: List[float],
                                  k: int = K_STAB) -> Dict[str, Any]:
    """
    Calcula las métricas en régimen permanente sobre los últimos k pasos:
      e_ss = mean(dist_k), v_ss = mean(vel_k)
      is_success = (e_ss <= TARGET_RADIUS_EVAL) and (v_ss <= VEL_TOL_EVAL)
    """
    k_steps = min(len(distances), k)
    dist_k = distances[-k_steps:]
    vel_k = joint_speeds[-k_steps:]

    e_ss = float(np.mean(dist_k))
    v_ss = float(np.mean(vel_k))
    is_success = bool(e_ss <= TARGET_RADIUS_EVAL and v_ss <= VEL_TOL_EVAL)

    return {
        "e_ss": e_ss,
        "v_ss": v_ss,
        "mean_dist_last_k": e_ss,
        "mean_vel_last_k": v_ss,
        "is_success": is_success,
        "k_steps": k_steps
    }


def compute_aggregate_metrics(episodes_df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula métricas agregadas normalizadas a partir de un DataFrame de episodios."""
    total = len(episodes_df)
    if total == 0:
        return {}

    successes = int(episodes_df["success"].sum()) if "success" in episodes_df else 0
    p_hat, ci_low, ci_high = wilson_score_interval(successes, total)

    e_ss_series = episodes_df["e_ss"] if "e_ss" in episodes_df else episodes_df["final_distance"]
    v_ss_series = episodes_df["v_ss"] if "v_ss" in episodes_df else episodes_df.get("final_joint_speed", pd.Series([0.0]*total))
    ts_series = episodes_df["t_s"] if "t_s" in episodes_df else pd.Series([0.0]*total)
    ju_series = episodes_df["j_u"] if "j_u" in episodes_df else pd.Series([0.0]*total)
    j_lqr = episodes_df["j_lqr"] if "j_lqr" in episodes_df else pd.Series([0.0]*total)

    mean_delta_a = (float(episodes_df["mean_action_delta"].dropna().mean())
                    if "mean_action_delta" in episodes_df and len(episodes_df["mean_action_delta"].dropna()) > 0
                    else np.nan)

    return {
        "total_episodes": total,
        "successful_episodes": successes,
        "success_rate": successes / total if total > 0 else 0.0,
        "success_rate_ci": format_ci(successes, total),
        "ci_low": ci_low,
        "ci_high": ci_high,
        # Error estacionario e_ss (m)
        "e_ss_mean": float(e_ss_series.mean()),
        "e_ss_median": float(e_ss_series.median()),
        "e_ss_p90": float(e_ss_series.quantile(0.90)),
        "final_dist_mean": float(episodes_df["final_distance"].mean()),
        # Velocidad residual v_ss (rad/s)
        "v_ss_mean": float(v_ss_series.mean()),
        "v_ss_median": float(v_ss_series.median()),
        # Tiempo de establecimiento t_s (s)
        "t_s_mean": float(ts_series.mean()),
        "t_s_median": float(ts_series.median()),
        # Esfuerzo de control J_u
        "j_u_mean": float(ju_series.mean()),
        # Costo LQR canónico
        "j_lqr_mean": float(j_lqr.mean()),
        "j_lqr_median": float(j_lqr.median()),
        "j_lqr_std": float(j_lqr.std(ddof=1)) if total > 1 else 0.0,
        "mean_action_delta": mean_delta_a
    }
