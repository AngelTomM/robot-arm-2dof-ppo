"""
Métricas de Evaluación para Fase D (MDP Continuo con Regulación LQR).
Define las constantes canónicas y el cálculo de la evaluación dual:
- Opción 2 (Principal): Estabilización asintótica en los últimos k=20 pasos.
- Opción 3 (Canónica): Retorno/costo acumulado integral LQR independiente del reward de entrenamiento.
"""

import math
import numpy as np
import pandas as pd

# ==============================================================================
# Constantes Canónicas de Evaluación para Fase D (MDP LQR)
# ==============================================================================
K_STAB = 20                   # Ventana de estabilización: 20 pasos finales (t in [T-19, T])
TARGET_RADIUS_EVAL = 0.06     # Umbral de distancia media final (m)
VEL_TOL_EVAL = 0.10           # Umbral de velocidad articular media final (rad/s)

W_POS_EVAL = 1.0              # Ponderación canónica de error cartesiano ||p_tgt - p_ee||^2
W_VEL_EVAL = 0.05             # Ponderación canónica de disipación cinética ||q_dot||^2
W_CTRL_EVAL = 0.01            # Ponderación canónica de esfuerzo de control ||a||^2


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple:
    """
    Calcula el intervalo de confianza de Wilson al 95% para proporciones binomiales.
    Retorna (center, low, high).
    """
    if total <= 0:
        return 0.0, 0.0, 0.0

    z = 1.95996  # 95% confianza
    p_hat = successes / total
    z2 = z ** 2

    denominator = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denominator
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * (total ** 2)))) / denominator

    low = max(0.0, center - spread)
    high = min(1.0, center + spread)
    return p_hat, low, high


def format_ci(successes: int, total: int) -> str:
    """Formatea la proporción y el intervalo de Wilson como 'XX.X% [YY.Y%, ZZ.Z%]'."""
    p_hat, low, high = wilson_score_interval(successes, total)
    return f"{p_hat * 100:.1f}% [{low * 100:.1f}%, {high * 100:.1f}%] ({successes}/{total})"


def compute_lqr_canonical_cost(distances: list, joint_speeds: list, actions: list) -> dict:
    """
    Calcula el costo LQR canónico acumulado y su desglose sobre trayectorias físicas crudas:
      J_lqr = - sum(W_POS_EVAL * d^2 + W_VEL_EVAL * v^2 + W_CTRL_EVAL * ||a||^2)
    """
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


def compute_stabilization_metrics(distances: list, joint_speeds: list, k: int = K_STAB) -> dict:
    """
    Calcula la métrica principal de estabilización en los últimos k pasos:
      mean_dist_k <= TARGET_RADIUS_EVAL (0.06 m) y mean_vel_k <= VEL_TOL_EVAL (0.10 rad/s)
    """
    k_steps = min(len(distances), k)
    dist_k = distances[-k_steps:]
    vel_k = joint_speeds[-k_steps:]

    mean_dist = float(np.mean(dist_k))
    mean_vel = float(np.mean(vel_k))
    is_success = bool(mean_dist <= TARGET_RADIUS_EVAL and mean_vel <= VEL_TOL_EVAL)

    return {
        "mean_dist_last_k": mean_dist,
        "mean_vel_last_k": mean_vel,
        "is_success": is_success,
        "k_steps": k_steps
    }


def compute_aggregate_metrics(episodes_df: pd.DataFrame) -> dict:
    """
    Calcula métricas agregadas obligatorias a partir del DataFrame de episodios evaluados.
    """
    total = len(episodes_df)
    if total == 0:
        return {}

    successes = int(episodes_df["success"].sum()) if "success" in episodes_df else 0
    p_hat, ci_low, ci_high = wilson_score_interval(successes, total)

    final_dist = episodes_df["final_distance"]
    mean_dist_k = episodes_df["mean_dist_last_20"] if "mean_dist_last_20" in episodes_df else final_dist
    mean_vel_k = episodes_df["mean_vel_last_20"] if "mean_vel_last_20" in episodes_df else episodes_df.get("final_joint_speed", pd.Series([0.0]*total))
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
        "final_dist_mean": float(final_dist.mean()),
        "final_dist_median": float(final_dist.median()),
        "final_dist_p90": float(final_dist.quantile(0.90)),
        "mean_dist_last_20_mean": float(mean_dist_k.mean()),
        "mean_dist_last_20_median": float(mean_dist_k.median()),
        "mean_vel_last_20_mean": float(mean_vel_k.mean()),
        "mean_vel_last_20_median": float(mean_vel_k.median()),
        "j_lqr_mean": float(j_lqr.mean()),
        "j_lqr_median": float(j_lqr.median()),
        "j_lqr_std": float(j_lqr.std(ddof=1)) if total > 1 else 0.0,
        "mean_action_delta": mean_delta_a
    }
