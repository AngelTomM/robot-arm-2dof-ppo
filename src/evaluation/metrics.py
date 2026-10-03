"""
Cálculo de Métricas Obligatorias, Intervalos de Wilson e Indicadores Operativos (Sección 9 y 16).
"""
import math
import numpy as np
import pandas as pd

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple:
    """
    Calcula el intervalo de confianza de Wilson al 95% para proporciones binomiales.
    Retorna (center, low, high).
    """
    if total <= 0:
        return 0.0, 0.0, 0.0
    
    z = 1.95996  # Para 95% de confianza
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

def compute_aggregate_metrics(episodes_df: pd.DataFrame, thresholds_cfg: dict) -> dict:
    """
    Calcula el conjunto completo de métricas agregadas obligatorias a partir de un DataFrame de episodios.
    """
    total = len(episodes_df)
    if total == 0:
        return {}

    op_thresh = thresholds_cfg.get("operational_thresholds", {})
    u_b = float(op_thresh.get("u_b", 0.30))
    n_o = int(op_thresh.get("n_exits_threshold", 2))
    n_c = int(op_thresh.get("n_cross_threshold", 3))

    # 1. Éxito estricto y Truncamientos
    successes = int(episodes_df["success"].sum())
    truncated = int(episodes_df["truncated"].sum())

    # 2. Indicadores Operativos (Sección 9.6)
    # Llegada balística: primer ingreso con joint_speed > u_b
    ballistic_mask = episodes_df["joint_speed_at_first_entry"].apply(lambda v: (v is not None) and (not np.isnan(v)) and (v > u_b))
    n_ballistic = int(ballistic_mask.sum())

    # Episodio oscilatorio: n_exits >= n_o O n_cross >= n_c
    n_exits_col = "n_exits" if "n_exits" in episodes_df.columns else "n_exits_after_first_entry"
    n_cross_col = "n_cross" if "n_cross" in episodes_df.columns else "n_error_sign_crossings"
    oscillatory_mask = (episodes_df[n_exits_col] >= n_o) | (episodes_df[n_cross_col] >= n_c)
    n_oscillatory = int(oscillatory_mask.sum())

    # 3. Pasos hasta el éxito (solo episodios exitosos)
    success_df = episodes_df[episodes_df["success"] == True]
    n_success = len(success_df)
    if n_success > 0:
        steps_median = float(success_df["steps_to_success"].median())
        steps_mean = float(success_df["steps_to_success"].mean())
        steps_p90 = float(success_df["steps_to_success"].quantile(0.90))
    else:
        steps_median, steps_mean, steps_p90 = np.nan, np.nan, np.nan

    # 4. Distancia Final
    final_dist = episodes_df["final_distance"]
    dist_mean = float(final_dist.mean())
    dist_median = float(final_dist.median())
    dist_p90 = float(final_dist.quantile(0.90))
    dist_max = float(final_dist.max())

    # 5. Velocidades articulares y cartesianas
    entry_joint_speeds = episodes_df["joint_speed_at_first_entry"].dropna()
    success_joint_speeds = episodes_df["joint_speed_at_success"].dropna()
    mean_entry_joint_speed = float(entry_joint_speeds.mean()) if len(entry_joint_speeds) > 0 else np.nan
    mean_success_joint_speed = float(success_joint_speeds.mean()) if len(success_joint_speeds) > 0 else np.nan

    # 6. Suavidad (delta_action_norm medio por episodio)
    action_deltas = episodes_df["mean_action_delta"].dropna()
    mean_action_delta = float(action_deltas.mean()) if len(action_deltas) > 0 else np.nan
    median_action_delta = float(action_deltas.median()) if len(action_deltas) > 0 else np.nan
    p90_action_delta = float(action_deltas.quantile(0.90)) if len(action_deltas) > 0 else np.nan

    # 7. Tiempo en zona d < 0.30m
    time_near = episodes_df["time_inside_030"]
    mean_time_near = float(time_near.mean())

    return {
        "total_episodes": total,
        "successful_episodes": successes,
        "success_rate_ci": format_ci(successes, total),
        "success_rate": successes / total,
        "truncated_episodes": truncated,
        "truncation_rate_ci": format_ci(truncated, total),
        "truncation_rate": truncated / total,
        "ballistic_episodes": n_ballistic,
        "ballistic_rate_ci": format_ci(n_ballistic, total),
        "ballistic_rate": n_ballistic / total,
        "oscillatory_episodes": n_oscillatory,
        "oscillatory_rate_ci": format_ci(n_oscillatory, total),
        "oscillatory_rate": n_oscillatory / total,
        "median_steps_to_success": steps_median,
        "mean_steps_to_success": steps_mean,
        "p90_steps_to_success": steps_p90,
        "final_dist_mean": dist_mean,
        "final_dist_median": dist_median,
        "final_dist_p90": dist_p90,
        "final_dist_max": dist_max,
        "mean_joint_speed_entry": mean_entry_joint_speed,
        "mean_joint_speed_success": mean_success_joint_speed,
        "mean_action_delta": mean_action_delta,
        "median_action_delta": median_action_delta,
        "p90_action_delta": p90_action_delta,
        "mean_time_inside_030": mean_time_near
    }
