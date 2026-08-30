"""
Herramienta Avanzada de Comparación y Resumen de Experimentos.
Lee los archivos experiment_meta.json y extrae automáticamente las métricas de rendimiento reales
de los logs de TensorBoard (Recompensa Máxima, Recompensa Final, Longitud Media de Episodio, etc.).
Genera una tabla detallada en consola y un reporte estructurado en reports/EXPERIMENTS_SUMMARY.md.
"""

import os
import glob
import json
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def extract_metrics_from_logs(exp_name):
    """Extrae métricas reales clave desde los eventos de TensorBoard en logs/<exp_name>/."""
    log_dir = f"logs/{exp_name}"
    event_files = glob.glob(f"{log_dir}/**/events.out.tfevents.*", recursive=True)

    default_metrics = {
        "Recompensa Máx (Best)": "N/D",
        "Recompensa Final": "N/D",
        "Pasos / Episodio (Prom)": "N/D",
        "Estado": "Sin Logs"
    }

    if not event_files:
        return default_metrics

    try:
        # Cargar el archivo de logs más reciente del experimento
        latest_log = max(event_files, key=os.path.getmtime)
        ea = event_accumulator.EventAccumulator(latest_log)
        ea.Reload()

        scalars = ea.Tags().get("scalars", [])

        # 1. Recompensa media de episodios (ep_rew_mean)
        if "rollout/ep_rew_mean" in scalars:
            rewards = [e.value for e in ea.Scalars("rollout/ep_rew_mean")]
            if rewards:
                default_metrics["Recompensa Máx (Best)"] = f"{max(rewards):.2f}"
                default_metrics["Recompensa Final"] = f"{rewards[-1]:.2f}"

        # 2. Longitud de episodios (ep_len_mean: pasos antes de terminar)
        if "rollout/ep_len_mean" in scalars:
            lengths = [e.value for e in ea.Scalars("rollout/ep_len_mean")]
            if lengths:
                default_metrics["Pasos / Episodio (Prom)"] = f"{lengths[-1]:.1f}"

        # 3. Evaluación periódica de EvalCallback si existe
        if "eval/mean_reward" in scalars:
            eval_rewards = [e.value for e in ea.Scalars("eval/mean_reward")]
            if eval_rewards:
                default_metrics["Recompensa Máx (Best)"] = f"{max(eval_rewards):.2f} *"

        default_metrics["Estado"] = "Completado"
    except Exception as e:
        default_metrics["Estado"] = "Error lectura"

    return default_metrics


def generate_experiment_summary():
    print("=" * 85)
    print(" RESUMEN Y COMPARACION AVANZADA DE RENDIMIENTO DE MODELOS")
    print("=" * 85)

    meta_files = glob.glob("models/**/experiment_meta.json", recursive=True)

    if not meta_files:
        print(">>> No se encontraron metadatos en 'models/'. Ejecuta entrenamientos primero.")
        return

    records = []
    for mf in meta_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)

            params = data.get("parameters", {})
            exp = params.get("experiment", {})
            env = params.get("env", {})
            ppo = params.get("ppo", {})
            train = params.get("training", {})

            exp_name = exp.get("name", os.path.basename(os.path.dirname(mf)))

            # Extraer métricas reales de TensorBoard
            metrics = extract_metrics_from_logs(exp_name)

            records.append({
                "Experimento": exp_name,
                "Autor": exp.get("author", "N/A"),
                "Recompensa": env.get("reward_type", "distance_only"),
                "Max Speed": env.get("max_speed", 2.0),
                "Radio Meta": env.get("target_radius", 0.08),
                "LR": ppo.get("learning_rate", 3e-4),
                "n_steps": ppo.get("n_steps", 1024),
                "Timesteps": train.get("total_timesteps", 150000),
                "Recompensa Máx": metrics["Recompensa Máx (Best)"],
                "Pasos/Episodio": metrics["Pasos / Episodio (Prom)"],
                "Fecha": data.get("timestamp", "N/A")
            })
        except Exception as e:
            print(f"Error procesando {mf}: {e}")

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    # Guardar en Markdown dentro de reports/
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    out_md = os.path.join(reports_dir, "EXPERIMENTS_SUMMARY.md")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Comparación de Experimentos\n\n")
        f.write("Resumen consolidado de parámetros de configuración y métricas de desempeño por experimento.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n---\n")
        f.write("- **Recompensa Máx**: Mejor retorno medio obtenido durante la evaluación del modelo.\n")
        f.write("- **Pasos/Episodio**: Duración media del episodio (un valor menor indica resolución más rápida de la tarea).\n")

    print(f"\n>>> [OK] Reporte actualizado en: {out_md}")


if __name__ == "__main__":
    generate_experiment_summary()
