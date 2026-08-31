"""
Herramienta Avanzada de Comparación y Resumen de Experimentos.
Lee los archivos de metadatos (reports/runs/*.json y models/**/experiment_meta.json)
y los logs de TensorBoard para unificar automáticamente los resultados de todos los miembros del equipo.
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
        "Pasos / Episodio (Prom)": "N/D"
    }

    if not event_files:
        return default_metrics

    try:
        latest_log = max(event_files, key=os.path.getmtime)
        ea = event_accumulator.EventAccumulator(latest_log)
        ea.Reload()

        scalars = ea.Tags().get("scalars", [])

        # 1. Recompensa periódica de evaluación
        if "eval/mean_reward" in scalars:
            eval_rewards = [e.value for e in ea.Scalars("eval/mean_reward")]
            if eval_rewards:
                default_metrics["Recompensa Máx (Best)"] = f"{max(eval_rewards):.2f} *"
        elif "rollout/ep_rew_mean" in scalars:
            rewards = [e.value for e in ea.Scalars("rollout/ep_rew_mean")]
            if rewards:
                default_metrics["Recompensa Máx (Best)"] = f"{max(rewards):.2f}"

        # 2. Longitud media de episodio
        if "rollout/ep_len_mean" in scalars:
            lengths = [e.value for e in ea.Scalars("rollout/ep_len_mean")]
            if lengths:
                default_metrics["Pasos / Episodio (Prom)"] = f"{lengths[-1]:.1f}"

    except Exception:
        pass

    return default_metrics


def generate_experiment_summary():
    print("=" * 85)
    print(" RESUMEN Y COMPARACION AVANZADA DE RENDIMIENTO DE MODELOS")
    print("=" * 85)

    # 1. Buscar metadatos compartidos en reports/runs/ y modelos locales en models/
    shared_runs = glob.glob("reports/runs/*.json")
    local_meta_files = glob.glob("models/**/experiment_meta.json", recursive=True)

    all_meta_sources = set(shared_runs + local_meta_files)

    if not all_meta_sources:
        print(">>> No se encontraron experimentos en 'reports/runs/' ni en 'models/'.")
        return

    records = {}

    for mf in all_meta_sources:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)

            params = data.get("parameters", {})
            exp = params.get("experiment", {})
            env = params.get("env", {})
            ppo = params.get("ppo", {})
            train = params.get("training", {})

            exp_name = exp.get("name", "sin_nombre")
            author = exp.get("author", "Anonimo")
            timestamp = data.get("timestamp", "N/A")

            # Clave única para evitar colisiones: nombre + autor
            unique_key = f"{exp_name}_{author}"

            # Extraer métricas de TensorBoard local si existen, o usar las guardadas en el json
            metrics = extract_metrics_from_logs(exp_name)
            if metrics["Recompensa Máx (Best)"] == "N/D" and "metrics" in data:
                metrics["Recompensa Máx (Best)"] = data["metrics"].get("Recompensa Máx (Best)", "N/D")
                metrics["Pasos / Episodio (Prom)"] = data["metrics"].get("Pasos / Episodio (Prom)", "N/D")

            records[unique_key] = {
                "Experimento": exp_name,
                "Autor": author,
                "Recompensa": env.get("reward_type", "distance_only"),
                "Max Speed": env.get("max_speed", 2.0),
                "Radio Meta": env.get("target_radius", 0.08),
                "LR": ppo.get("learning_rate", 3e-4),
                "n_steps": ppo.get("n_steps", 1024),
                "Timesteps": train.get("total_timesteps", 150000),
                "Recompensa Máx": metrics["Recompensa Máx (Best)"],
                "Pasos/Episodio": metrics["Pasos / Episodio (Prom)"],
                "Fecha": timestamp
            }
        except Exception as e:
            print(f"Error procesando {mf}: {e}")

    df = pd.DataFrame(list(records.values()))

    # Ordenar por Recompensa Máx o Fecha si están disponibles
    print(df.to_string(index=False))

    # Guardar en Markdown dentro de reports/
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    out_md = os.path.join(reports_dir, "EXPERIMENTS_SUMMARY.md")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# 📊 Comparación de Experimentos\n\n")
        f.write("Resumen consolidado de parámetros de configuración y métricas de desempeño por experimento de todos los autores.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n---\n")
        f.write("### 💡 Interpretación de Métricas:\n")
        f.write("- **Recompensa Máx**: Mejor retorno medio obtenido durante la evaluación del modelo (valores más cercanos a 0 o positivos indican mejor desempeño).\n")
        f.write("- **Pasos/Episodio**: Duración media del episodio (un valor menor indica resolución más rápida de la tarea).\n")

    print(f"\n>>> [OK] Reporte consolidado actualizado en: {out_md}")


if __name__ == "__main__":
    generate_experiment_summary()
