"""
Herramienta Avanzada de Comparación y Resumen de Experimentos.
Lee los archivos de metadatos (reports/runs/*.json y models/**/experiment_meta.json)
y los logs de TensorBoard para unificar y ordenar automáticamente los resultados del equipo.
Soporta ordenamiento configurable (--sort reward | author | date | steps) y filtrado.
"""

import os
import glob
import json
import argparse
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def extract_metrics_from_logs(exp_name):
    """Extrae métricas reales clave desde los eventos de TensorBoard en logs/<exp_name>/."""
    log_dir = f"logs/{exp_name}"
    event_files = glob.glob(f"{log_dir}/**/events.out.tfevents.*", recursive=True)

    default_metrics = {
        "raw_reward": -9999.0,
        "Recompensa Máx": "N/D",
        "Pasos/Episodio": "N/D"
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
                max_r = max(eval_rewards)
                default_metrics["raw_reward"] = max_r
                default_metrics["Recompensa Máx"] = f"{max_r:.2f} *"
        elif "rollout/ep_rew_mean" in scalars:
            rewards = [e.value for e in ea.Scalars("rollout/ep_rew_mean")]
            if rewards:
                max_r = max(rewards)
                default_metrics["raw_reward"] = max_r
                default_metrics["Recompensa Máx"] = f"{max_r:.2f}"

        # 2. Longitud media de episodio
        if "rollout/ep_len_mean" in scalars:
            lengths = [e.value for e in ea.Scalars("rollout/ep_len_mean")]
            if lengths:
                default_metrics["Pasos/Episodio"] = f"{lengths[-1]:.1f}"

    except Exception:
        pass

    return default_metrics


def generate_experiment_summary(sort_by="reward", include_unregistered=False):
    print("=" * 85)
    print(" RESUMEN Y COMPARACION AVANZADA DE RENDIMIENTO DE MODELOS")
    print("=" * 85)

    # 1. Fuente oficial de experimentos: reports/runs/*.json
    meta_sources = glob.glob("reports/runs/*.json")

    # Si se activa el flag --all, también incluye modelos locales de models/ no registrados en runs/
    if include_unregistered:
        local_meta_files = glob.glob("models/**/experiment_meta.json", recursive=True)
        meta_sources = list(set(meta_sources + local_meta_files))

    if not meta_sources:
        print(">>> No se encontraron experimentos en 'reports/runs/'.")
        print(">>> (Usa 'python compare_experiments.py --all' si quieres escanear todos los modelos locales en models/)")
        return

    records = {}

    for mf in meta_sources:
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

            unique_key = f"{exp_name}_{author}"

            metrics = extract_metrics_from_logs(exp_name)

            records[unique_key] = {
                "Experimento": exp_name,
                "Autor": author,
                "Recompensa": env.get("reward_type", "distance_only"),
                "Max Speed": env.get("max_speed", 2.0),
                "Radio Meta": env.get("target_radius", 0.08),
                "LR": ppo.get("learning_rate", 3e-4),
                "n_steps": ppo.get("n_steps", 1024),
                "Timesteps": train.get("total_timesteps", 150000),
                "raw_reward": metrics["raw_reward"],
                "Recompensa Máx": metrics["Recompensa Máx"],
                "Pasos/Episodio": metrics["Pasos/Episodio"],
                "Fecha": timestamp
            }
        except Exception as e:
            print(f"Error procesando {mf}: {e}")

    df = pd.DataFrame(list(records.values()))

    # --- ORDENAMIENTO CONFIGURABLE ---
    if sort_by == "reward":
        df = df.sort_values(by="raw_reward", ascending=False)
    elif sort_by == "author":
        df = df.sort_values(by=["Autor", "raw_reward"], ascending=[True, False])
    elif sort_by == "date":
        df = df.sort_values(by="Fecha", ascending=False)
    elif sort_by == "steps":
        df = df.sort_values(by="Timesteps", ascending=False)

    # Eliminar columna auxiliar numérica usada para ordenar
    df_display = df.drop(columns=["raw_reward"])

    print(df_display.to_string(index=False))

    # Guardar en Markdown dentro de reports/
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    out_md = os.path.join(reports_dir, "EXPERIMENTS_SUMMARY.md")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# 📊 Comparación de Experimentos\n\n")
        f.write(f"Resumen consolidado de parámetros y métricas (Ordenado por: **{sort_by.upper()}**).\n\n")
        f.write(df_display.to_markdown(index=False))
        f.write("\n\n---\n")
        f.write("### 💡 Interpretación de Métricas:\n")
        f.write("- **Recompensa Máx**: Mejor retorno medio obtenido (valores más cercanos a 0 o positivos = mejor modelo).\n")
        f.write("- **Pasos/Episodio**: Duración media del episodio (menor valor = resolución más rápida).\n")

    print(f"\n>>> [OK] Reporte actualizado (orden: {sort_by}) en: {out_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidador y comparador de experimentos")
    parser.add_argument(
        "--sort", "-s",
        type=str,
        default="reward",
        choices=["reward", "author", "date", "steps"],
        help="Criterio de ordenamiento: reward (por desempeño mayor a menor), author (por autor), date (más reciente), steps (más timesteps)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Incluir todos los modelos locales en models/ aunque hayan sido eliminados de reports/runs/"
    )
    args = parser.parse_args()

    generate_experiment_summary(sort_by=args.sort, include_unregistered=args.all)
