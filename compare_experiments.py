"""
Herramienta de Comparación y Resumen de Experimentos.
Lee los archivos experiment_meta.json de cada modelo y los logs de TensorBoard para generar
una tabla resumen en consola y en un archivo markdown (EXPERIMENT_SUMMARY.md).
"""

import os
import glob
import json
import pandas as pd


def generate_experiment_summary():
    print("=" * 70)
    print(" RESUMEN DE EXPERIMENTOS Y MODELOS")
    print("=" * 70)

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

            model_dir = os.path.dirname(mf)
            best_model_exists = os.path.exists(os.path.join(model_dir, "best_model", "best_model.zip"))

            records.append({
                "Experimento": exp.get("name", "N/A"),
                "Autor": exp.get("author", "N/A"),
                "Recompensa": env.get("reward_type", "N/A"),
                "Max Speed": env.get("max_speed", "N/A"),
                "Learning Rate": ppo.get("learning_rate", "N/A"),
                "n_steps": ppo.get("n_steps", "N/A"),
                "Timesteps": train.get("total_timesteps", "N/A"),
                "Fecha": data.get("timestamp", "N/A"),
                "Best Model Listo": "Sí" if best_model_exists else "No"
            })
        except Exception as e:
            print(f"Error leyendo {mf}: {e}")

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    # Guardar en Markdown dentro de una carpeta dedicada 'reports/'
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    out_md = os.path.join(reports_dir, "EXPERIMENTS_SUMMARY.md")
    
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# 📊 Resumen Comparativo de Experimentos\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*Actualizado automáticamente con `python compare_experiments.py`*\n")

    print(f"\n>>> [OK] Resumen actualizado en: {out_md}")


if __name__ == "__main__":
    generate_experiment_summary()
