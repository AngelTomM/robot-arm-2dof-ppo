"""
Script para generar gráficos automáticos y análisis del entrenamiento.
Extrae los datos de los logs de TensorBoard / Stable-Baselines3 y los guarda como imagen.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def plot_training_results(log_dir="logs"):
    # Buscar carpetas de logs generadas por Stable-Baselines3
    event_files = glob.glob(f"{log_dir}/**/events.out.tfevents.*", recursive=True)

    if not event_files:
        print(f">>> [AVISO] No se encontraron archivos de logs en '{log_dir}'.")
        print(">>> Asegúrate de haber ejecutado al menos un entrenamiento con 'train.py'.")
        return

    print(f">>> Analizando {len(event_files)} archivo(s) de logs...")

    # Cargar el último archivo de eventos
    latest_event_file = max(event_files, key=os.path.getmtime)
    print(f">>> Leyendo log más reciente: {latest_event_file}")

    ea = event_accumulator.EventAccumulator(latest_event_file)
    ea.Reload()

    # Métricas disponibles en TensorBoard
    available_tags = ea.Tags()["scalars"]
    print(">>> Métricas registradas:", available_tags)

    plt.figure(figsize=(14, 8))
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    # 1. Gráfica de Recompensa Media por Episodio (Episodic Mean Reward)
    target_tags = {
        "rollout/ep_rew_mean": ("Recompensa Media por Episodio", "Episodic Mean Reward", 1),
        "rollout/ep_len_mean": ("Longitud Media del Episodio", "Episodic Mean Length (steps)", 2),
        "train/loss": ("Pérdida Total (Total Loss)", "Loss", 3),
        "train/entropy_loss": ("Entropía de la Política (Exploración)", "Entropy Loss", 4)
    }

    plot_idx = 1
    for tag, (title, ylabel, pos) in target_tags.items():
        if tag in available_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            plt.subplot(2, 2, plot_idx)
            plt.plot(steps, values, label=tag.split('/')[-1], color="#2980b9", linewidth=2)
            plt.title(title, fontsize=12, fontweight='bold')
            plt.xlabel("Timesteps")
            plt.ylabel(ylabel)
            plt.grid(True, linestyle="--", alpha=0.6)
            plot_idx += 1

    plt.tight_layout()
    os.makedirs("analysis", exist_ok=True)
    out_path = "analysis/training_curves.png"
    plt.savefig(out_path, dpi=300)
    print(f">>> [OK] Gráficas de entrenamiento guardadas con éxito en: {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_training_results()
