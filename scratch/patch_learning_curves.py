import re

plot_path = r"c:\Users\ulica\Documents\AprendizajeAuo2026\Fase_D_V3\plot.py"

with open(plot_path, "r", encoding="utf-8") as f:
    content = f.read()

# Normalize CRLF to LF for matching
content = content.replace("\r\n", "\n")

target_block = """def plot_model_learning_curves(eval_data_list: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
                               model_name: str, model_fig_dir: Path) -> None:
    \"\"\"
    Genera gráficas de aprendizaje (retorno y tasa de éxito) ajustadas automáticamente a una o varias semillas.
    Cada elemento en eval_data_list es (seed, timesteps, results, successes).
    \"\"\"
    apply_plot_style()
    if not eval_data_list:
        return

    # 1. Gráfica de Retorno Episódico
    fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=300)
    all_means = []
    common_t = eval_data_list[0][1]

    for seed, t, res, _ in eval_data_list:
        m = np.mean(res, axis=1)
        all_means.append(m)
        if len(eval_data_list) > 1:
            ax.plot(t, m, alpha=0.45, linewidth=1.2, label=f"Semilla {seed}")

    all_means_arr = np.array(all_means)
    grand_mean = np.mean(all_means_arr, axis=0)
    grand_std = np.std(all_means_arr, axis=0) if len(eval_data_list) > 1 else np.std(eval_data_list[0][2], axis=1)

    label_mean = f"Retorno Medio ($\mu$) ({len(eval_data_list)} semillas)" if len(eval_data_list) > 1 else "Retorno Medio"
    ax.plot(common_t, grand_mean, color="#27ae60", linewidth=2.2, label=label_mean)
    ax.fill_between(common_t, grand_mean - grand_std, grand_mean + grand_std, color="#27ae60", alpha=0.22,
                    label=r"Dispersión $\pm 1\sigma$")

    ax.set_xlabel("Pasos de Interacción (Timesteps)")
    ax.set_ylabel("Retorno Acumulado LQR")
    ax.set_title(f"Curva de Aprendizaje: Retorno Episódico — {model_name}", pad=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    p_ret = model_fig_dir / "learning_curve_return.png"
    fig.savefig(p_ret)
    plt.close(fig)
    print(f">>> [FIGURA] Guardada: {p_ret.name}")

    # 2. Gráfica de Tasa de Éxito
    fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=300)
    all_succ = []
    has_successes = any(s[3] is not None for s in eval_data_list)

    if has_successes:
        for seed, t, _, succ in eval_data_list:
            if succ is not None:
                sr = np.mean(succ, axis=1) * 100.0
                all_succ.append(sr)
                if len(eval_data_list) > 1:
                    ax.plot(t, sr, alpha=0.45, linewidth=1.2, label=f"Semilla {seed}")

        if all_succ:
            all_succ_arr = np.array(all_succ)
            mean_succ = np.mean(all_succ_arr, axis=0)
            std_succ = np.std(all_succ_arr, axis=0)
            label_s = f"Tasa de Estabilización ($\mu$) ({len(eval_data_list)} semillas)" if len(eval_data_list) > 1 else "Tasa de Estabilización"
            ax.plot(common_t, mean_succ, color="#2b5c8f", marker="o", markersize=3.5, linewidth=2.0, label=label_s)
            if len(eval_data_list) > 1:
                ax.fill_between(common_t, np.maximum(0, mean_succ - std_succ), np.minimum(100, mean_succ + std_succ),
                                color="#2b5c8f", alpha=0.20, label=r"Dispersión $\pm 1\sigma$")

        ax.set_ylim(-2, 105)
        ax.set_ylabel("Tasa de Estabilización (%)")
        ax.set_xlabel("Pasos de Interacción (Timesteps)")
        ax.set_title(f"Curva de Aprendizaje: Tasa de Estabilización — {model_name}", pad=10)
        ax.legend(loc="upper left")
        fig.tight_layout()
        p_succ = model_fig_dir / "learning_curve_success_rate.png"
        fig.savefig(p_succ)
        plt.close(fig)
        print(f">>> [FIGURA] Guardada: {p_succ.name}")

    # 3. Multisemilla consolidada (2 paneles)
    if len(eval_data_list) > 1 and has_successes and all_succ:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.5), dpi=300, sharex=True)
        ax1.plot(common_t, grand_mean, color="#27ae60", linewidth=2.0, label=rf"Media $\mu$ ({len(eval_data_list)} semillas)")
        ax1.fill_between(common_t, grand_mean - grand_std, grand_mean + grand_std, color="#27ae60", alpha=0.22)
        ax1.set_ylabel("Retorno Acumulado LQR")
        ax1.set_title(f"Convergencia Multisemilla con Bandas Estadísticas — {model_name}", pad=8)
        ax1.legend(loc="lower right")

        ax2.plot(common_t, mean_succ, color="#2b5c8f", linewidth=2.0, label=r"Tasa de Éxito $\mu$")
        ax2.fill_between(common_t, np.maximum(0, mean_succ - std_succ), np.minimum(100, mean_succ + std_succ),
                         color="#2b5c8f", alpha=0.20)
        ax2.set_ylabel("Tasa Éxito (%)")
        ax2.set_xlabel("Pasos de Interacción (Timesteps)")
        ax2.set_ylim(-2, 105)
        ax2.legend(loc="upper left")

        fig.tight_layout()
        p_multi = model_fig_dir / "multiseed_learning_curves.png"
        fig.savefig(p_multi)
        plt.close(fig)
        print(f">>> [FIGURA] Guardada: {p_multi.name}")"""

replacement_block = """def plot_model_learning_curves(eval_data_list: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
                               model_name: str, model_fig_dir: Path,
                               requested_metrics: Optional[List[str]] = None) -> None:
    \"\"\"
    Genera gráficas de aprendizaje (retorno y tasa de éxito) ajustadas automáticamente a una o varias semillas.
    Cada elemento en eval_data_list es (seed, timesteps, results, successes).
    \"\"\"
    apply_plot_style()
    if not eval_data_list:
        return

    # Pre-cómputo de retorno
    all_means = []
    common_t = eval_data_list[0][1]
    for seed, t, res, _ in eval_data_list:
        m = np.mean(res, axis=1)
        all_means.append(m)

    all_means_arr = np.array(all_means)
    grand_mean = np.mean(all_means_arr, axis=0)
    grand_std = np.std(all_means_arr, axis=0) if len(eval_data_list) > 1 else np.std(eval_data_list[0][2], axis=1)

    # Pre-cómputo de éxito
    all_succ = []
    has_successes = any(s[3] is not None for s in eval_data_list)
    if has_successes:
        for seed, t, _, succ in eval_data_list:
            if succ is not None:
                sr = np.mean(succ, axis=1) * 100.0
                all_succ.append(sr)

    if all_succ:
        all_succ_arr = np.array(all_succ)
        mean_succ = np.mean(all_succ_arr, axis=0)
        std_succ = np.std(all_succ_arr, axis=0)
    else:
        mean_succ, std_succ = None, None

    # 1. Gráfica de Retorno Episódico
    if requested_metrics is None or "learning_return" in requested_metrics or "learning_curves" in requested_metrics:
        fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=300)
        for seed, t, res, _ in eval_data_list:
            m = np.mean(res, axis=1)
            if len(eval_data_list) > 1:
                ax.plot(t, m, alpha=0.45, linewidth=1.2, label=f"Semilla {seed}")

        label_mean = f"Retorno Medio ($\mu$) ({len(eval_data_list)} semillas)" if len(eval_data_list) > 1 else "Retorno Medio"
        ax.plot(common_t, grand_mean, color="#27ae60", linewidth=2.2, label=label_mean)
        ax.fill_between(common_t, grand_mean - grand_std, grand_mean + grand_std, color="#27ae60", alpha=0.22,
                        label=r"Dispersión $\pm 1\sigma$")

        ax.set_xlabel("Pasos de Interacción (Timesteps)")
        ax.set_ylabel("Retorno Acumulado LQR")
        ax.set_title(f"Curva de Aprendizaje: Retorno Episódico — {model_name}", pad=10)
        ax.legend(loc="lower right")
        fig.tight_layout()
        p_ret = model_fig_dir / "learning_curve_return.png"
        fig.savefig(p_ret)
        plt.close(fig)
        print(f">>> [FIGURA] Guardada: {p_ret.name}")

    # 2. Gráfica de Tasa de Éxito
    if (requested_metrics is None or "learning_success_rate" in requested_metrics or "learning_curves" in requested_metrics) and has_successes and all_succ:
        fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=300)
        for seed, t, _, succ in eval_data_list:
            if succ is not None:
                sr = np.mean(succ, axis=1) * 100.0
                if len(eval_data_list) > 1:
                    ax.plot(t, sr, alpha=0.45, linewidth=1.2, label=f"Semilla {seed}")

        label_s = f"Tasa de Estabilización ($\mu$) ({len(eval_data_list)} semillas)" if len(eval_data_list) > 1 else "Tasa de Estabilización"
        ax.plot(common_t, mean_succ, color="#2b5c8f", marker="o", markersize=3.5, linewidth=2.0, label=label_s)
        if len(eval_data_list) > 1:
            ax.fill_between(common_t, np.maximum(0, mean_succ - std_succ), np.minimum(100, mean_succ + std_succ),
                            color="#2b5c8f", alpha=0.20, label=r"Dispersión $\pm 1\sigma$")

        ax.set_ylim(-2, 105)
        ax.set_ylabel("Tasa de Estabilización (%)")
        ax.set_xlabel("Pasos de Interacción (Timesteps)")
        ax.set_title(f"Curva de Aprendizaje: Tasa de Estabilización — {model_name}", pad=10)
        ax.legend(loc="upper left")
        fig.tight_layout()
        p_succ = model_fig_dir / "learning_curve_success_rate.png"
        fig.savefig(p_succ)
        plt.close(fig)
        print(f">>> [FIGURA] Guardada: {p_succ.name}")

    # 3. Multisemilla consolidada (2 paneles)
    if (requested_metrics is None or "learning_curves" in requested_metrics or ("learning_return" in requested_metrics and "learning_success_rate" in requested_metrics)) and len(eval_data_list) > 1 and has_successes and all_succ:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.5), dpi=300, sharex=True)
        ax1.plot(common_t, grand_mean, color="#27ae60", linewidth=2.0, label=rf"Media $\mu$ ({len(eval_data_list)} semillas)")
        ax1.fill_between(common_t, grand_mean - grand_std, grand_mean + grand_std, color="#27ae60", alpha=0.22)
        ax1.set_ylabel("Retorno Acumulado LQR")
        ax1.set_title(f"Convergencia Multisemilla con Bandas Estadísticas — {model_name}", pad=8)
        ax1.legend(loc="lower right")

        ax2.plot(common_t, mean_succ, color="#2b5c8f", linewidth=2.0, label=r"Tasa de Éxito $\mu$")
        ax2.fill_between(common_t, np.maximum(0, mean_succ - std_succ), np.minimum(100, mean_succ + std_succ),
                         color="#2b5c8f", alpha=0.20)
        ax2.set_ylabel("Tasa Éxito (%)")
        ax2.set_xlabel("Pasos de Interacción (Timesteps)")
        ax2.set_ylim(-2, 105)
        ax2.legend(loc="upper left")

        fig.tight_layout()
        p_multi = model_fig_dir / "multiseed_learning_curves.png"
        fig.savefig(p_multi)
        plt.close(fig)
        print(f">>> [FIGURA] Guardada: {p_multi.name}")"""

if target_block in content:
    content = content.replace(target_block, replacement_block)
    with open(plot_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: plot_model_learning_curves patched successfully.")
else:
    print("ERROR: target_block not found in plot.py")
