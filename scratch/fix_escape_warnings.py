plot_path = r"c:\Users\ulica\Documents\AprendizajeAuo2026\Fase_D_V3\plot.py"

with open(plot_path, "r", encoding="utf-8") as f:
    c = f.read()

c = c.replace(
    'ax1.plot(t, dist_mm, color="#2980b9", linewidth=2.0, label="Distancia cartesiana $\\|e(t)\\|$")',
    'ax1.plot(t, dist_mm, color="#2980b9", linewidth=2.0, label=r"Distancia cartesiana $\\|e(t)\\|$")'
)

c = c.replace(
    'label_mean = f"Retorno Medio ($\\mu$) ({len(eval_data_list)} semillas)" if len(eval_data_list) > 1 else "Retorno Medio"',
    'label_mean = rf"Retorno Medio ($\\mu$) ({len(eval_data_list)} semillas)" if len(eval_data_list) > 1 else "Retorno Medio"'
)

c = c.replace(
    'label_s = f"Tasa de Estabilización ($\\mu$) ({len(eval_data_list)} semillas)" if len(eval_data_list) > 1 else "Tasa de Estabilización"',
    'label_s = rf"Tasa de Estabilización ($\\mu$) ({len(eval_data_list)} semillas)" if len(eval_data_list) > 1 else "Tasa de Estabilización"'
)

c = c.replace(
    'ax2.set_ylabel("Magnitud Torque $\\|\\\\tau\\|$ [N·m]")',
    'ax2.set_ylabel(r"Magnitud Torque $\\|\\tau\\|$ [N·m]")'
)

with open(plot_path, "w", encoding="utf-8") as f:
    f.write(c)

print("Warning fixes applied.")
