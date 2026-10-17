import pandas as pd

df = pd.read_csv('reports/phase_c_s1_analysis/metricas_checkpoints.csv')
steps = sorted(df['checkpoint_step'].unique())

lines = []
lines.append("| Checkpoint | Pasos Entorno | Éxito Estricto (%) | Truncamiento (%) | Nunca Entra (%) | Entra sin Hold (%) | Variación $\\|\\Delta \\mathbf{a}\\|$ |")
lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

for st in steps:
    sub = df[df['checkpoint_step'] == st]
    succ_m, succ_s = sub['success_rate'].mean() * 100, sub['success_rate'].std(ddof=1) * 100
    trunc_m, trunc_s = sub['truncated_rate'].mean() * 100, sub['truncated_rate'].std(ddof=1) * 100
    ne_m, ne_s = sub['never_entered_rate'].mean() * 100, sub['never_entered_rate'].std(ddof=1) * 100
    enh_m, enh_s = sub['entered_no_hold_rate'].mean() * 100, sub['entered_no_hold_rate'].std(ddof=1) * 100
    act_m, act_s = sub['action_delta_mean'].mean(), sub['action_delta_mean'].std(ddof=1)
    
    st_fmt = f"{st:_}".replace('_', r'\,')
    
    s_succ = f"{succ_m:.2f} \\pm {succ_s:.2f}".replace('.', '{,}')
    s_trunc = f"{trunc_m:.2f} \\pm {trunc_s:.2f}".replace('.', '{,}')
    s_ne = f"{ne_m:.2f} \\pm {ne_s:.2f}".replace('.', '{,}')
    s_enh = f"{enh_m:.2f} \\pm {enh_s:.2f}".replace('.', '{,}')
    s_act = f"{act_m:.4f} \\pm {act_s:.4f}".replace('.', '{,}')
    
    if st == 327680:
        lines.append(f"| **`ckpt_{st}`** | **${st_fmt}$** | **${s_succ}$** | **${s_trunc}$** | **${s_ne}$** | **${s_enh}$** | **${s_act}$** |")
    elif st == 655360:
        lines.append(f"| **`ckpt_{st}`** | **${st_fmt}$** | **${s_succ}$** | **${s_trunc}$** | **${s_ne}$** | **${s_enh}$** | **${s_act}$** |")
    elif st == 286720:
        lines.append(f"| `ckpt_{st}` | ${st_fmt}$ | ${s_succ}$ | ${s_trunc}$ | ${s_ne}$ | ${s_enh}$ | ${s_act}$ |")
    else:
        lines.append(f"| `ckpt_{st}` | ${st_fmt}$ | ${s_succ}$ | ${s_trunc}$ | ${s_ne}$ | ${s_enh}$ | ${s_act}$ |")

output_str = "\n".join(lines)
print(output_str)

with open('reports/phase_c_s1_analysis/tabla_4_1_corregida.md', 'w', encoding='utf-8') as f:
    f.write(output_str)
