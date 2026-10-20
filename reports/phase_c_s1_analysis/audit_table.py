import pandas as pd

with open('reports/phase_c_s1_analysis/diagnostico_s1.md', 'r', encoding='utf-8') as f:
    text = f.read()

table_lines = [l for l in text.split('\n') if l.strip().startswith('| `ckpt_') or l.strip().startswith('| **`ckpt_')]
df_csv = pd.read_csv('reports/phase_c_s1_analysis/metricas_checkpoints.csv')

parsed_rows = []
for line in table_lines:
    parts = [p.strip().replace('**', '').replace('$', '').replace('\\,', '').replace('{', '').replace('}', '') for p in line.split('|')[1:-1]]
    ckpt = parts[0].strip('`')
    step = int(parts[1])
    succ = parts[2].replace(',', '.')
    trunc = parts[3].replace(',', '.')
    ne = parts[4].replace(',', '.')
    enh = parts[5].replace(',', '.')
    delta_a = parts[6].replace(',', '.')
    parsed_rows.append({
        'ckpt': ckpt, 'step': step, 'succ': succ, 'trunc': trunc, 'ne': ne, 'enh': enh, 'delta_a': delta_a
    })

print(f"{'Step':>7} | {'Col':>7} | {'Report (diagnostico_s1.md)':>25} | {'Recalculated (ddof=1)':>25} | {'Match?'}")
print("-" * 80)

discrepancies = []

for r in parsed_rows:
    st = r['step']
    sub = df_csv[df_csv['checkpoint_step'] == st]
    succ_m, succ_s = sub['success_rate'].mean() * 100, sub['success_rate'].std(ddof=1) * 100
    trunc_m, trunc_s = sub['truncated_rate'].mean() * 100, sub['truncated_rate'].std(ddof=1) * 100
    ne_m, ne_s = sub['never_entered_rate'].mean() * 100, sub['never_entered_rate'].std(ddof=1) * 100
    enh_m, enh_s = sub['entered_no_hold_rate'].mean() * 100, sub['entered_no_hold_rate'].std(ddof=1) * 100
    act_m, act_s = sub['action_delta_mean'].mean(), sub['action_delta_mean'].std(ddof=1)

    checks = [
        ('Exito', r['succ'], f"{succ_m:.2f} +- {succ_s:.2f}"),
        ('Trunc', r['trunc'], f"{trunc_m:.2f} +- {trunc_s:.2f}"),
        ('NE', r['ne'], f"{ne_m:.2f} +- {ne_s:.2f}"),
        ('ENH', r['enh'], f"{enh_m:.2f} +- {enh_s:.2f}"),
        ('DeltaA', r['delta_a'], f"{act_m:.4f} +- {act_s:.4f}")
    ]

    for col_name, rep_val, calc_val in checks:
        # compare rep_val vs calc_val
        val_clean = rep_val.replace('\\pm', '+-')
        rep_m, rep_s = [float(x.strip()) for x in val_clean.split('+-')]
        calc_m, calc_s = [float(x.strip()) for x in calc_val.split('+-')]
        
        diff_m = abs(rep_m - calc_m)
        diff_s = abs(rep_s - calc_s)
        
        matched = (diff_m < 0.05 and diff_s < 0.05)
        if not matched:
            discrepancies.append((st, col_name, rep_val, calc_val, diff_m, diff_s))
            print(f"{st:7d} | {col_name:>7} | {rep_val:>25} | {calc_val:>25} | DIFF (dm={diff_m:.2f}, ds={diff_s:.2f})")

print(f"\nTotal discrepancies found: {len(discrepancies)}")
