import sys
from pathlib import Path
REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))
import yaml
import pandas as pd
from stable_baselines3 import PPO
from src.evaluation.evaluator import evaluate_on_dataset
from src.evaluation.datasets import save_and_get_datasets

val_set, _ = save_and_get_datasets()

records = []
for exp in ['X5_near', 'X3']:
    with open(REPO_ROOT / f'configs/phase_a/{exp}.yaml') as f:
        cfg = yaml.safe_load(f)
    for s in [42, 100, 2026]:
        norm_final = str(REPO_ROOT / f'experiments/phase_a/{exp}/seed_{s}/models/vecnormalize_stats.pkl')
        
        # 1. best_model.zip
        m_best_path = str(REPO_ROOT / f'experiments/phase_a/{exp}/seed_{s}/models/best_model/best_model.zip')
        m_best = PPO.load(m_best_path)
        df_best, met_best = evaluate_on_dataset(m_best, val_set, 'val', 'deterministic', env_config=cfg, vec_normalize_path=norm_final)
        
        # 2. ppo_arm2d_final.zip
        m_final_path = str(REPO_ROOT / f'experiments/phase_a/{exp}/seed_{s}/models/ppo_arm2d_final.zip')
        m_final = PPO.load(m_final_path)
        df_fin, met_fin = evaluate_on_dataset(m_final, val_set, 'val', 'deterministic', env_config=cfg, vec_normalize_path=norm_final)
        
        succ_b = met_best['success_rate'] * 100.0
        succ_f = met_fin['success_rate'] * 100.0
        print(f"{exp} seed {s}:")
        print(f"  best_model:  succ={succ_b:.1f}%, steps={met_best['median_steps_to_success']}, delta_a={met_best['mean_action_delta']:.4f}")
        print(f"  final_model: succ={succ_f:.1f}%, steps={met_fin['median_steps_to_success']}, delta_a={met_fin['mean_action_delta']:.4f}")
        
        records.append({'exp': exp, 'seed': s, 'model_variant': 'best_model', 'success_rate': met_best['success_rate'], 'median_steps': met_best['median_steps_to_success'], 'action_delta': met_best['mean_action_delta']})
        records.append({'exp': exp, 'seed': s, 'model_variant': 'final_model', 'success_rate': met_fin['success_rate'], 'median_steps': met_fin['median_steps_to_success'], 'action_delta': met_fin['mean_action_delta']})

out_df = pd.DataFrame(records)
out_dir = REPO_ROOT / 'reports/data/phase_b'
out_dir.mkdir(parents=True, exist_ok=True)
out_df.to_csv(out_dir / 'phase_a_models_val_comparison.csv', index=False)
print('\n[OK] Guardado en reports/data/phase_b/phase_a_models_val_comparison.csv')
