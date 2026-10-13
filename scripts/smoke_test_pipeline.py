"""
Prueba de Humo (Smoke Test) del Pipeline de Entrenamiento de Fase A (Etapa E-4).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import glob
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from arm2d_env import Arm2DEnv
from src.training.train_phase_a import train_agent
from src.evaluation.datasets import save_and_get_datasets
from src.evaluation.evaluator import evaluate_on_dataset

def run_smoke_test():
    print("=" * 80)
    print(" INICIANDO PRUEBA DE HUMO DEL PIPELINE DE ENTRENAMIENTO (ETAPA E-4)")
    print("=" * 80)

    test_cfg = {
        "experiment": {
            "name": "smoke_test",
            "description": "Prueba de humo del pipeline completo"
        },
        "env": {
            "l1": 1.0,
            "l2": 1.0,
            "max_speed": 2.0,
            "target_radius": 0.06,
            "max_steps": 50,
            "dt": 0.05,
            "success_mode": "hold",
            "hold_steps": 5,
            "velocity_tolerance": 0.3,
            "reward_type": "smooth_precision",
            "reward_params": {
                "distance_penalty_coef": 1.0,
                "action_penalty_coef": 0.01,
                "success_bonus": 10.0,
                "near_zone_threshold": 0.3,
                "velocity_penalty_near": 0.1
            }
        },
        "normalization": {
            "norm_obs": True,
            "norm_reward": False,
            "clip_obs": 10.0
        },
        "ppo": {
            "learning_rate": 3e-4,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.001,
            "num_envs": 2,
            "net_arch": "64x64_tanh"
        },
        "training": {
            "total_timesteps": 2048,
            "eval_freq": 1024,
            "n_eval_episodes": 2,
            "save_freq": 1024
        }
    }

    smoke_base_dir = "experiments/phase_a/smoke_test_run"
    if os.path.exists(smoke_base_dir):
        shutil.rmtree(smoke_base_dir)

    # 1. Ejecutar Entrenamiento
    print("\n>>> [1/4] Ejecutando entrenamiento PPO con 2048 timesteps...")
    model, exp_dir = train_agent(test_cfg, seed=123, output_base_dir=smoke_base_dir)
    print(f">>> Directorio de salida: {exp_dir}")

    # 2. Verificar existencia de archivos clave
    models_dir = os.path.join(exp_dir, "models")
    logs_dir = os.path.join(exp_dir, "logs")

    final_model_zip = os.path.join(models_dir, "ppo_arm2d_final.zip")
    norm_stats_pkl = os.path.join(models_dir, "vecnormalize_stats.pkl")
    meta_json = os.path.join(exp_dir, "experiment_meta.json")

    assert os.path.exists(final_model_zip), f"Falta modelo final: {final_model_zip}"
    assert os.path.exists(norm_stats_pkl), f"Faltan estadísticas VecNormalize: {norm_stats_pkl}"
    assert os.path.exists(meta_json), f"Faltan metadatos: {meta_json}"

    tb_files = glob.glob(f"{logs_dir}/**/events.out.tfevents.*", recursive=True)
    assert len(tb_files) > 0, f"No se generaron eventos de TensorBoard en {logs_dir}"
    print(">>> [2/4] [OK] Archivos generados correctamente (modelo .zip, normalizador .pkl, logs TB y metadatos).")

    # 3. Cargar Modelo y Normalizador para Evaluación
    print("\n>>> [3/4] Probando carga del modelo y estadísticas de normalización...")
    eval_env = DummyVecEnv([lambda: Arm2DEnv(config=test_cfg)])
    eval_env = VecNormalize.load(norm_stats_pkl, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    loaded_model = PPO.load(final_model_zip, env=eval_env)
    print(">>> [OK] Modelo y VecNormalize cargados con éxito.")

    # 4. Evaluación sobre una muestra del conjunto de Validación
    print("\n>>> [4/4] Ejecutando evaluación con el modelo cargado...")
    val_set, _ = save_and_get_datasets()
    sample_val = val_set[:5]

    df_eval, metrics = evaluate_on_dataset(loaded_model, sample_val, dataset_name="validation_sample",
                                          env_config=test_cfg)
    eval_env.close()

    print(f">>> Éxito en muestra: {metrics['success_rate_ci']}")
    print(f">>> Pasos evaluados: {df_eval['steps'].tolist()}")

    print("\n" + "=" * 80)
    print(" >>> [OK] ¡PRUEBA DE HUMO (SMOKE TEST) COMPLETADA EXITOSAMENTE! <<<")
    print("=" * 80)

if __name__ == "__main__":
    run_smoke_test()
