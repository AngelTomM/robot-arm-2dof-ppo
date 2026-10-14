"""
Prueba de Humo Integral para la Fase C (Etapa C-3):
Ejecuta un ciclo completo y ultrarrápido:
1. Entrenamiento SAC de 1.000 pasos con guardado atómico en paso 500 y 1.000.
2. Verificación de existencia e integridad de model.zip y vecnormalize_stats.pkl.
3. Evaluación del checkpoint sobre una muestra de 5 episodios de Validación C.
4. Extracción de telemetría por paso y verificación de métricas.
"""
import os
import sys
import tempfile
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_c_train_sac import train_phase_c_sac_agent
from scripts.phase_c_eval_checkpoints import evaluate_single_checkpoint
from scripts.phase_c_step_data import collect_step_telemetry
from src.controllers.jacobian_controller import JacobianDampedController

def run_smoke_test():
    print(f"\n{'='*75}")
    print(" INICIANDO PRUEBA DE HUMO INTEGRAL — FASE C")
    print(f"{'='*75}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        print(f">>> [1/4] Directorio temporal de prueba: {tmp_path}")

        # Configuración mínima de humo basada en S1
        smoke_cfg = {
            "experiment": {"name": "S1_smoke", "phase": "C"},
            "env": {
                "l1": 1.0, "l2": 1.0, "max_speed": 1.5, "target_radius": 0.06,
                "max_steps": 50, "dt": 0.05, "success_mode": "hold", "hold_steps": 5,
                "velocity_tolerance": 0.30, "reward_type": "smooth_precision",
                "reward_params": {
                    "distance_penalty_coef": 1.0, "action_penalty_coef": 0.02,
                    "success_bonus": 15.0, "near_zone_threshold": 0.30, "velocity_penalty_near": 0.15
                }
            },
            "normalization": {"norm_obs": True, "norm_reward": False, "clip_obs": 10.0},
            "sac": {
                "learning_rate": 3e-4, "buffer_size": 2000, "learning_starts": 100,
                "batch_size": 32, "tau": 0.005, "gamma": 0.99, "train_freq": 1,
                "gradient_steps": 1, "ent_coef": "auto", "net_arch": "256x256_relu", "num_envs": 1
            },
            "training": {"total_timesteps": 600, "checkpoint_freq": 300}
        }

        # 1. Ejecutar entrenamiento ultracorto (600 timesteps, checkpoints en 300 y 600)
        print(">>> [2/4] Ejecutando entrenamiento corto de SAC (600 pasos)...")
        metrics = train_phase_c_sac_agent(
            config_path_or_dict=smoke_cfg,
            seed=999,
            total_timesteps_override=600,
            checkpoint_freq_override=300,
            output_base_dir=str(tmp_path)
        )

        ckpt_dir_300 = tmp_path / "S1_smoke" / "seed_999" / "checkpoints" / "ckpt_300"
        ckpt_dir_600 = tmp_path / "S1_smoke" / "seed_999" / "checkpoints" / "ckpt_600"

        assert ckpt_dir_300.exists(), "ckpt_300 debe existir"
        assert (ckpt_dir_300 / "model.zip").exists(), "model.zip debe existir en ckpt_300"
        assert (ckpt_dir_300 / "vecnormalize_stats.pkl").exists(), "vecnormalize_stats.pkl debe existir en ckpt_300"

        assert ckpt_dir_600.exists(), "ckpt_600 debe existir"
        assert (ckpt_dir_600 / "model.zip").exists(), "model.zip debe existir en ckpt_600"
        assert (ckpt_dir_600 / "vecnormalize_stats.pkl").exists(), "vecnormalize_stats.pkl debe existir en ckpt_600"
        print(">>> [OK] Checkpoints generados con pares atómicos completos.")

        # 2. Evaluar checkpoint sobre 5 episodios de Validación C
        print(">>> [3/4] Evaluando checkpoint sobre muestra de 5 episodios de Validación C...")
        val_c_path = REPO_ROOT / "data" / "datasets" / "phase_c" / "validation_c_500.json"
        with open(val_c_path, "r", encoding="utf-8") as f:
            val_dataset = json.load(f)[:5]

        env_cfg = {
            "env": {
                "max_speed": 1.5, "success_mode": "hold", "hold_steps": 20,
                "velocity_tolerance": 0.10, "target_radius": 0.06
            }
        }

        row, ep_df = evaluate_single_checkpoint(
            ckpt_dir=ckpt_dir_600,
            val_dataset=val_dataset,
            env_cfg=env_cfg,
            eval_mode="deterministic",
            algo_hint="SAC"
        )
        assert len(ep_df) == 5, f"Se esperaban 5 episodios evaluados, se obtuvieron {len(ep_df)}"
        assert "success" in ep_df.columns, "Columna success debe existir"
        assert "steps" in ep_df.columns, "Columna steps debe existir"
        print(f">>> [OK] Evaluación determinista exitosa: tasa de éxito {row['success_rate']*100:.1f}%.")

        # 3. Extraer telemetría por paso
        print(">>> [4/4] Probando extracción de telemetría por paso...")
        c0 = JacobianDampedController(kp=3.0, lambda_damping=0.01, max_speed=1.5)
        df_step = collect_step_telemetry(c0, val_dataset[:2], env_cfg, eval_mode="deterministic")
        assert len(df_step) > 0, "Telemetría no debe ser vacía"
        assert "distance" in df_step.columns and "joint_speed" in df_step.columns
        print(f">>> [OK] Telemetría extraída correctamente: {len(df_step)} pasos registrados.")

    print(f"\n{'='*75}")
    print(" >>> PRUEBA DE HUMO DE FASE C SUPERADA CON ÉXITO ABSOLUTO <<<")
    print(f"{'='*75}\n")

if __name__ == "__main__":
    run_smoke_test()
