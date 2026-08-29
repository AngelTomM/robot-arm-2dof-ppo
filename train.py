"""
Script de Entrenamiento con PPO (Proximal Policy Optimization) para el Brazo Robótico 2D.
Utiliza Stable-Baselines3 con vectorización de entornos y monitoreo de métricas.
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from arm2d_env import Arm2DEnv


def train():
    # 1. Directorios para guardar modelos y logs
    models_dir = "models/PPO"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(">>> Creando entornos vectorizados paralelos...")
    # Creamos 4 entornos en paralelo para acelerar el muestreo de trayectorias
    num_envs = 4
    vec_env = make_vec_env(lambda: Arm2DEnv(), n_envs=num_envs)

    # Entorno separado para evaluación
    eval_env = make_vec_env(lambda: Arm2DEnv(), n_envs=1)

    # 2. Configurar Callbacks
    # EvalCallback evalúa periódicamente la política y guarda el mejor modelo
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{models_dir}/best_model",
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    # CheckpointCallback guarda el progreso cada N pasos
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=f"{models_dir}/checkpoints",
        name_prefix="ppo_arm2d"
    )

    # 3. Configurar el Agente PPO
    print(">>> Inicializando agente PPO...")
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,  # Coeficiente de entropía para fomentar exploración
        verbose=1,
        tensorboard_log=log_dir
    )

    # 4. Iniciar Entrenamiento
    total_timesteps = 150_000
    print(f">>> Iniciando entrenamiento por {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # 5. Guardar Modelo Final
    final_model_path = f"{models_dir}/ppo_arm2d_final"
    model.save(final_model_path)
    print(f"\n>>> ¡Entrenamiento completado! Modelo final guardado en: {final_model_path}.zip")
    print(f">>> Mejor modelo guardado en: {models_dir}/best_model/best_model.zip")


if __name__ == "__main__":
    train()
