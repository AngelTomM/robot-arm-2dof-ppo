"""
Paso 5: Renderizado Interactivo en Tiempo Real para Brazo Robótico 2 DOF.
Permite ver al agente de RL en acción y controlar el objetivo de forma interactiva con el mouse o con la barra espaciadora.
Soporta la carga automática de modelos PPO o SAC según los metadatos o formato del checkpoint.
"""

import os
import sys
import json
import argparse
import math
import numpy as np
import pygame
import yaml
from stable_baselines3 import PPO, SAC
from arm2d_env import Arm2DEnv


def find_model_config(model_path, config_path=None):
    """Carga la configuración asociada al modelo o usa valores por defecto."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # Intentar buscar experiment_meta.json en la carpeta del modelo
    if model_path:
        model_dir = os.path.dirname(os.path.abspath(model_path))
        candidates = [
            os.path.join(model_dir, "experiment_meta.json"),
            os.path.join(os.path.dirname(model_dir), "experiment_meta.json")
        ]
        for meta_file in candidates:
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        print(f">>> [CONFIG] Ajustando entorno a los parámetros del modelo ({meta_file})...")
                        return meta.get("parameters", {})
                except Exception as e:
                    print(f">>> [AVISO] No se pudo leer {meta_file}: {e}")

    return None


def load_trained_model(model_path, env, algo_hint=None):
    """Carga el modelo intentando PPO o SAC según corresponda."""
    if not model_path or not os.path.exists(model_path):
        return None

    algo = (algo_hint or "").upper()

    if algo == "SAC":
        try:
            print(f">>> Cargando modelo SAC desde: {model_path}")
            return SAC.load(model_path, env=env)
        except Exception:
            pass

    if algo == "PPO":
        try:
            print(f">>> Cargando modelo PPO desde: {model_path}")
            return PPO.load(model_path, env=env)
        except Exception:
            pass

    # Si no se especificó o falló, intentar primero con PPO y fallback a SAC
    try:
        print(f">>> Intentando cargar modelo como PPO: {model_path}")
        return PPO.load(model_path, env=env)
    except Exception:
        print(f">>> Cargando modelo como SAC: {model_path}")
        return SAC.load(model_path, env=env)


def interactive_render(model_path=None, config_path=None):
    print("=" * 60)
    print(" SIMULADOR INTERACTIVO DE BRAZO ROBÓTICO 2D (PPO / SAC)")
    print("=" * 60)
    print("Controles:")
    print(" - Click Izquierdo en la ventana : Mover el objetivo (Target) a esa posición")
    print(" - Barra Espaciadora            : Generar nuevo objetivo aleatorio")
    print(" - Tecla 'R'                     : Reiniciar posición del brazo")
    print(" - Tecla 'ESC' o cerrar ventana  : Salir")
    print("=" * 60)

    # Resolver ruta de modelo por defecto si no se proporcionó
    if not model_path:
        default_paths = [
            "models/exp_sac_baseline/best_model/best_model.zip",
            "models/exp2_smooth_precision/best_model/best_model.zip",
            "models/exp1_baseline/best_model/best_model.zip"
        ]
        for p in default_paths:
            if os.path.exists(p):
                model_path = p
                break

    # Cargar configuración y resolver algoritmo
    cfg = find_model_config(model_path, config_path) or {}
    algo_name = cfg.get("algorithm", None)

    # Crear entorno
    env = Arm2DEnv(render_mode="human", config=cfg)
    obs, info = env.reset()

    # Cargar modelo PPO o SAC
    model = load_trained_model(model_path, env, algo_name)

    if model is None:
        print(">>> [AVISO] No se encontró modelo válido. Ejecutando con acciones aleatorias.")

    running = True
    manual_target = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    manual_target = False
                    obs, _ = env.reset()
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = pygame.mouse.get_pos()
                    world_x = (mx - env.origin[0]) / env.scale
                    world_y = -(my - env.origin[1]) / env.scale
                    env.set_target(world_x, world_y)
                    manual_target = True
                    obs = env._get_obs()

        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        if (terminated or truncated) and not manual_target:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulador interactivo para Brazo 2D (PPO / SAC)")
    parser.add_argument("model_path", nargs="?", default=None, help="Ruta al modelo .zip")
    parser.add_argument("--config", "-c", type=str, default=None, help="Ruta opcional al YAML de configuración")
    args = parser.parse_args()

    interactive_render(args.model_path, args.config)
