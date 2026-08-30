"""
Paso 5: Renderizado Interactivo en Tiempo Real para Brazo Robótico 2 DOF.
Permite ver al agente de RL en acción y controlar el objetivo de forma interactiva con el mouse o con la barra espaciadora.
Soporta la carga automática de la configuración con la que se entrenó el modelo (experiment_meta.json o --config).
"""

import os
import sys
import json
import argparse
import math
import numpy as np
import pygame
import yaml
from stable_baselines3 import PPO
from arm2d_env import Arm2DEnv


def find_model_config(model_path, config_path=None):
    """Carga la configuración asociada al modelo o usa valores por defecto."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # Intentar buscar experiment_meta.json en la carpeta del modelo
    if model_path:
        # Si model_path es 'models/exp2_smooth_precision/best_model/best_model.zip',
        # buscamos el experiment_meta.json en la carpeta del experimento
        model_dir = os.path.dirname(os.path.abspath(model_path))
        
        # Buscar en la carpeta actual y en la carpeta padre
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


def interactive_render(model_path=None, config_path=None):
    print("=" * 60)
    print(" SIMULADOR INTERACTIVO DE BRAZO ROBÓTICO 2D (PPO)")
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
            "models/exp2_smooth_precision/best_model/best_model.zip",
            "models/exp1_baseline/best_model/best_model.zip",
            "models/PPO/best_model/best_model.zip"
        ]
        for p in default_paths:
            if os.path.exists(p):
                model_path = p
                break

    # Cargar la configuración exacta del modelo para que coincidan los observation spaces
    cfg = find_model_config(model_path, config_path)

    # Crear entorno con la configuración adecuada
    env = Arm2DEnv(render_mode="human", config=cfg)
    obs, info = env.reset()

    # Cargar modelo entrenado si existe
    model = None
    if model_path and os.path.exists(model_path):
        print(f">>> Cargando modelo entrenado desde: {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print(">>> [AVISO] No se encontró modelo. Ejecutando con acciones aleatorias.")

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
    parser = argparse.ArgumentParser(description="Simulador interactivo para Brazo 2D")
    parser.add_argument("model_path", nargs="?", default=None, help="Ruta al modelo .zip")
    parser.add_argument("--config", "-c", type=str, default=None, help="Ruta opcional al YAML de configuración")
    args = parser.parse_args()

    interactive_render(args.model_path, args.config)
