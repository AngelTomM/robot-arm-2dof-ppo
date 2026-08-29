"""
Paso 5: Renderizado Interactivo en Tiempo Real para Brazo Robótico 2 DOF.
Permite ver al agente de RL en acción y controlar el objetivo de forma interactiva con el mouse o con la barra espaciadora.
"""

import os
import sys
import math
import numpy as np
import pygame
from stable_baselines3 import PPO
from arm2d_env import Arm2DEnv


def interactive_render(model_path=None):
    print("=" * 60)
    print(" SIMULADOR INTERACTIVO DE BRAZO ROBÓTICO 2D (PPO)")
    print("=" * 60)
    print("Controles:")
    print(" - Click Izquierdo en la ventana : Mover el objetivo (Target) a esa posición")
    print(" - Barra Espaciadora            : Generar nuevo objetivo aleatorio")
    print(" - Tecla 'R'                     : Reiniciar posición del brazo")
    print(" - Tecla 'ESC' o cerrar ventana  : Salir")
    print("=" * 60)

    # Crear entorno en modo 'human'
    env = Arm2DEnv(render_mode="human")
    obs, info = env.reset()

    # Cargar modelo entrenado si existe
    model = None
    if model_path and os.path.exists(model_path):
        print(f">>> Cargando modelo entrenado desde: {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        # Buscar modelo por defecto si existe
        default_paths = [
            "models/PPO/best_model/best_model.zip",
            "models/PPO/ppo_arm2d_final.zip"
        ]
        for p in default_paths:
            if os.path.exists(p):
                print(f">>> Encontrado modelo: {p}")
                model = PPO.load(p, env=env)
                break
        if model is None:
            print(">>> [AVISO] No se encontró modelo entrenado. Ejecutando con acciones aleatorias.")

    running = True
    manual_target = False

    while running:
        # Manejo de eventos de Pygame
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
                if event.button == 1:  # Click izquierdo
                    # Convertir coordenadas de pantalla a métricas del simulador
                    mx, my = pygame.mouse.get_pos()
                    world_x = (mx - env.origin[0]) / env.scale
                    world_y = -(my - env.origin[1]) / env.scale
                    env.set_target(world_x, world_y)
                    manual_target = True
                    obs = env._get_obs()

        # Decisión de acción (modelo RL o aleatoria)
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        if (terminated or truncated) and not manual_target:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    # Si se pasa la ruta del modelo como argumento por línea de comandos
    custom_model = sys.argv[1] if len(sys.argv) > 1 else None
    interactive_render(custom_model)
