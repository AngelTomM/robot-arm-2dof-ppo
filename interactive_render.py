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
import zipfile
import numpy as np
import pygame
import yaml
from stable_baselines3 import PPO, SAC
from src import Arm2DEnv


def load_trained_model(model_path):
    """Carga el modelo SB3 detectando el algoritmo desde su propio .zip.

    Un zip de SB3 guarda en 'data' la clase de política serializada; el
    campo '__module__' ('stable_baselines3.sac.policies' frente a
    'stable_baselines3.common.policies') permite distinguir un SAC de un
    PPO sin depender del nombre del fichero. Con retén: si los metadatos
    no se pueden leer, se prueba PPO y después SAC.
    """
    algo = None
    try:
        with zipfile.ZipFile(model_path) as zf:
            data = json.loads(zf.read("data").decode("utf-8"))
        policy_module = str(data.get("policy_class", {}).get("__module__", ""))
        algo = "SAC" if "sac" in policy_module else "PPO"
    except Exception as e:
        print(f">>> [AVISO] No se pudo leer el algoritmo del zip ({e}); probando ambos.")

    if algo == "SAC":
        print(">>> [MODELO] Algoritmo detectado: SAC")
        return SAC.load(str(model_path))
    if algo == "PPO":
        print(">>> [MODELO] Algoritmo detectado: PPO")
        try:
            return PPO.load(str(model_path))
        except Exception:
            pass
    # Retén: metadatos ilegibles o PPO fallido -> probar SAC (y viceversa)
    for loader, nombre in ((PPO.load, "PPO"), (SAC.load, "SAC")):
        try:
            model = loader(str(model_path))
            print(f">>> [MODELO] Algoritmo detectado (por retén): {nombre}")
            return model
        except Exception:
            continue
    raise ValueError(f"No se pudo cargar el modelo como PPO ni como SAC: {model_path}")


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
        
        # Buscar en la carpeta actual y en carpetas superiores (útil para checkpoints)
        candidates = [
            os.path.join(model_dir, "experiment_meta.json"),
            os.path.join(os.path.dirname(model_dir), "experiment_meta.json"),
            os.path.join(os.path.dirname(os.path.dirname(model_dir)), "experiment_meta.json"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(model_dir))), "experiment_meta.json")
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
    print(" SIMULADOR INTERACTIVO DE BRAZO ROBÓTICO 2D (PPO/SAC)")
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
            "experiments/phase_X/prueba/PPO_1/seed_42/models/model.zip",
            "experiments/phase_X/prueba/SAC_1/seed_42/models/model.zip",
            "experiments/phase_X/smoke/PPO/seed_42/models/model.zip"
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

    # Buscar estadísticas de normalización si existen (Fase A)
    norm_wrapper = None
    if model_path:
        model_dir = os.path.dirname(os.path.abspath(model_path))
        norm_candidates = [
            os.path.join(model_dir, "vecnormalize_stats.pkl"),
            os.path.join(os.path.dirname(model_dir), "vecnormalize_stats.pkl"),
            os.path.join(os.path.dirname(model_dir), "models", "vecnormalize_stats.pkl")
        ]
        for norm_file in norm_candidates:
            if os.path.exists(norm_file):
                try:
                    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
                    dummy = DummyVecEnv([lambda: Arm2DEnv(config=cfg)])
                    norm_wrapper = VecNormalize.load(norm_file, dummy)
                    norm_wrapper.training = False
                    norm_wrapper.norm_reward = False
                    print(f">>> [NORM] Estadísticas de normalización cargadas desde: {norm_file}")
                    break
                except Exception as e:
                    print(f">>> [AVISO] No se pudo cargar VecNormalize ({norm_file}): {e}")

    # Cargar modelo entrenado si existe
    model = None
    if model_path and os.path.exists(model_path):
        print(f">>> Cargando modelo entrenado desde: {model_path}")
        model = load_trained_model(model_path)
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
            policy_obs = norm_wrapper.normalize_obs(obs) if norm_wrapper else obs
            action, _ = model.predict(policy_obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        # HUD en pantalla (Telemetría continua LQR)
        if env.screen is not None:
            try:
                pygame.font.init()
                font = pygame.font.SysFont("Consolas", 14)
                dist_s = font.render(f"Dist:  {info['distance']:.4f} m", True, (240, 240, 240))
                spd_s = font.render(f"Speed: {info['joint_speed']:.4f} rad/s", True, (240, 240, 240))
                rew_s = font.render(f"Rew:   {reward:.4f}", True, (46, 204, 113) if reward > -0.1 else (231, 76, 60))
                env.screen.blit(dist_s, (15, 15))
                env.screen.blit(spd_s, (15, 35))
                env.screen.blit(rew_s, (15, 55))
                pygame.display.flip()
            except Exception:
                pass

        if (terminated or truncated) and not manual_target:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulador interactivo para Brazo 2D")
    parser.add_argument("model_path", nargs="?", default=None, help="Ruta al modelo .zip")
    parser.add_argument("--config", "-c", type=str, default=None, help="Ruta opcional al YAML de configuración")
    args = parser.parse_args()

    interactive_render(args.model_path, args.config)
