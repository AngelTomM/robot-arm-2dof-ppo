"""
Visualizador y Renderizado Interactivo en Tiempo Real para Manipulador 2-DOF (MuJoCo Reacher-v5).
Permite evaluar visualmente políticas de RL (PPO, SAC) o controladores analíticos (C0) en 3D.
"""
import os
import sys
import json
import time
import argparse
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import yaml

from src import Arm2DEnv, JacobianDampedController, load_env_config


def load_trained_model(model_path: str):
    """Carga un modelo serializado de Stable-Baselines3 detectando el algoritmo del .zip."""
    from stable_baselines3 import PPO, SAC

    algo = None
    try:
        with zipfile.ZipFile(model_path) as zf:
            data = json.loads(zf.read("data").decode("utf-8"))
        policy_module = str(data.get("policy_class", {}).get("__module__", ""))
        algo = "SAC" if "sac" in policy_module else "PPO"
    except Exception as e:
        print(f">>> [AVISO] Metadatos no legibles ({e}); probando PPO y luego SAC.")

    if algo == "SAC":
        print(">>> [MODELO] Algoritmo detectado: SAC")
        return SAC.load(str(model_path))
    if algo == "PPO":
        print(">>> [MODELO] Algoritmo detectado: PPO")
        try:
            return PPO.load(str(model_path))
        except Exception:
            pass

    for loader, nombre in ((PPO.load, "PPO"), (SAC.load, "SAC")):
        try:
            model = loader(str(model_path))
            print(f">>> [MODELO] Algoritmo detectado: {nombre}")
            return model
        except Exception:
            continue
    raise ValueError(f"No se pudo cargar el modelo desde: {model_path}")


def interactive_render(model_path: Optional[str] = None, config_path: Optional[str] = None,
                       c0: bool = False, episodes: int = 10):
    """Bucle de renderizado interactivo en MuJoCo."""
    env_cfg = load_env_config(config_path) if config_path else load_env_config()
    print(">>> Inicializando entorno de simulación dinámica MuJoCo Reacher-v5 (render_mode='human')...")

    env = Arm2DEnv(render_mode="human", config=env_cfg)

    # Buscar estadísticas VecNormalize si aplica
    norm_wrapper = None
    if model_path and not c0:
        model_dir = os.path.dirname(os.path.abspath(model_path))
        for cand in [
            os.path.join(model_dir, "vecnormalize_stats.pkl"),
            os.path.join(os.path.dirname(model_dir), "vecnormalize_stats.pkl"),
            os.path.join(os.path.dirname(model_dir), "models", "vecnormalize_stats.pkl"),
        ]:
            if os.path.exists(cand):
                try:
                    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
                    dummy = DummyVecEnv([lambda: Arm2DEnv(config=env_cfg)])
                    norm_wrapper = VecNormalize.load(cand, dummy)
                    norm_wrapper.training = False
                    norm_wrapper.norm_reward = False
                    print(f">>> [NORM] Estadísticas de normalización cargadas: {cand}")
                    break
                except Exception as e:
                    print(f">>> [AVISO] No se pudo cargar VecNormalize ({cand}): {e}")

    # Cargar agente
    if c0:
        print(">>> [CONTROLADOR] Ejecutando baseline cartesiano PD con transposición del Jacobiano (C0).")
        agent = JacobianDampedController.from_config(env_cfg)
    elif model_path and os.path.exists(model_path):
        agent = load_trained_model(model_path)
    else:
        print(">>> [AVISO] Sin modelo ni C0 especificado; ejecutando controlador analítico C0 por defecto.")
        agent = JacobianDampedController.from_config(env_cfg)

    print("=" * 70)
    print(" Presione Ctrl+C en la consola para interrumpir la simulación.")
    print("=" * 70)

    try:
        for ep in range(episodes):
            obs, info = env.reset()
            ep_return = 0.0
            step_count = 0

            print(f"\n--- Episodio {ep + 1}/{episodes} ---")
            print(f"Objetivo: ({info['target_x']:.3f}, {info['target_y']:.3f}) | Distancia inicial: {info['distance']:.4f} m")

            while True:
                policy_obs = norm_wrapper.normalize_obs(obs) if norm_wrapper else obs
                if hasattr(agent, "predict"):
                    pred_out = agent.predict(policy_obs, deterministic=True)
                    action = pred_out[0] if isinstance(pred_out, tuple) else pred_out
                else:
                    action = agent.get_action(policy_obs)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                step_count += 1

                # Renderizar fotograma en la ventana de MuJoCo
                env.render()
                time.sleep(0.01)

                if step_count % 20 == 0 or terminated or truncated:
                    print(f"Paso {step_count:3d}: Dist = {info['distance']:.4f} m | Vel = {info['joint_speed']:.3f} rad/s | Torque = [{action[0]:.2f}, {action[1]:.2f}] | Rew = {reward:.3f}")

                if terminated or truncated:
                    print(f"Episodio {ep + 1} finalizado. Retorno acumulado: {ep_return:.2f} | Distancia final: {info['distance']:.4f} m")
                    time.sleep(0.5)
                    break

    except KeyboardInterrupt:
        print("\n>>> Simulación interrumpida por el usuario.")
    finally:
        env.close()
        print(">>> Entorno MuJoCo cerrado correctamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizador MuJoCo en tiempo real para manipulador planar 2-DOF")
    parser.add_argument("model_path", nargs="?", default=None, help="Ruta al modelo entrenado (.zip)")
    parser.add_argument("--config", "-c", type=str, default=None, help="Ruta al YAML de configuración del entorno")
    parser.add_argument("--c0", action="store_true", help="Ejecutar el controlador cinemático C0")
    parser.add_argument("--episodes", "-e", type=int, default=10, help="Número de episodios a simular")
    args = parser.parse_args()

    interactive_render(args.model_path, args.config, c0=args.c0, episodes=args.episodes)
