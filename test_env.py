"""
Script de verificación rápida del entorno Gymnasium personalizado.
"""
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from arm2d_env import Arm2DEnv

def test_environment():
    print(">>> Instanciando Arm2DEnv...")
    env = Arm2DEnv()
    
    print(">>> Ejecutando Gymnasium check_env...")
    check_env(env.unwrapped)
    print(">>> [OK] ¡Entorno 100% compatible con la API de Gymnasium!")

    obs, info = env.reset()
    print(f">>> Obs inicial (shape {obs.shape}): {obs}")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Paso {step+1}: Reward={reward:.4f}, Terminated={terminated}, Info={info}")

    env.close()
    print(">>> [OK] Test de pasos finalizado con éxito.")

if __name__ == "__main__":
    test_environment()
