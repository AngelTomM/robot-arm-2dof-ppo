"""
Callbacks Personalizados para el Registro de Métricas de Entrenamiento y Sincronización de VecNormalize (Sección 5.7, 11.3, 16.2).
"""
import os
import collections
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

class PhaseATrainingCallback(BaseCallback):
    """
    Callback para registrar en TensorBoard:
    - Tasa de éxito durante el entrenamiento (global y en ventana de 100 episodios).
    - Longitud de permanencia máxima media (hold_counter_max).
    - Descomposición de episodios terminados vs truncados.
    """
    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.episode_successes = collections.deque(maxlen=window_size)
        self.episode_lengths = collections.deque(maxlen=window_size)
        self.episode_holds = collections.deque(maxlen=window_size)
        
        self.total_episodes = 0
        self.total_successes = 0
        self.total_truncated = 0

    def _on_step(self) -> bool:
        # Extraer infos de todos los entornos vectorizados
        infos = self.locals.get("infos", [])
        for info in infos:
            # En Gymnasium / SB3, Monitor coloca 'episode' en info al finalizar un episodio
            if "episode" in info:
                self.total_episodes += 1
                is_succ = bool(info.get("is_success", False))
                hold_max = int(info.get("hold_counter_max", 0))
                ep_len = int(info["episode"]["l"])

                self.episode_successes.append(1.0 if is_succ else 0.0)
                self.episode_lengths.append(ep_len)
                self.episode_holds.append(hold_max)

                if is_succ:
                    self.total_successes += 1
                else:
                    self.total_truncated += 1

                # Registrar escalares periódicamente
                if self.total_episodes % 10 == 0:
                    win_succ_rate = float(np.mean(self.episode_successes))
                    mean_hold = float(np.mean(self.episode_holds))
                    self.logger.record("train/success_rate_window", win_succ_rate)
                    self.logger.record("train/mean_hold_max_window", mean_hold)
                    self.logger.record("train/total_episodes_completed", self.total_episodes)
                    self.logger.record("train/total_successes", self.total_successes)
                    self.logger.record("train/total_truncated", self.total_truncated)

        return True


class PhaseAEvalCallback(EvalCallback):
    """
    EvalCallback que sincroniza estrictamente las estadísticas de VecNormalize
    manteniendo training=False y norm_reward=False en el entorno de evaluación (Sección 11.3).
    """
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sincronizar estadísticas de normalización del entorno de entrenamiento al de evaluación
            if isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                sync_envs_normalization(self.training_env, self.eval_env)
                self.eval_env.training = False
                self.eval_env.norm_reward = False

            return super()._on_step()
        return True
