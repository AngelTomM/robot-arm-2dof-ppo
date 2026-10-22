"""
Callbacks para el Registro de Métricas Continuas LQR y Sincronización de VecNormalize (Fase D).
"""
import collections
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization


class LQRMetricsCallback(BaseCallback):
    """
    Callback para registrar en TensorBoard métricas continuas del entorno MDP:
    - Retorno medio del episodio (ep_rew_mean)
    - Distancia cartesiana final media (final_distance_mean)
    - Velocidad articular final media (final_joint_speed_mean)
    - Longitud media de episodio (ep_len_mean)
    """
    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.episode_returns = collections.deque(maxlen=window_size)
        self.episode_lengths = collections.deque(maxlen=window_size)
        self.episode_distances = collections.deque(maxlen=window_size)
        self.episode_speeds = collections.deque(maxlen=window_size)
        self.total_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.total_episodes += 1
                ep_r = float(info["episode"]["r"])
                ep_l = int(info["episode"]["l"])
                dist = float(info.get("distance", 0.0))
                spd = float(info.get("joint_speed", 0.0))

                self.episode_returns.append(ep_r)
                self.episode_lengths.append(ep_l)
                self.episode_distances.append(dist)
                self.episode_speeds.append(spd)

                if self.total_episodes % 10 == 0:
                    self.logger.record("train/ep_rew_mean", float(np.mean(self.episode_returns)))
                    self.logger.record("train/ep_len_mean", float(np.mean(self.episode_lengths)))
                    self.logger.record("train/final_distance_mean", float(np.mean(self.episode_distances)))
                    self.logger.record("train/final_joint_speed_mean", float(np.mean(self.episode_speeds)))
                    self.logger.record("train/total_episodes_completed", self.total_episodes)

        return True


class VecNormalizeSyncEvalCallback(EvalCallback):
    """
    EvalCallback que sincroniza estrictamente las estadísticas de VecNormalize
    manteniendo training=False y norm_reward=False en el entorno de evaluación.
    """
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                sync_envs_normalization(self.training_env, self.eval_env)
                self.eval_env.training = False
                self.eval_env.norm_reward = False

            return super()._on_step()
        return True


# Alias de compatibilidad
PhaseATrainingCallback = LQRMetricsCallback
PhaseAEvalCallback = VecNormalizeSyncEvalCallback
