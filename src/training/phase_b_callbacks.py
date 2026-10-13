"""
Callbacks para Fase B: Guardado atómico acoplado de Checkpoints (Modelo + VecNormalize).
Asegura que cada checkpoint conserve de manera sincronizada e íntegra sus estadísticas de normalización.
"""
import os
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

class PhaseBCheckpointCallback(BaseCallback):
    """
    Guarda periódicamente checkpoints conteniendo tanto el modelo .zip
    como el archivo vecnormalize_stats.pkl en un subdirectorio exclusivo por paso.
    """
    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.last_saved_step = 0
        self.saved_checkpoints = []

    def _save_checkpoint(self, step: int):
        ckpt_dir = self.save_path / f"ckpt_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Guardar pesos del modelo
        model_path = ckpt_dir / "model.zip"
        self.model.save(str(model_path))

        # 2. Guardar estadísticas de VecNormalize si aplica
        vec_norm = self.training_env
        if isinstance(vec_norm, VecNormalize):
            stats_path = ckpt_dir / "vecnormalize_stats.pkl"
            vec_norm.save(str(stats_path))
        
        self.last_saved_step = step
        self.saved_checkpoints.append(step)
        if self.verbose > 0:
            print(f">>> [CHECKPOINT] Guardado en paso {step:,}: {ckpt_dir}")

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.num_timesteps > 0:
            # Guardar si se superó el múltiplo de save_freq y no se guardó en este paso
            if self.num_timesteps % self.save_freq == 0 and self.num_timesteps != self.last_saved_step:
                self._save_checkpoint(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        # Asegurar guardado en el paso final exacto
        if self.num_timesteps != self.last_saved_step:
            self._save_checkpoint(self.num_timesteps)
