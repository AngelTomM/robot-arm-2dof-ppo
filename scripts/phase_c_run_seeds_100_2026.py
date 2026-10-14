"""
Script auxiliar para ejecutar secuencialmente el entrenamiento de SAC S1 para las semillas 100 y 2026.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_c_train_sac import train_phase_c_sac_agent

def main():
    seeds = [100, 2026]
    for s in seeds:
        print(f"\n{'#'*80}")
        print(f" INICIANDO ENTRENAMIENTO SECUENCIAL: SEMILLA {s}")
        print(f"{'#'*80}\n")
        train_phase_c_sac_agent("configs/phase_c/S1.yaml", seed=s)
        print(f"\n{'#'*80}")
        print(f" FINALIZADO ENTRENAMIENTO: SEMILLA {s}")
        print(f"{'#'*80}\n")

if __name__ == "__main__":
    main()
