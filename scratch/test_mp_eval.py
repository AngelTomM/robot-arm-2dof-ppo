import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time
from stable_baselines3 import PPO

def worker_eval(path_str):
    m = PPO.load(path_str)
    return str(m)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    p = "experiments/phase_b/extension/X1b/seed_42/checkpoints/ckpt_40960/model.zip"
    print("Testing 1 call in pool...")
    with ProcessPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(worker_eval, [p]))
    print("Success:", res)
