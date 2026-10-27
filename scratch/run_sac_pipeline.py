import os
import sys
import time
import subprocess
from pathlib import Path

REPO_ROOT = Path(r"c:\Users\ulica\Documents\AprendizajeAuo2026\Fase_D_V3")
PYTHON_EXE = sys.executable

MODELS = [
    "SAC_4",
    "SAC_BUF_L",
    "SAC_BUF_S",
    "SAC_EXP_H",
    "SAC_EXP_LOW",
    "SAC_NOM_NREWARD",
    "SAC_NOM",
]

CONFIG_MAP = {
    m: REPO_ROOT / f"configs/SAC_eval/{m}.yaml" for m in MODELS
}

COMPARATIVES = [
    "SAC_BUF_S,SAC_NOM,SAC_BUF_L",
    "SAC_EXP_LOW,SAC_NOM,SAC_EXP_H",
    "SAC_4,SAC_NOM,SAC_NOM_NREWARD",
]


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def run_cmd(cmd_list, description=""):
    log(f"--- INICIO: {description} ---")
    log(f"Comando: {' '.join(str(x) for x in cmd_list)}")
    t0 = time.time()
    res = subprocess.run(cmd_list, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0
    if res.returncode != 0:
        log(f"ERROR: Falló {description} con código {res.returncode} en {elapsed:.1f}s")
        return False
    log(f"COMPLETADO: {description} en {elapsed:.1f}s ({elapsed/60:.2f} min)")
    return True


def main():
    log("=== INICIANDO PIPELINE DE ENTRENAMIENTO, EVALUACIÓN Y GRAFICACIÓN DE SAC ===")
    
    # 1. ENTRENAMIENTO
    for model_name in MODELS:
        cfg_path = CONFIG_MAP[model_name]
        if not cfg_path.exists():
            log(f"ADVERTENCIA: Archivo de configuración no encontrado: {cfg_path}")
            continue

        cmd = [
            PYTHON_EXE, "train.py",
            "--config", str(cfg_path),
            "--device", "cpu"
        ]
        ok = run_cmd(cmd, f"Entrenamiento de {model_name}")
        if not ok:
            log(f"Interrumpiendo o continuando tras fallo en {model_name}")

    # 2. EVALUACIÓN (300 episodios)
    for model_name in MODELS:
        base_dir = REPO_ROOT / f"experiments/SAC_eval/{model_name}/seed_42/models"
        best_model = base_dir / "best_model/best_model.zip"
        final_model = base_dir / "sac_arm2d_final.zip"

        target_model = best_model if best_model.exists() else final_model
        if not target_model.exists():
            log(f"ADVERTENCIA: No se encontró modelo para evaluar {model_name} en {base_dir}")
            continue

        cmd = [
            PYTHON_EXE, "evaluate.py",
            "--model-path", str(target_model),
            "--label", model_name,
            "--dataset", "300"
        ]
        run_cmd(cmd, f"Evaluación de {model_name} sobre 300 episodios")

    # 3. GENERACIÓN DE FIGURAS INDIVIDUALES
    for model_name in MODELS:
        cmd = [
            PYTHON_EXE, "plot.py",
            "--dataset", "300",
            "--model", model_name
        ]
        run_cmd(cmd, f"Figuras individuales para {model_name}")

    # 4. GENERACIÓN DE FIGURAS COMPARATIVAS
    for comp in COMPARATIVES:
        cmd = [
            PYTHON_EXE, "plot.py",
            "--dataset", "300",
            "--compare", comp,
            "--metric", "all"
        ]
        run_cmd(cmd, f"Comparativa consolidada: {comp}")

    log("=== PIPELINE COMPLETO FINALIZADO CON ÉXITO ===")


if __name__ == "__main__":
    main()
