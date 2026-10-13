"""
Ejecución de Tanda 1 — Agente X1a (Fase B Extensión):
- Configuración: configs/phase_b/extension/X1a.yaml
- Semillas: [42, 100, 2026]
- Presupuesto: 655.360 timesteps por semilla (80 rollouts de 8.192 timesteps).
- Checkpoints: Cada 40.960 timesteps (16 checkpoints por corrida, exactamente).
- Destino: experiments/phase_b/extension/X1a/seed_<seed>/
- Exporta: reports/data/phase_b/extension/tanda_1_x1a_summary.json
"""
import sys
import os
import time
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_b_ext_train import train_phase_b_ext_agent

def run_tanda_1():
    cfg_path = REPO_ROOT / "configs" / "phase_b" / "extension" / "X1a.yaml"
    seeds = [42, 100, 2026]
    output_base_dir = REPO_ROOT / "experiments" / "phase_b" / "extension"
    reports_data_dir = REPO_ROOT / "reports" / "data" / "phase_b" / "extension"
    reports_data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("=== TANDA 1 (FASE B EXTENSIÓN): AGENTE X1a (3 SEMILLAS) ===")
    print(f"=== Config: {cfg_path.relative_to(REPO_ROOT)}")
    print(f"=== Presupuesto por corrida: 655,360 pasos (16 checkpoints)")
    print("=" * 80)

    summary = {
        "experiment": "X1a",
        "tanda": 1,
        "total_timesteps_per_seed": 655360,
        "checkpoint_freq": 40960,
        "expected_checkpoints_per_seed": 16,
        "runs": []
    }

    tanda_start = time.time()

    for s in seeds:
        print(f"\n>>> [TANDA 1: X1a] Iniciando semilla {s}...")
        t0 = time.time()
        model, exp_dir, ckpts = train_phase_b_ext_agent(
            config_path_or_dict=cfg_path,
            seed=s,
            total_timesteps_override=655360,
            checkpoint_freq_override=40960,
            output_base_dir=str(output_base_dir)
        )
        t_elapsed = time.time() - t0
        fps = 655360.0 / t_elapsed if t_elapsed > 0 else 0.0

        # Verificación estricta de checkpoints guardados
        ckpts_dir = exp_dir / "checkpoints"
        saved_dirs = sorted([p for p in ckpts_dir.iterdir() if p.is_dir()], key=lambda p: int(p.name.split("_")[1]))
        ckpts_count = len(saved_dirs)

        # Verificar integridad de cada checkpoint
        all_models_exist = all((d / "model.zip").exists() for d in saved_dirs)
        norm_files_count = sum(1 for d in saved_dirs if (d / "vecnormalize_stats.pkl").exists())

        run_info = {
            "seed": s,
            "exp_dir": str(exp_dir.relative_to(REPO_ROOT)),
            "elapsed_seconds": round(t_elapsed, 2),
            "elapsed_minutes": round(t_elapsed / 60.0, 2),
            "fps": round(fps, 1),
            "checkpoints_count": ckpts_count,
            "all_models_exist": all_models_exist,
            "has_vecnormalize_files": norm_files_count,
            "status": "COMPLETED" if (ckpts_count == 16 and all_models_exist) else "INCOMPLETE"
        }
        summary["runs"].append(run_info)

        print(f">>> [TANDA 1: X1a] Semilla {s} finalizada en {t_elapsed/60.0:.2f} min ({fps:.1f} FPS)")
        print(f">>> Checkpoints verificados: {ckpts_count}/16 | Modelos íntegros: {all_models_exist}")

    total_tanda_time = time.time() - tanda_start
    summary["total_elapsed_seconds"] = round(total_tanda_time, 2)
    summary["total_elapsed_minutes"] = round(total_tanda_time / 60.0, 2)
    summary["total_checkpoints"] = sum(r["checkpoints_count"] for r in summary["runs"])

    summary_file = reports_data_dir / "tanda_1_x1a_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"=== TANDA 1 CONCLUIDA EXITOSAMENTE EN {summary['total_elapsed_minutes']} MINUTOS ===")
    print(f"=== Resumen guardado en: {summary_file.relative_to(REPO_ROOT)} ===")
    print("=" * 80)

if __name__ == "__main__":
    run_tanda_1()
