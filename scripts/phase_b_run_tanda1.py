"""
Runner de Tanda 1 de Fase B: X5_near (3 semillas: 42, 100, 2026).
Presupuesto: 1.007.616 pasos (123 rollouts).
Guarda 25 checkpoints atómicos (k=5 rollouts = 40.960 pasos).
"""
import sys
import time
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_b_train import train_phase_b_agent

def run_tanda_1():
    cfg_path = REPO_ROOT / "configs" / "phase_b" / "X5_near.yaml"
    seeds = [42, 100, 2026]
    total_steps = 1007616
    ckpt_freq = 40960
    
    tanda_report = {
        "experiment": "X5_near",
        "seeds": seeds,
        "total_timesteps_exact": total_steps,
        "checkpoint_step_interval": ckpt_freq,
        "runs": {}
    }

    print(f"\n=======================================================")
    print(f"=== INICIANDO TANDA 1 FASE B: X5_near (3 SEMILLAS) ===")
    print(f"=======================================================")

    t0_global = time.time()

    for s in seeds:
        print(f"\n>>> [TANDA 1] Iniciando Semilla {s} ({seeds.index(s)+1}/{len(seeds)})...")
        t0 = time.time()
        
        model, exp_dir, ckpts = train_phase_b_agent(
            config_path_or_dict=cfg_path,
            seed=s,
            total_timesteps_override=total_steps,
            checkpoint_freq_override=ckpt_freq,
            output_base_dir=str(REPO_ROOT / "experiments" / "phase_b")
        )
        elapsed = time.time() - t0
        fps = total_steps / elapsed if elapsed > 0 else 0.0

        # Verificación de integridad de checkpoints guardados
        ckpts_dir = exp_dir / "checkpoints"
        saved_dirs = sorted([p for p in ckpts_dir.iterdir() if p.is_dir()])
        all_intact = True
        missing_files = []
        for c in saved_dirs:
            if not (c / "model.zip").exists():
                all_intact = False
                missing_files.append(f"{c.name}: model.zip faltante")
            if not (c / "vecnormalize_stats.pkl").exists():
                all_intact = False
                missing_files.append(f"{c.name}: vecnormalize_stats.pkl faltante")

        tanda_report["runs"][str(s)] = {
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60.0,
            "mean_fps": fps,
            "checkpoints_count": len(saved_dirs),
            "checkpoints_steps": [int(p.name.replace("ckpt_", "")) for p in saved_dirs],
            "all_checkpoints_intact": all_intact,
            "missing_files": missing_files
        }

        print(f">>> [TANDA 1 OK] Semilla {s} finalizada en {elapsed/60.0:.2f} min ({fps:.1f} FPS)")
        print(f">>> [TANDA 1 OK] Checkpoints guardados: {len(saved_dirs)} (Todos íntegros: {all_intact})")

    total_elapsed = time.time() - t0_global
    tanda_report["total_elapsed_minutes"] = total_elapsed / 60.0

    report_path = REPO_ROOT / "reports" / "data" / "phase_b" / "tanda_1_x5_near_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(tanda_report, f, indent=2)

    print(f"\n=======================================================")
    print(f"=== TANDA 1 COMPLETADA EN {total_elapsed/60.0:.2f} MINUTOS ===")
    print(f"=== Reporte guardado en: {report_path} ===")
    print(f"=======================================================")

if __name__ == "__main__":
    run_tanda_1()
