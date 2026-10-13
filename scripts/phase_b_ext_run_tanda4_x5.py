"""
Ejecución de Tanda 4 — Agente X5 (Fase B Extensión):
- Configuración: configs/phase_b/extension/X5.yaml
- Semillas: [42, 100, 2026]
- Presupuesto: 655.360 timesteps por semilla (80 rollouts de 8.192 timesteps).
- Checkpoints: Cada 40.960 timesteps (16 checkpoints por corrida, exactamente).
- Arquitectura: 256x256 ReLU, recompensa x5_vel (w_v=0.1, sigma_v=0.15), VecNormalize(obs=True, reward=False).
- Destino: experiments/phase_b/extension/X5/seed_<seed>/
- Exporta: reports/data/phase_b/extension/tanda_4_x5_summary.json
"""
import sys
import os
import time
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_b_ext_train import train_phase_b_ext_agent

def run_tanda_4():
    cfg_path = REPO_ROOT / "configs" / "phase_b" / "extension" / "X5.yaml"
    seeds = [42, 100, 2026]
    output_base_dir = REPO_ROOT / "experiments" / "phase_b" / "extension"
    reports_data_dir = REPO_ROOT / "reports" / "data" / "phase_b" / "extension"
    reports_data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("=== TANDA 4 (FASE B EXTENSIÓN): AGENTE X5 (3 SEMILLAS) ===")
    print(f"=== Config: {cfg_path.relative_to(REPO_ROOT)}")
    print(f"=== Presupuesto por corrida: 655,360 pasos (16 checkpoints)")
    print(f"=== Recompensa: x5_vel (w_v=0.1, sigma_v=0.15) + VecNormalize(obs)")
    print("=" * 80)

    summary = {
        "experiment": "X5",
        "tanda": 4,
        "total_timesteps_per_seed": 655360,
        "checkpoint_freq": 40960,
        "expected_checkpoints_per_seed": 16,
        "runs": []
    }

    tanda_start = time.time()

    for s in seeds:
        print(f"\n>>> [TANDA 4: X5] Iniciando semilla {s}...")
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

        ckpts_dir = exp_dir / "checkpoints"
        saved_dirs = sorted([p for p in ckpts_dir.iterdir() if p.is_dir()], key=lambda p: int(p.name.split("_")[1]))
        ckpts_count = len(saved_dirs)

        all_models_exist = all((d / "model.zip").exists() for d in saved_dirs)
        norm_files_count = sum(1 for d in saved_dirs if (d / "vecnormalize_stats.pkl").exists())
        all_norm_exist = (norm_files_count == ckpts_count)

        run_info = {
            "seed": s,
            "exp_dir": str(exp_dir.relative_to(REPO_ROOT)),
            "elapsed_seconds": round(t_elapsed, 2),
            "elapsed_minutes": round(t_elapsed / 60.0, 2),
            "fps": round(fps, 1),
            "checkpoints_count": ckpts_count,
            "all_models_exist": all_models_exist,
            "all_vecnormalize_exist": all_norm_exist,
            "status": "COMPLETED" if (ckpts_count == 16 and all_models_exist and all_norm_exist) else "INCOMPLETE"
        }
        summary["runs"].append(run_info)

        print(f">>> [TANDA 4: X5] Semilla {s} finalizada en {t_elapsed/60.0:.2f} min ({fps:.1f} FPS)")
        print(f">>> Checkpoints verificados: {ckpts_count}/16 | Modelos: {all_models_exist} | VecNormalize: {all_norm_exist}")

    total_tanda_time = time.time() - tanda_start
    summary["total_elapsed_seconds"] = round(total_tanda_time, 2)
    summary["total_elapsed_minutes"] = round(total_tanda_time / 60.0, 2)
    summary["total_checkpoints"] = sum(r["checkpoints_count"] for r in summary["runs"])

    summary_file = reports_data_dir / "tanda_4_x5_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"=== TANDA 4 CONCLUIDA EXITOSAMENTE EN {summary['total_elapsed_minutes']} MINUTOS ===")
    print(f"=== Resumen guardado en: {summary_file.relative_to(REPO_ROOT)} ===")
    print("=" * 80)

if __name__ == "__main__":
    run_tanda_4()
