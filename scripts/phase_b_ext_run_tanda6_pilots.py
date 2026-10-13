"""
Ejecución de Tanda 6 — Pilotos n=1 (X1b, X1c, X1s, X2b) (Fase B Extensión):
- Semilla: [42] (solo 1 corrida por piloto, excluidos de regla 4.2).
- Presupuesto: 655.360 timesteps por piloto (80 rollouts de 8.192 timesteps).
- Checkpoints: Cada 40.960 timesteps (16 checkpoints por corrida).
- Destino: experiments/phase_b/extension/<agente>/seed_42/
- Exporta: reports/data/phase_b/extension/tanda_6_pilots_summary.json
"""
import sys
import os
import time
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_b_ext_train import train_phase_b_ext_agent

def run_tanda_6():
    pilots = ["X1b", "X1c", "X1s", "X2b"]
    seed = 42
    output_base_dir = REPO_ROOT / "experiments" / "phase_b" / "extension"
    reports_data_dir = REPO_ROOT / "reports" / "data" / "phase_b" / "extension"
    reports_data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("=== TANDA 6 (FASE B EXTENSIÓN): PILOTOS n=1 (X1b, X1c, X1s, X2b) ===")
    print(f"=== Semilla única: {seed}")
    print(f"=== Presupuesto por piloto: 655,360 pasos (16 checkpoints)")
    print("=" * 80)

    summary = {
        "experiment": "Pilots_n1",
        "tanda": 6,
        "pilots": pilots,
        "seed": seed,
        "total_timesteps_per_pilot": 655360,
        "checkpoint_freq": 40960,
        "expected_checkpoints_per_pilot": 16,
        "runs": []
    }

    tanda_start = time.time()

    for p in pilots:
        cfg_path = REPO_ROOT / "configs" / "phase_b" / "extension" / f"{p}.yaml"
        print(f"\n>>> [TANDA 6: PILOTO {p}] Iniciando entrenamiento semilla {seed}...")
        t0 = time.time()
        model, exp_dir, ckpts = train_phase_b_ext_agent(
            config_path_or_dict=cfg_path,
            seed=seed,
            total_timesteps_override=655360,
            checkpoint_freq_override=40960,
            output_base_dir=str(output_base_dir)
        )
        t_elapsed = time.time() - t0
        fps = 655360.0 / t_elapsed if t_elapsed > 0 else 0.0

        ckpts_dir = exp_dir / "checkpoints"
        saved_dirs = sorted([d for d in ckpts_dir.iterdir() if d.is_dir()], key=lambda d: int(d.name.split("_")[1]))
        ckpts_count = len(saved_dirs)

        all_models_exist = all((d / "model.zip").exists() for d in saved_dirs)
        
        # Check vecnormalize if required by config
        norm_files_count = sum(1 for d in saved_dirs if (d / "vecnormalize_stats.pkl").exists())
        # X1b, X1c, X1s don't use norm_obs; X2b uses norm_obs
        expect_norm = (p == "X2b")
        norm_ok = (norm_files_count == ckpts_count) if expect_norm else (norm_files_count == 0)

        run_info = {
            "pilot": p,
            "seed": seed,
            "exp_dir": str(exp_dir.relative_to(REPO_ROOT)),
            "elapsed_seconds": round(t_elapsed, 2),
            "elapsed_minutes": round(t_elapsed / 60.0, 2),
            "fps": round(fps, 1),
            "checkpoints_count": ckpts_count,
            "all_models_exist": all_models_exist,
            "vecnormalize_count": norm_files_count,
            "expected_vecnormalize": expect_norm,
            "status": "COMPLETED" if (ckpts_count == 16 and all_models_exist and norm_ok) else "INCOMPLETE"
        }
        summary["runs"].append(run_info)

        print(f">>> [TANDA 6: {p}] Finalizado en {t_elapsed/60.0:.2f} min ({fps:.1f} FPS)")
        print(f">>> Checkpoints verificados: {ckpts_count}/16 | Modelos: {all_models_exist} | VecNormalize: {norm_files_count}")

    total_tanda_time = time.time() - tanda_start
    summary["total_elapsed_seconds"] = round(total_tanda_time, 2)
    summary["total_elapsed_minutes"] = round(total_tanda_time / 60.0, 2)
    summary["total_checkpoints"] = sum(r["checkpoints_count"] for r in summary["runs"])

    summary_file = reports_data_dir / "tanda_6_pilots_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"=== TANDA 6 CONCLUIDA EXITOSAMENTE EN {summary['total_elapsed_minutes']} MINUTOS ===")
    print(f"=== Resumen guardado en: {summary_file.relative_to(REPO_ROOT)} ===")
    print("=" * 80)

if __name__ == "__main__":
    run_tanda_6()
