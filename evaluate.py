"""
Pipeline de Evaluación Unificado y Agnóstico (Fase D).

Evalúa políticas de Aprendizaje por Refuerzo (PPO, SAC) o el controlador
cinemático analítico C0 sobre datasets reproducibles bajo idénticas condiciones físicas.
Calcula formalmente métricas de control mecatrónico:
  - Error estacionario (e_ss)
  - Velocidad residual (v_ss)
  - Tiempo de establecimiento (t_s)
  - Costo de control (J_u)
  - Retorno canónico LQR (J_lqr)
  - Tasa de éxito e intervalo de Wilson
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from stable_baselines3 import PPO, SAC

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import (
    JacobianDampedController,
    evaluate_on_dataset,
    save_and_get_datasets,
    load_env_config,
)


def load_rl_agent(model_path: Path) -> Tuple[Any, Optional[Path]]:
    """
    Carga un modelo entrenado en Stable-Baselines3 detectando el algoritmo (PPO/SAC)
    y localiza estadísticas de normalización VecNormalize si existen.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el archivo del modelo: {model_path}")

    # Localizar estadísticas de normalización adyacentes
    model_dir = model_path.parent
    norm_candidates = [
        model_dir / "vecnormalize_stats.pkl",
        model_dir.parent / "models" / "vecnormalize_stats.pkl",
        model_dir.parent / "vecnormalize_stats.pkl"
    ]
    norm_path = next((p for p in norm_candidates if p.exists()), None)

    # Carga de modelo
    for loader, name in ((PPO.load, "PPO"), (SAC.load, "SAC")):
        try:
            model = loader(str(model_path))
            print(f">>> [EVAL] Modelo cargado exitosamente como {name}: {model_path.name}")
            return model, norm_path
        except Exception:
            continue

    raise ValueError(f"No fue posible instanciar el modelo como PPO ni como SAC: {model_path}")


def run_evaluation(agent_type: str,
                   model_path: Optional[str] = None,
                   vec_norm_path: Optional[str] = None,
                   label: Optional[str] = None,
                   kp: float = 50.0,
                   kd: float = 0.05,
                   lambda_damping: float = 0.01,
                   dataset_path: Optional[str] = None,
                   env_config_path: Optional[str] = None,
                   output_dir: str = "reports/data/evaluation",
                   eval_mode: str = "deterministic",
                   generate_plots: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Ejecuta la evaluación agnóstica para un agente o controlador sobre el dataset de prueba."""
    env_cfg = load_env_config(env_config_path)

    # 1. Cargar conjunto de datos
    if dataset_path and Path(dataset_path).exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        ds_name = Path(dataset_path).stem
    else:
        val_set, _ = save_and_get_datasets()
        dataset = val_set
        ds_name = "validation_set_200"

    # 2. Configurar agente a evaluar
    if agent_type.lower() == "c0":
        agent_label = label or "C0"
        agent = JacobianDampedController(kp=kp, kd=kd, lambda_damping=lambda_damping, env_config=env_cfg)
        norm_file = None
    else:
        if not model_path:
            raise ValueError("Se requiere --model-path para evaluar modelos de RL.")
        m_path = Path(model_path)
        if label:
            agent_label = label
        elif m_path.stem in ["best_model", "model"] and len(m_path.resolve().parents) >= 4:
            agent_label = m_path.resolve().parents[3].name
        else:
            agent_label = m_path.stem
        agent, auto_norm = load_rl_agent(m_path)
        norm_file = Path(vec_norm_path) if vec_norm_path else auto_norm


    print(f"\n{'='*75}")
    print(f" EVALUACIÓN MECATRÓNICA: {agent_label} ({eval_mode.upper()})")
    print(f" Dataset: {ds_name} ({len(dataset)} episodios)")
    if norm_file:
        print(f" VecNormalize: {norm_file}")
    print(f"{'='*75}\n")

    # 3. Ejecutar protocolo de evaluación
    eval_res = evaluate_on_dataset(
        agent=agent,
        dataset=dataset,
        dataset_name=ds_name,
        eval_mode=eval_mode,
        env_config=env_cfg,
        vec_normalize_path=str(norm_file) if norm_file else None
    )
    if len(eval_res) == 3:
        episodes_df, summary, trajectories_df = eval_res
    else:
        episodes_df, summary = eval_res
        trajectories_df = None

    # 4. Guardar resultados estructurados en carpeta exclusiva por tamaño de dataset y modelo
    ds_len = len(dataset)
    if ds_len == 200 or "200" in ds_name:
        dataset_tag = "eval_200"
    elif ds_len == 300 or "300" in ds_name:
        dataset_tag = "eval_300"
    else:
        dataset_tag = f"eval_{ds_len}"

    out_dir_path = REPO_ROOT / output_dir
    model_data_dir = out_dir_path / dataset_tag / agent_label
    model_data_dir.mkdir(parents=True, exist_ok=True)

    # Directorio de compatibilidad directa
    legacy_model_dir = out_dir_path / agent_label
    legacy_model_dir.mkdir(parents=True, exist_ok=True)

    csv_path = model_data_dir / "episodes.csv"
    json_path = model_data_dir / "summary.json"
    legacy_csv = out_dir_path / f"{agent_label}_episodes.csv"
    legacy_json = out_dir_path / f"{agent_label}_summary.json"

    episodes_df.insert(0, "agent", agent_label)
    episodes_df.to_csv(csv_path, index=False)
    episodes_df.to_csv(legacy_model_dir / "episodes.csv", index=False)
    episodes_df.to_csv(legacy_csv, index=False)

    if trajectories_df is not None and not trajectories_df.empty:
        trajectories_df.insert(0, "agent", agent_label)
        trajectories_df.to_csv(model_data_dir / "sample_trajectories.csv", index=False)
        trajectories_df.to_csv(legacy_model_dir / "sample_trajectories.csv", index=False)

    summary_out = {
        "agent": agent_label,
        "eval_set": ds_name,
        "eval_mode": eval_mode,
        "metrics": summary
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)
    with open(legacy_model_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)
    with open(legacy_json, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    # 5. Salida por consola
    print(f">>> [RESULTADOS {agent_label}] ({dataset_tag})")
    print(f"  • Tasa de Éxito:               {summary['success_rate_ci']}")
    print(f"  • Error Estacionario e_ss:     {summary.get('e_ss_mean', 0.0):.4f} m (p90: {summary.get('e_ss_p90', 0.0):.4f} m)")
    print(f"  • Velocidad Residual v_ss:     {summary.get('v_ss_mean', 0.0):.4f} rad/s")
    print(f"  • Tiempo Establecimiento t_s:  {summary.get('t_s_mean', 0.0):.3f} s")
    print(f"  • Esfuerzo de Control J_u:     {summary.get('j_u_mean', 0.0):.4f}")
    print(f"  • Costo LQR Canónico J_lqr:    {summary.get('j_lqr_mean', 0.0):.2f} (std: {summary.get('j_lqr_std', 0.0):.2f})")
    print(f">>> Directorio de datos del modelo: {model_data_dir}")

    # 6. Generación opcional de figuras
    if generate_plots:
        try:
            from plot import generate_all_plots
            n_val = "200" if "200" in dataset_tag or ds_len == 200 else ("300" if "300" in dataset_tag or ds_len == 300 else str(ds_len))
            fig_dir = REPO_ROOT / f"reports/figures_{n_val}"
            generate_all_plots(data_dir=out_dir_path / dataset_tag, out_dir=fig_dir, specific_model=agent_label)
            print(f">>> [PLOTS] Figuras actualizadas en: {fig_dir / agent_label}\n")
        except Exception as e:
            print(f">>> [AVISO] No fue posible generar automáticamente las figuras: {e}\n")

    return episodes_df, summary_out



def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Evaluador unificado de políticas RL y baseline cinemático C0.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agent", type=str, default=None,
                        help="Tipo de agente: 'ppo', 'sac', 'c0' o ruta al modelo .zip.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Ruta directa al archivo de pesos model.zip.")
    parser.add_argument("--label", type=str, default=None,
                        help="Etiqueta identificatoria del modelo para reportes y gráficas (ej. PPO_2).")
    parser.add_argument("--c0", action="store_true", default=False,
                        help="Evaluar baseline analítico C0.")
    parser.add_argument("--kp", type=float, default=50.0,
                        help="Ganancia proporcional para C0 (por defecto: 50.0).")
    parser.add_argument("--kd", type=float, default=0.05,
                        help="Ganancia derivativa de amortiguamiento para C0 (por defecto: 0.05).")
    parser.add_argument("--lambda-damping", type=float, default=0.01,
                        help="Amortiguamiento Levenberg-Marquardt para C0 (por defecto: 0.01).")
    parser.add_argument("--vecnormalize", type=str, default=None,
                        help="Ruta opcional al archivo vecnormalize_stats.pkl.")
    parser.add_argument("--dataset", type=str, default="data/datasets/validation_set_200.json",
                        help="Ruta al JSON del conjunto de episodios de evaluación.")
    parser.add_argument("--env-config", type=str, default="configs/environment.yaml",
                        help="Ruta a la configuración física del entorno (SSOT).")
    parser.add_argument("--output-dir", type=str, default="reports/data/evaluation",
                        help="Directorio destino para los CSV y JSON de resultados.")
    parser.add_argument("--mode", type=str, default="deterministic",
                        choices=["deterministic", "stochastic"],
                        help="Modo de ejecución de la política.")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Regenerar automáticamente las figuras comparativas en reports/figures/.")
    args = parser.parse_args()

    # Resolver tipo de agente
    agent_type = "c0" if args.c0 or (args.agent and args.agent.lower() == "c0") else "rl"
    model_path = args.model_path or (args.agent if agent_type != "c0" else None)

    run_evaluation(
        agent_type=agent_type,
        model_path=model_path,
        vec_norm_path=args.vecnormalize,
        label=args.label,
        kp=args.kp,
        kd=args.kd,
        lambda_damping=args.lambda_damping,
        dataset_path=args.dataset,
        env_config_path=args.env_config,
        output_dir=args.output_dir,
        eval_mode=args.mode,
        generate_plots=args.plot
    )




if __name__ == "__main__":
    main()
