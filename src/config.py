"""
Gestor de Carga y Desacoplamiento de Configuraciones YAML.

Permite separar la especificación de los parámetros invariantes del entorno físico/MDP
de los hiperparámetros particulares de cada agente (PPO, SAC, C0).
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_CONFIG = REPO_ROOT / "configs" / "environment.yaml"


def resolve_path(path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """Resuelve una ruta relativa respecto al directorio base o a la raíz del repositorio."""
    p = Path(path)
    if p.is_absolute():
        return p
    if base_dir and (base_dir / p).exists():
        return (base_dir / p).resolve()
    if (REPO_ROOT / p).exists():
        return (REPO_ROOT / p).resolve()
    # Si base_dir fue provisto, devolver relativo a este aunque no exista aún
    return ((base_dir / p) if base_dir else (REPO_ROOT / p)).resolve()


def load_env_config(env_config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Carga el archivo YAML de configuración del entorno.
    Garantiza retornar un diccionario estructurado bajo la clave 'env'.
    """
    target_path = resolve_path(env_config_path, REPO_ROOT) if env_config_path else DEFAULT_ENV_CONFIG
    if not target_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración del entorno: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "env" not in data:
        data = {"env": data}

    data["env_config_resolved"] = str(target_path)
    return data


def load_config(config_path: Union[str, Path],
                env_config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Carga la configuración completa del experimento fusionando el archivo de entorno común
    con la configuración específica del agente.

    Parámetros:
      config_path: Ruta al archivo YAML del agente (ej. configs/phase_D/prueba/PPO_1.yaml).
      env_config_path: Ruta opcional para forzar un archivo de entorno distinto.
    """
    cfg_file = resolve_path(config_path, REPO_ROOT)
    if not cfg_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración del agente: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Determinar ruta del archivo de entorno
    explicit_env_path = env_config_path or cfg.get("env_config")
    resolved_env_path = None
    if explicit_env_path:
        resolved_env_path = resolve_path(explicit_env_path, cfg_file.parent)
    elif "env" not in cfg:
        # Si no define env ni env_config, usar el entorno común por defecto
        resolved_env_path = DEFAULT_ENV_CONFIG

    if resolved_env_path and resolved_env_path.exists():
        env_dict = load_env_config(resolved_env_path)
        base_env = env_dict.get("env", {})
        agent_env_override = cfg.get("env", {})
        # Fusión preservando overrides explícitos del agente si existieran
        merged_env = {**base_env, **agent_env_override}
        cfg["env"] = merged_env
        cfg["env_config_resolved"] = env_dict.get("env_config_resolved", str(resolved_env_path))
    elif "env" not in cfg:
        # Fallback de seguridad cargando el entorno por defecto
        env_dict = load_env_config()
        cfg["env"] = env_dict.get("env", {})
        cfg["env_config_resolved"] = env_dict.get("env_config_resolved")

    return cfg
