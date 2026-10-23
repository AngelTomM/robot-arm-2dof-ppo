"""
Paquete principal para el control y aprendizaje por refuerzo de manipulador robótico planar 2-DOF (Fase D).
Exporta directamente las clases, controladores, métricas y utilidades del pipeline canónico.
"""
from .arm2d_env import Arm2DEnv
from .jacobian_controller import JacobianDampedController
from .callbacks import (
    LQRMetricsCallback,
    VecNormalizeSyncEvalCallback,
    CoupledCheckpointCallback,
)
from .config import load_config, load_env_config, resolve_path
from .evaluator import evaluate_on_dataset
from .metrics import (
    K_STAB,
    TARGET_RADIUS_EVAL,
    VEL_TOL_EVAL,
    W_POS_EVAL,
    W_VEL_EVAL,
    W_CTRL_EVAL,
    wilson_score_interval,
    compute_stabilization_metrics,
    compute_settling_time,
    compute_control_effort,
    compute_lqr_canonical_cost,
    compute_aggregate_metrics,
)
from .datasets import generate_dataset, save_and_get_datasets

__all__ = [
    "Arm2DEnv",
    "JacobianDampedController",
    "LQRMetricsCallback",
    "VecNormalizeSyncEvalCallback",
    "CoupledCheckpointCallback",
    "load_config",
    "load_env_config",
    "resolve_path",
    "evaluate_on_dataset",
    "generate_dataset",
    "save_and_get_datasets",
    "K_STAB",
    "TARGET_RADIUS_EVAL",
    "VEL_TOL_EVAL",
    "W_POS_EVAL",
    "W_VEL_EVAL",
    "W_CTRL_EVAL",
    "wilson_score_interval",
    "compute_stabilization_metrics",
    "compute_settling_time",
    "compute_control_effort",
    "compute_lqr_canonical_cost",
    "compute_aggregate_metrics",
]
