"""Pruebas unitarias para el subsistema de resolución y mapeo paramétrico de métricas en plot.py."""

import pytest
from plot import CANONICAL_METRICS, METRIC_ALIASES, METRIC_MACROS, resolve_requested_metrics


def test_default_metric_resolution():
    """Verifica que sin argumento se resuelvan las 9 métricas estándar de evaluación."""
    res = resolve_requested_metrics(None)
    assert len(res) == 9
    assert "success_rate" in res
    assert "steady_state_error" in res
    assert "temporal_trajectories" in res


def test_individual_metric_resolution():
    """Verifica la resolución exacta y por alias de métricas individuales."""
    assert resolve_requested_metrics("explained_variance") == ["explained_variance"]
    assert resolve_requested_metrics("var_exp") == ["explained_variance"]
    assert resolve_requested_metrics("ev") == ["explained_variance"]
    assert resolve_requested_metrics("train/explained_variance") == ["explained_variance"]

    assert resolve_requested_metrics("approx_kl") == ["approx_kl"]
    assert resolve_requested_metrics("kl") == ["approx_kl"]
    assert resolve_requested_metrics("train/approx_kl") == ["approx_kl"]

    assert resolve_requested_metrics("steady_state_error") == ["steady_state_error"]
    assert resolve_requested_metrics("e_ss") == ["steady_state_error"]


def test_multiple_and_composite_metrics():
    """Verifica el parseo de listas combinadas separadas por coma sin duplicados."""
    res = resolve_requested_metrics("explained_variance,approx_kl,e_ss")
    assert res == ["explained_variance", "approx_kl", "steady_state_error"]

    res_macro = resolve_requested_metrics("optimizer")
    assert "loss" in res_macro
    assert "policy_loss" in res_macro
    assert "value_loss" in res_macro
    assert "entropy" in res_macro
    assert "approx_kl" in res_macro
    assert "explained_variance" in res_macro
    assert "optimizer_summary" in res_macro


def test_all_macro_resolution():
    """Verifica que la macro 'all' expanda todas las métricas de evaluación y entrenamiento (18 en total)."""
    res = resolve_requested_metrics("all")
    assert len(res) == 18
    assert "success_rate" in res
    assert "explained_variance" in res
    assert "learning_return" in res


def test_invalid_metric_raises_value_error():
    """Verifica que una métrica inexistente lance un ValueError con la lista de opciones disponibles."""
    with pytest.raises(ValueError) as excinfo:
        resolve_requested_metrics("metrica_inexistente_xyz")
    assert "Métrica desconocida" in str(excinfo.value)


def test_canonical_metrics_metadata_integrity():
    """Verifica que todas las métricas canónicas posean metadatos esenciales consistentes."""
    for key, meta in CANONICAL_METRICS.items():
        assert "category" in meta
        assert "filename" in meta
        assert "model_filename" in meta
        if meta["category"] == "optimizer":
            assert "tag" in meta
            assert meta["tag"].startswith("train/")
            assert "ylabel" in meta
