# -*- coding: utf-8 -*-
"""Pruebas unitarias para el parseo, resolución y calificación por sección de modelos en plot.py."""

import pytest
import pandas as pd
from plot import parse_model_identifier, get_model_section, resolve_model_in_partition


def test_parse_model_identifier_compound():
    """Verifica la separación de sección y nombre de modelo con barras normales o invertidas."""
    name, sec = parse_model_identifier("PPO_eval_500k/PPO_7")
    assert name == "PPO_7"
    assert sec == "PPO_eval_500k"

    name_win, sec_win = parse_model_identifier("entorno\\PPO_3")
    assert name_win == "PPO_3"
    assert sec_win == "entorno"


def test_parse_model_identifier_simple():
    """Verifica el comportamiento con identificadores simples y sección por defecto."""
    name, sec = parse_model_identifier("PPO_7")
    assert name == "PPO_7"
    assert sec is None

    name_def, sec_def = parse_model_identifier("PPO_7", default_section="entorno")
    assert name_def == "PPO_7"
    assert sec_def == "entorno"


def test_get_model_section_resolution():
    """Verifica la inferencia jerárquica de la sección a partir del identificador o configuraciones."""
    # Identificador calificado
    assert get_model_section("PPO_eval_500k/PPO_7") == "PPO_eval_500k"
    # Argumento explícito
    assert get_model_section("PPO_7", explicit_section="SAC_eval_300k") == "SAC_eval_300k"
    # Inferencia vía configs
    assert get_model_section("PPO_17") == "PPO_eval_500k"
    # Inferencia con sufijo de semilla (con o sin guion)
    assert get_model_section("PPO_NOM_seed_42") == "PPO_eval_500k"
    assert get_model_section("PPO_NOM_seed42") == "PPO_eval_500k"


def test_resolve_model_in_partition():
    """Verifica la resolución exacta, desambiguación y control de excepciones en la partición."""
    sample_df1 = pd.DataFrame({"agent": ["PPO_7"], "e_ss": [0.01]})
    sample_df2 = pd.DataFrame({"agent": ["PPO_7"], "e_ss": [0.02]})
    sample_df3 = pd.DataFrame({"agent": ["PPO_17"], "e_ss": [0.005]})

    part_dict = {
        "entorno/PPO_7": sample_df1,
        "PPO_eval_500k/PPO_7": sample_df2,
        "PPO_eval_500k/PPO_17": sample_df3,
        "PPO_17": sample_df3,
    }

    # 1. Búsqueda calificada inequívoca
    name, sec, df = resolve_model_in_partition(part_dict, "entorno/PPO_7")
    assert name == "PPO_7"
    assert sec == "entorno"
    assert df["e_ss"].iloc[0] == 0.01

    name, sec, df = resolve_model_in_partition(part_dict, "PPO_eval_500k/PPO_7")
    assert name == "PPO_7"
    assert sec == "PPO_eval_500k"
    assert df["e_ss"].iloc[0] == 0.02

    # 2. Búsqueda simple con flag de sección
    name, sec, df = resolve_model_in_partition(part_dict, "PPO_7", default_section="entorno")
    assert name == "PPO_7"
    assert sec == "entorno"

    # 3. Detección de ambigüedad si se consulta PPO_7 sin especificar sección
    with pytest.raises(ValueError) as excinfo:
        resolve_model_in_partition(part_dict, "PPO_7")
    assert "Ambigüedad detectada" in str(excinfo.value)
    assert "entorno" in str(excinfo.value)
    assert "PPO_eval_500k" in str(excinfo.value)

    # 4. Búsqueda inequívoca de modelo único
    name, sec, df = resolve_model_in_partition(part_dict, "PPO_17")
    assert name == "PPO_17"
    assert sec == "PPO_eval_500k"

    # 5. Modelo inexistente lanza KeyError
    with pytest.raises(KeyError):
        resolve_model_in_partition(part_dict, "MODELO_INEXISTENTE")



def test_extract_model_and_seed_from_path():
    """Verifica la deducción automática del nombre del modelo y su semilla a partir de la ruta."""
    from pathlib import Path
    from evaluate import extract_model_and_seed_from_path

    # Estructura canónica con carpeta seed_*
    p1 = Path("experiments/PPO_eval_500k/PPO_NOM/seed_42/models/best_model/best_model.zip")
    name1, seed1 = extract_model_and_seed_from_path(p1)
    assert name1 == "PPO_NOM"
    assert seed1 == "42"

    p2 = Path("experiments/entorno/PPO_1/seed_100/best_model.zip")
    name2, seed2 = extract_model_and_seed_from_path(p2)
    assert name2 == "PPO_1"
    assert seed2 == "100"

    # Ruta sin seed_*
    p3 = Path("models/custom_policy.zip")
    name3, seed3 = extract_model_and_seed_from_path(p3)
    assert name3 == "custom_policy"
    assert seed3 is None
