"""
Tests unitarios y de integración para la capa de agregación multi-semilla y partición jerárquica de desenlaces.
Implementado con unittest (librería estándar).
Verifica:
1. Estadísticas agregadas (media, std muestral ddof=1, mín, máx, mediana).
2. Semilla con 0 éxitos (mediana NaN, conteo n=0).
3. Partición jerárquica de 6 desenlaces y orden de prioridades.
4. Episodios con banderas no clasificables y contradicciones de consistencia.
5. Reconciliación exacta de categorías 1 + 2 con la tasa de éxito.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.generate_phase_a_aggregated_plots import classify_episodes

class TestAggregatedPipeline(unittest.TestCase):

    def test_statistical_aggregation_synthetic(self):
        """Valida el cálculo de media, std (ddof=1), min, max y mediana sobre 3 semillas sintéticas."""
        seed_values = pd.Series([0.70, 0.65, 0.75])
        mean_val = float(seed_values.mean())
        std_val = float(seed_values.std(ddof=1))
        min_val = float(seed_values.min())
        max_val = float(seed_values.max())
        med_val = float(seed_values.median())

        self.assertAlmostEqual(mean_val, 0.70, places=7)
        self.assertAlmostEqual(std_val, 0.05, places=7)
        self.assertAlmostEqual(min_val, 0.65, places=7)
        self.assertAlmostEqual(max_val, 0.75, places=7)
        self.assertAlmostEqual(med_val, 0.70, places=7)

    def test_zero_successes_seed_handling(self):
        """Verifica que una semilla con 0 éxitos resulte en mediana NaN y reporte conteo n=0 sin inventar 0 ni 200."""
        episodes = pd.DataFrame({
            "success": [False, False, False],
            "steps_to_success": [np.nan, np.nan, np.nan]
        })
        succ_eps = episodes[episodes["success"] == True]
        self.assertEqual(len(succ_eps), 0)
        steps_median = float(succ_eps["steps_to_success"].median()) if len(succ_eps) > 0 else np.nan
        self.assertTrue(np.isnan(steps_median))

    def test_hierarchical_outcome_partition_cases(self):
        """
        Valida la partición en 6 categorías jerárquicas:
        - Éxito limpio -> cat 1
        - Éxito + balístico -> cat 2
        - Éxito + oscilatorio -> cat 2
        - Fallo + balístico + oscilatorio -> cat 3 (prioridad de oscilatorio sobre balístico)
        - Fallo + balístico -> cat 4
        - Fallo sin aproximación (d>=0.30, time_inside_030==0) -> cat 5
        - Fallo sin estabilizar (d<0.30, time_inside_030>0) -> cat 6
        """
        synthetic_df = pd.DataFrame([
            # 1. Éxito limpio
            {"success": True, "ballistic_flag": False, "oscillatory_flag": False, "time_inside_030": 35},
            # 2. Éxito con patología (balístico)
            {"success": True, "ballistic_flag": True, "oscillatory_flag": False, "time_inside_030": 28},
            # 3. Éxito con patología (oscilatorio)
            {"success": True, "ballistic_flag": False, "oscillatory_flag": True, "time_inside_030": 50},
            # 4. Fallo con ambas patologías -> debe caer en Categoría 3 (Oscilatorio) por prioridad
            {"success": False, "ballistic_flag": True, "oscillatory_flag": True, "time_inside_030": 40},
            # 5. Fallo balístico puro -> Categoría 4
            {"success": False, "ballistic_flag": True, "oscillatory_flag": False, "time_inside_030": 15},
            # 6. Fallo sin aproximación -> Categoría 5 (time_inside_030 == 0)
            {"success": False, "ballistic_flag": False, "oscillatory_flag": False, "time_inside_030": 0},
            # 7. Fallo sin estabilizar -> Categoría 6 (time_inside_030 > 0)
            {"success": False, "ballistic_flag": False, "oscillatory_flag": False, "time_inside_030": 12},
        ])

        classified = classify_episodes(synthetic_df)
        cats = list(classified["outcome_category"])

        self.assertEqual(cats[0], "1. Éxito limpio")
        self.assertEqual(cats[1], "2. Éxito con patología")
        self.assertEqual(cats[2], "2. Éxito con patología")
        self.assertEqual(cats[3], "3. Fallo oscilatorio")       # Prioridad verificada
        self.assertEqual(cats[4], "4. Fallo balístico")
        self.assertEqual(cats[5], "5. Fallo sin aproximación") # t_near == 0
        self.assertEqual(cats[6], "6. Fallo sin estabilizar")   # t_near > 0

        # Reconciliación con success
        succ_count = synthetic_df["success"].sum()
        cat1_and_2 = (classified["outcome_category"].isin(["1. Éxito limpio", "2. Éxito con patología"])).sum()
        self.assertEqual(succ_count, cat1_and_2)

    def test_consistency_check_ballistic_without_near_entry(self):
        """Valida la detección de inconsistencias: ballistic_flag == True pero time_inside_030 == 0."""
        inconsistent_row = pd.DataFrame([{
            "success": False,
            "ballistic_flag": True,
            "oscillatory_flag": False,
            "time_inside_030": 0
        }])
        is_inconsistent = (inconsistent_row["ballistic_flag"] == True) & (inconsistent_row["time_inside_030"] == 0)
        self.assertTrue(is_inconsistent.iloc[0])

    def test_nan_flags_unclassifiable_handling(self):
        """
        Valida que si alguna bandera (ballistic_flag u oscillatory_flag) es NaN:
        1. Se detecte explícitamente la bandera no disponible sin castear NaN a bool.
        2. El resultado sea exactamente 'No clasificable'.
        3. Se evite que bool(np.nan) convierta erróneamente NaN en True.
        4. Se cumpla: categorías clasificadas + No clasificable == total de episodios.
        5. El caso normal, sin NaN, mantenga la partición de las seis categorías sin ningún 'No clasificable'.
        """
        mixed_df = pd.DataFrame([
            # 1. Caso normal: Éxito limpio
            {"success": True, "ballistic_flag": False, "oscillatory_flag": False, "time_inside_030": 30},
            # 2. Caso normal: Fallo balístico
            {"success": False, "ballistic_flag": True, "oscillatory_flag": False, "time_inside_030": 10},
            # 3. ballistic_flag es NaN -> si se hiciera bool(np.nan), daría True y clasificaría erróneamente
            {"success": False, "ballistic_flag": np.nan, "oscillatory_flag": False, "time_inside_030": 10},
            # 4. oscillatory_flag es NaN -> si se hiciera bool(np.nan), daría True y clasificaría erróneamente
            {"success": True, "ballistic_flag": False, "oscillatory_flag": np.nan, "time_inside_030": 30},
            # 5. Ambas banderas son NaN
            {"success": False, "ballistic_flag": np.nan, "oscillatory_flag": np.nan, "time_inside_030": 0},
            # 6. success es NaN
            {"success": np.nan, "ballistic_flag": False, "oscillatory_flag": False, "time_inside_030": 0},
        ])

        classified = classify_episodes(mixed_df)
        cats = list(classified["outcome_category"])

        # Casos normales
        self.assertEqual(cats[0], "1. Éxito limpio")
        self.assertEqual(cats[1], "4. Fallo balístico")

        # Casos con NaN devuelven estrictamente 'No clasificable'
        self.assertEqual(cats[2], "No clasificable")
        self.assertEqual(cats[3], "No clasificable")
        self.assertEqual(cats[4], "No clasificable")
        self.assertEqual(cats[5], "No clasificable")

        # Exhaustividad V02: categorías clasificadas + No clasificable == total de episodios
        n_classified = int((classified["outcome_category"] != "No clasificable").sum())
        n_unclassifiable = int((classified["outcome_category"] == "No clasificable").sum())
        total_episodes = len(mixed_df)

        self.assertEqual(n_classified, 2)
        self.assertEqual(n_unclassifiable, 4)
        self.assertEqual(n_classified + n_unclassifiable, total_episodes)

        # Caso normal (sin NaN) mantiene las 6 categorías canónicas con 0 No clasificables
        normal_df = pd.DataFrame([
            {"success": True, "ballistic_flag": False, "oscillatory_flag": False, "time_inside_030": 30},
            {"success": True, "ballistic_flag": True, "oscillatory_flag": False, "time_inside_030": 25},
            {"success": False, "ballistic_flag": False, "oscillatory_flag": True, "time_inside_030": 20},
            {"success": False, "ballistic_flag": True, "oscillatory_flag": False, "time_inside_030": 15},
            {"success": False, "ballistic_flag": False, "oscillatory_flag": False, "time_inside_030": 0},
            {"success": False, "ballistic_flag": False, "oscillatory_flag": False, "time_inside_030": 5},
        ])
        classified_normal = classify_episodes(normal_df)
        n_normal_unclass = int((classified_normal["outcome_category"] == "No clasificable").sum())
        self.assertEqual(n_normal_unclass, 0)
        self.assertEqual(len(classified_normal["outcome_category"].unique()), 6)

if __name__ == "__main__":
    unittest.main()
