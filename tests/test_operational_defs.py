"""
Pruebas Unitarias de Definiciones Operativas de Llegada Balística y Oscilación (Sección 9.6 y 15).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np


def classify_trajectory(joint_speed_at_first_entry, n_exits, n_cross,
                        u_b=0.3, n_o=2, n_c=3):
    """
    Clasifica un episodio según las definiciones operativas fijadas en 9.6:
    - Llegada balística: joint_speed_at_first_entry > U_b
    - Oscilatorio: n_exits >= N_o O n_cross >= N_c
    """
    is_ballistic = (joint_speed_at_first_entry is not None) and (joint_speed_at_first_entry > u_b)
    is_oscillatory = (n_exits >= n_o) or (n_cross >= n_c)
    return is_ballistic, is_oscillatory


class TestOperationalDefs(unittest.TestCase):
    def test_ballistic_entry_classification(self):
        # Caso 1: Entrada suave a 0.08 rad/s <= 0.3 -> No balística
        ballistic, oscillatory = classify_trajectory(joint_speed_at_first_entry=0.08, n_exits=0, n_cross=0)
        self.assertFalse(ballistic)
        self.assertFalse(oscillatory)

        # Caso 2: Entrada rápida a 1.2 rad/s > 0.3 -> Balística
        ballistic, oscillatory = classify_trajectory(joint_speed_at_first_entry=1.2, n_exits=0, n_cross=0)
        self.assertTrue(ballistic)
        self.assertFalse(oscillatory)

    def test_oscillatory_classification(self):
        # Caso 1: Sobrepaso múltiple (n_exits = 2 >= 2) -> Oscilatorio
        ballistic, oscillatory = classify_trajectory(joint_speed_at_first_entry=0.1, n_exits=2, n_cross=0)
        self.assertFalse(ballistic)
        self.assertTrue(oscillatory)

        # Caso 2: Cruces repetidos de vector de error (n_cross = 3 >= 3) -> Oscilatorio
        ballistic, oscillatory = classify_trajectory(joint_speed_at_first_entry=0.1, n_exits=0, n_cross=3)
        self.assertFalse(ballistic)
        self.assertTrue(oscillatory)

        # Caso 3: Entrada limpia y frenado inmediato -> Estable (ni balística ni oscilatoria)
        ballistic, oscillatory = classify_trajectory(joint_speed_at_first_entry=0.05, n_exits=0, n_cross=0)
        self.assertFalse(ballistic)
        self.assertFalse(oscillatory)


if __name__ == "__main__":
    unittest.main()
