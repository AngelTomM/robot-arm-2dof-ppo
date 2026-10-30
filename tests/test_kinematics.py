"""
Pruebas Unitarias de Cinemática Directa y Matriz Jacobiana:
Validación Analítica contrastada contra el Motor de Física MuJoCo (Reacher-v5).
"""
import math
import unittest
import numpy as np
import mujoco

from src import Arm2DEnv


class TestKinematicsMuJoCo(unittest.TestCase):
    def setUp(self):
        self.env = Arm2DEnv()

    def tearDown(self):
        self.env.close()

    def test_forward_kinematics_vs_mujoco(self):
        """Contrasta la cinemática directa planar analítica contra las posiciones 3D de MuJoCo."""
        test_angles = [
            (0.0, 0.0),
            (math.pi / 4, 0.0),
            (math.pi / 2, 0.0),
            (-math.pi / 3, math.pi / 4),
            (1.2, -1.0),
            (-2.0, 1.5),
        ]

        for th1, th2 in test_angles:
            # 1. Cinemática analítica
            p_elbow_ana, p_ee_ana = self.env._forward_kinematics(th1, th2)

            # 2. Posición directa en MuJoCo
            qpos = np.array([th1, th2, 0.1, 0.1], dtype=np.float64)
            qvel = np.zeros(4, dtype=np.float64)
            self.env.set_state(qpos, qvel)
            mujoco.mj_forward(self.env.model, self.env.data)

            p_ee_mujoco = self.env.reacher_env.unwrapped.get_body_com("fingertip")[:2]
            p_elbow_mujoco = self.env.reacher_env.unwrapped.get_body_com("body1")[:2]

            np.testing.assert_allclose(p_ee_ana, p_ee_mujoco, atol=1e-4,
                                       err_msg=f"Discrepancia en efector final para angles ({th1}, {th2})")
            np.testing.assert_allclose(p_elbow_ana, p_elbow_mujoco, atol=1e-4,
                                       err_msg=f"Discrepancia en codo para angles ({th1}, {th2})")

    def test_jacobian_analytical_vs_mujoco(self):
        """Contrasta la matriz Jacobiana analítica contra mj_jacBody nativo de MuJoCo."""
        test_angles = [
            (0.0, 0.0),
            (0.5, -0.8),
            (-1.2, 0.5),
            (math.pi / 3, -math.pi / 4),
            (-2.1, 1.8),
        ]

        fingertip_body_id = self.env.model.body("fingertip").id

        for th1, th2 in test_angles:
            # 1. Jacobiano analítico 2x2
            J_ana = self.env._compute_jacobian(th1, th2)

            # 2. Jacobiano cinemático de cuerpo en MuJoCo (3 x nv)
            qpos = np.array([th1, th2, 0.1, 0.1], dtype=np.float64)
            qvel = np.zeros(4, dtype=np.float64)
            self.env.set_state(qpos, qvel)
            mujoco.mj_forward(self.env.model, self.env.data)

            jacp = np.zeros((3, self.env.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(self.env.model, self.env.data, jacp, None, fingertip_body_id)
            J_mujoco = jacp[:2, :2]

            np.testing.assert_allclose(J_ana, J_mujoco, atol=1e-5,
                                       err_msg=f"Discrepancia Jacobiano en angles ({th1}, {th2})")

    def test_observation_vector_consistency(self):
        """Verifica la estructura y coherencia del vector de observación de 10D de Reacher-v5."""
        obs, info = self.env.reset(seed=42)

        self.assertEqual(obs.shape, (10,))
        th1 = float(self.env.data.qpos[0])
        th2 = float(self.env.data.qpos[1])
        qvel = self.env.data.qvel[:2]

        # Componentes trigonométricas de articulaciones
        np.testing.assert_allclose(obs[0], math.cos(th1), atol=1e-6)
        np.testing.assert_allclose(obs[1], math.cos(th2), atol=1e-6)
        np.testing.assert_allclose(obs[2], math.sin(th1), atol=1e-6)
        np.testing.assert_allclose(obs[3], math.sin(th2), atol=1e-6)

        # Coordenadas objetivo
        np.testing.assert_allclose(obs[4:6], self.env.data.qpos[2:4], atol=1e-6)

        # Velocidades articulares
        np.testing.assert_allclose(obs[6:8], qvel, atol=1e-6)

        # Vector cartesiano relativo (p_ee - p_tgt)
        p_ee = self.env.reacher_env.unwrapped.get_body_com("fingertip")[:2]
        p_tgt = self.env.reacher_env.unwrapped.get_body_com("target")[:2]
        np.testing.assert_allclose(obs[8:10], p_ee - p_tgt, atol=1e-6)

        # Verificación de distancia euclidiana en info
        np.testing.assert_almost_equal(info["distance"], np.linalg.norm(obs[8:10]), decimal=6)

    def test_mujoco_joint_limits(self):
        """Verifica que el rango articular de joint1 en el modelo MuJoCo coincida con [-3.0, 3.0] rad."""
        joint1_range = self.env.model.jnt_range[1]
        np.testing.assert_allclose(joint1_range, [-3.0, 3.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
