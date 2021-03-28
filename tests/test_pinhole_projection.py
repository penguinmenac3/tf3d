import unittest
import tf3d
import numpy as np


class TestPinholeProjection(unittest.TestCase):
    def setUp(self) -> None:
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        self.P_matrix = np.array([
            [721.5377, 0.0, 609.5593, 44.85728],
            [0.0, 721.5377, 172.854, 0.2163791],
            [0.0, 0.0, 1.0, 0.002745884]
        ])

    def test_create_pinhole_projection(self):
        solution = np.array([
            [2, 0, 100, 0],
            [0, 1, 200, 0],
            [0, 0, 1, 0],
        ])
        P_matrix = tf3d.pinhole_matrix(fx=2, fy=1, cx=100, cy=200)
        assert P_matrix.shape == (3, 4)
        assert np.allclose(P_matrix, solution, atol=1e-6), f"Matrix not same!\nFound:\n{P_matrix}\nShould be:\n{solution}"

    def test_apply_pinhole_projection(self):
        point = np.array([1, 0, 20])
        solution = np.array([647.79011117, 172.84108888, 20.02498439])

        points = tf3d.pinhole_project(self.P_matrix, point)
        assert points.shape == (3,)
        assert np.allclose(points, solution, atol=1e-6), f"Matrix not same!\nFound:\n{points}\nShould be:\n{solution}"

    def test_apply_pinhole_projection_multi(self):
        points = np.random.rand(50, 3) * 40 - 20
        solution = np.array([tf3d.pinhole_project(self.P_matrix, point) for point in points])

        points = tf3d.pinhole_project(self.P_matrix, points)
        assert points.shape == (50, 3)
        assert np.allclose(points, solution, atol=1e-6), f"Matrix not same!\nFound:\n{points}\nShould be:\n{solution}"

    def test_apply_pinhole_reprojection(self):
        original = np.array([1, 0, 20])
        projected = tf3d.pinhole_project(self.P_matrix, original)
        reprojected = tf3d.pinhole_reproject(self.P_matrix, projected)
        assert reprojected.shape == (3,)
        assert np.allclose(reprojected, original, atol=1e-6), f"Matrix not same!\nFound:\n{reprojected}\nShould be:\n{original}"

    def test_apply_pinhole_reprojection_multi(self):
        original = np.random.rand(4, 3) * 40 - 20
        projected = tf3d.pinhole_project(self.P_matrix, original)
        reprojected = tf3d.pinhole_reproject(self.P_matrix, projected)
        assert reprojected.shape == (4, 3)
        assert np.allclose(reprojected, original, atol=1e-6), f"Matrix not same!\nFound:\n{reprojected}\nShould be:\n{original}\nError:\b{reprojected-original}"


if __name__ == "__main__":
    unittest.main()
