import unittest
import tf3d
import numpy as np
from pyquaternion import Quaternion


class TestRTMatrix(unittest.TestCase):
    def setUp(self) -> None:
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        self.Rt = tf3d.Rt_from_quaternion(Quaternion(axis=[1, 0, 0], angle=3.14159265/2))

    def test_inverse(self):
        inv = tf3d.Rt_inverse(self.Rt)
        solution = np.linalg.inv(self.Rt)
        assert inv.shape == (4, 4)
        assert np.allclose(inv, solution, atol=1e-6), f"Matrix not same!\nFound:\n{self.Rt}\nShould be:\n{solution}"

    def test_transform_point(self):
        point = np.array([1, 2, 3])
        solution = np.array([1, -3, 2])
        point = tf3d.transform(self.Rt, point)

        assert point.shape == (3,)
        assert np.allclose(point, solution, atol=1e-6), f"Matrix not same!\nFound:\n{self.Rt}\nShould be:\n{solution}"

    def test_transform_multiple_points(self):
        points = np.random.rand(50, 3)
        solution = np.array([tf3d.transform(self.Rt, point) for point in points])
        points = tf3d.transform(self.Rt, points)

        assert points.shape == (50, 3)
        assert np.allclose(points, solution, atol=1e-6), f"Matrix not same!\nFound:\n{self.Rt}\nShould be:\n{solution}"


if __name__ == "__main__":
    unittest.main()
