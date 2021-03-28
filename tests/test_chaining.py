import unittest
import tf3d
import numpy as np
from pyquaternion import Quaternion


class TestRTMatrix(unittest.TestCase):
    def setUp(self) -> None:
        np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})

    def test_chaining_offsets(self):
        offset = np.array([1, 2, 3])
        a = tf3d.Rt_from_offset(offset=offset)
        solution = np.array([
            [1, 0, 0, 4],
            [0, 1, 0, 8],
            [0, 0, 1, 12],
            [0, 0, 0, 1]
        ])
        Rt = tf3d.chain(a, a, a, a)
        assert Rt.shape == (4, 4)
        assert np.allclose(Rt, solution, atol=1e-6), f"Matrix not same!\nFound:\n{Rt}\nShould be:\n{solution}\nError:\n{Rt - solution}"

    def test_chaining_rotations(self):
        q_1 = Quaternion(axis=[1, 0, 0], angle=3.14159265/2)
        q_2 = Quaternion(axis=[0, 1, 0], angle=3.14159265/2)
        solution = tf3d.Rt_from_quaternion(q_2 * q_1)
        Rt_1 = tf3d.Rt_from_quaternion(q_1)
        Rt_2 = tf3d.Rt_from_quaternion(q_2)
        Rt = tf3d.chain(Rt_1, Rt_2)
        assert Rt.shape == (4, 4)
        assert np.allclose(Rt, solution, atol=1e-6), f"Matrix not same!\nFound:\n{Rt}\nShould be:\n{solution}\nError:\n{Rt - solution}"

    def test_chaining_rotation_and_projection(self):
        solution = np.array([
            [2, 100, 0, 302],
            [0, 200, -1, 602],
            [0, 1, 0, 3]
        ], dtype=np.float32)
        P_matrix = tf3d.pinhole_matrix(fx=2, fy=1, cx=100, cy=200)
        Rt = tf3d.Rt_from_quaternion(
            q=Quaternion(axis=[1, 0, 0], angle=3.14159265/2),
            t=np.array([1, 2, 3])
        )
        Rt = tf3d.chain(Rt, P_matrix)
        assert Rt.shape == (3, 4)
        assert np.allclose(Rt, solution, atol=1e-6), f"Matrix not same!\nFound:\n{Rt}\nShould be:\n{solution}\nError:\n{Rt - solution}"

if __name__ == "__main__":
    unittest.main()
