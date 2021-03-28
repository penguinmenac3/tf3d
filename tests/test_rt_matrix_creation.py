import unittest
import tf3d
import numpy as np
from pyquaternion import Quaternion


class TestRTMatrix(unittest.TestCase):
    def setUp(self) -> None:
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    def test_from_quaternion(self):
        q = Quaternion(axis=[1, 0, 0], angle=3.14159265)
        solution = q.transformation_matrix
        Rt_matrix = tf3d.Rt_from_quaternion(q)
        assert Rt_matrix.shape == (4, 4)
        assert (Rt_matrix == solution).all(), Rt_matrix


    def test_from_axis_angle(self):
        q = Quaternion(axis=[1, 0, 0], angle=3.14159265)
        solution = q.transformation_matrix
        Rt = tf3d.Rt_from_axis_angle(axis=[1, 0, 0], angle=3.14159265)
        assert Rt.shape == (4, 4)
        assert np.allclose(Rt, solution), f"Matrix not same!\nFound:\n{Rt}\nShould be:\n{solution}"

    def test_from_offset(self):
        offset = np.array([1, 2, 3])
        solution = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ])
        Rt = tf3d.Rt_from_offset(offset=offset)
        assert Rt.shape == (4, 4)
        assert np.allclose(Rt, solution), f"Matrix not same!\nFound:\n{Rt}\nShould be:\n{solution}"

    def test_R_from_RT(self):
        q = Quaternion(axis=[1, 0, 0], angle=3.14159265/2)
        solution = q.rotation_matrix

        Rt = tf3d.Rt_from_quaternion(q)
        R = tf3d.R_from_Rt(Rt)

        assert R.shape == (3, 3)
        assert np.allclose(R, solution, atol=1e-6), f"Matrix not same!\nFound:\n{Rt}\nShould be:\n{solution}"

    def test_t_from_RT(self):
        offset = np.array([1, 2, 3])
        Rt = tf3d.Rt_from_offset(offset)
        t = tf3d.t_from_Rt(Rt)
        
        assert t.shape == (3,)
        assert np.allclose(t, offset, atol=1e-6), f"Matrix not same!\nFound:\n{Rt}\nShould be:\n{offset}"


if __name__ == "__main__":
    unittest.main()
