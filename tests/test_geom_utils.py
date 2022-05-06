from flexa._geom_utils import *

import unittest

import numpy as np

class Test_geom_utils(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1, 0, 0])
        self.y = np.array([0, 1, 0])
        self.z = np.array([0, 0, 1])

    def assertArrayEqual(self, actual, expected, rtol=1e-05, atol=1e-08):
        self.assertTrue(
            np.all(np.isclose(actual, expected, rtol=rtol, atol=atol)))

    def test_angle(self):
        self.assertEqual(angle(self.x, self.x), 0)
        self.assertEqual(angle(self.x, self.y), np.pi / 2)

    def test_align(self):
        ref = self.z

        a = self.x + self.y + self.z
        self.assertArrayEqual(align(a, self.z), a)

        self.assertArrayEqual(align(self.y, ref), self.y)

        a = -1 * a
        self.assertTrue(np.dot(a, ref) < 0)
        self.assertArrayEqual(align(a, ref), a)
    
    def test_tri_normal(self):
        ref = self.z
        a = np.array([0, 1])
        b = np.array([1, 0])
        c = np.array([1, 1])
        self.assertArrayEqual(tri_normal(a, b, c, ref), ref)

        self.assertArrayEqual(tri_normal(np.vstack((a, b, c))), ref)

        n = (self.x + self.y + self.z) / np.sqrt(3)
        self.assertArrayEqual(tri_normal(self.y, self.x, self.z, ref), n)
        
        ref = -self.z
        self.assertArrayEqual(
            tri_normal(np.vstack((self.y, self.x, self.z)), ref=ref), -1 * n)

    def test_face_normal(self):
        n_points = 8 # must be even
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        z = np.where(np.arange(n_points) % 2 == 0, 1, -1)
        r = np.vstack((np.cos(angles), np.sin(angles), z)).T

        ref = self.x + self.y + self.z
        # make sure pass to tri_normal is working
        self.assertArrayEqual(tri_normal(self.y, self.x, self.z, ref), 
            ref / np.linalg.norm(ref))

        expected = self.z
        self.assertArrayEqual(face_normal(r, ref), expected, 
            atol=1e-10)

        self.assertArrayEqual(face_normal(r, -ref), -expected, 
            atol=1e-10)

    def test_closest_point(self):
        closest = np.repeat(1, 3) / 3 # on the plane x + y + z = 1
        p = np.repeat(8, 3)
        
        self.assertArrayEqual(closest_point(self.x, self.y, self.z, p), closest)

if __name__ == '__main__':
    unittest.main()