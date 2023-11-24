import unittest
from common import *


class TestCommon(unittest.TestCase):

    def test_vector3(self):
        v = Vector3(np.array([1, 2, 3]))
        self.assertEqual(v[0], 1)
        self.assertEqual(v[1], 2)
        self.assertEqual(v[2], 3)

    def test_vector3_normalized(self):
        v = Vector3.normalized(1, 2, 3)
        self.assertAlmostEqual(v[0], 0.26726124)
        self.assertAlmostEqual(v[1], 0.53452248)
        self.assertAlmostEqual(v[2], 0.80178373)

    def test_vector3_normalized2(self):
        v = Vector3.normalized(1, 2, 3)
        self.assertAlmostEqual(np.linalg.norm(v), 1)

    def test_plane(self):
        p = Plane(np.array([1, 2, 3, 4]))
        self.assertEqual(p[0], 1)
        self.assertEqual(p[1], 2)
        self.assertEqual(p[2], 3)
        self.assertEqual(p[3], 4)

    def test_plane_normalized(self):
        p = Plane.normalized(1, 2, 3, 4)
        self.assertAlmostEqual(p[0], 0.26726124)
        self.assertAlmostEqual(p[1], 0.53452248)
        self.assertAlmostEqual(p[2], 0.80178373)
        self.assertAlmostEqual(p[3], 1.069045)

    def test_plane_norm(self):
        p = Plane.normalized(1, 2, 3, 4)
        self.assertAlmostEqual(np.linalg.norm(p.norm()), 1)

    def test_plane_distance_to_points(self):
        p = Plane.normalized(0, 0, 1, 0)
        points = np.array([[1, 2, 3], [4, 5, 6]])
        result = p.distance_to_points(points)
        self.assertAlmostEqual(result[0], 3)
        self.assertAlmostEqual(result[1], 6)

    def test_plane_distance_to_points2(self):
        p = Plane.normalized(0, 1, 0, 0)
        points = np.array([[1, 2, 3], [4, 5, 6]])
        result = p.distance_to_points(points)
        self.assertAlmostEqual(result[0], 2)
        self.assertAlmostEqual(result[1], 5)

    def test_plane_distance_to_points3(self):
        p = Plane.normalized(1, 0, 0, 0)
        points = np.array([[1, 2, 3], [4, 5, 6]])
        result = p.distance_to_points(points)
        self.assertAlmostEqual(result[0], 1)
        self.assertAlmostEqual(result[1], 4)

    def test_camera_intrinsics(self):
        intrinsics = CameraIntrinsics(1, 2, 3, 4)
        self.assertEqual(intrinsics.fx, 1)
        self.assertEqual(intrinsics.fy, 2)
        self.assertEqual(intrinsics.cx, 3)
        self.assertEqual(intrinsics.cy, 4)


if __name__ == '__main__':
    unittest.main()
