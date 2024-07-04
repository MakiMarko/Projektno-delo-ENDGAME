import unittest
import numpy as np
import cv2
from UI.utils import combine_boxes, intersection_over_union
from URG.projektna1 import Triangle, update_triangle_pos_plane
from URG.projektna2 import preprocess_image, find_contours, grabcut_segmentation, graham_scan, resize_image
# Dummy data for tests
pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

class TestUtils(unittest.TestCase):
    def test_combine_boxes(self):
        boxA = [1, 1, 3, 3]
        boxB = [2, 2, 4, 4]
        expected = (1, 1, 4, 4)
        result = combine_boxes(boxA, boxB)
        self.assertEqual(result, expected)

        boxA = [0, 0, 1, 1]
        boxB = [1, 1, 2, 2]
        expected = (0, 0, 2, 2)
        result = combine_boxes(boxA, boxB)
        self.assertEqual(result, expected)

    def test_intersection_over_union(self):
        boxA = [1, 1, 3, 3]
        boxB = [2, 2, 4, 4]
        expected = 1 / 7  # Calculated IoU for these boxes
        result = intersection_over_union(boxA, boxB)
        self.assertAlmostEqual(result, expected, places=5)

        boxA = [0, 0, 1, 1]
        boxB = [1, 1, 2, 2]
        expected = 0.0  # No overlap
        result = intersection_over_union(boxA, boxB)
        self.assertAlmostEqual(result, expected, places=5)

        boxA = [0, 0, 2, 2]
        boxB = [1, 1, 3, 3]
        expected = 1 / 7  # Calculated IoU for these boxes
        result = intersection_over_union(boxA, boxB)
        self.assertAlmostEqual(result, expected, places=5)

    def test_resize_image(self):
        img = np.ones((800, 600, 3), dtype=np.uint8)
        resized = resize_image(img, 640)
        self.assertEqual(resized.shape, (640, 640, 3))

    def test_preprocess_image(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        processed = preprocess_image(img)
        self.assertEqual(processed.shape, (100, 100))

    def test_find_contours(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (25, 25), (75, 75), (255, 255, 255), -1)
        contours = find_contours(img)
        self.assertTrue(len(contours) > 0)

    def test_grabcut_segmentation(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (25, 25), (75, 75), (255, 255, 255), -1)
        rect = (25, 25, 50, 50)
        segmented = grabcut_segmentation(img, rect)
        self.assertEqual(segmented.shape, (100, 100, 3))

    def test_graham_scan(self):
        points = [(0, 0), (1, 1), (0, 1), (1, 0), (0.5, 0.5)]
        hull = graham_scan(points)
        expected_hull = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.assertEqual(hull, expected_hull)

class TestTriangleMethods(unittest.TestCase):
    def setUp(self):
        self.triangle = Triangle(0, 1, 2)

    def test_calculate_vector(self):
        expected = np.array([0, 0, 1])
        np.testing.assert_array_almost_equal(self.triangle.calculate_vector(), expected)

    def test_calculate_center(self):
        expected = np.array([1/3, 1/3, 0])
        np.testing.assert_array_almost_equal(self.triangle.calculate_center(), expected)

    def test_equal_triangles(self):
        other = Triangle(0, 1, 2)
        self.assertTrue(self.triangle.equal_triangles(other))
        different = Triangle(1, 2, 3)
        self.assertFalse(self.triangle.equal_triangles(different))

    def test_triangle_hash(self):
        self.assertEqual(self.triangle.triangle_hash(), hash((0, 1, 2)))

class TestQuickHullFunctions(unittest.TestCase):
    def test_update_triangle_pos_plane(self):
        triangles = [Triangle(0, 1, 2), Triangle(1, 2, 3)]
        update_triangle_pos_plane(triangles, pts)
        # Check if pos_plane is populated correctly
        self.assertTrue(len(triangles[0].pos_plane) > 0)
        self.assertTrue(len(triangles[1].pos_plane) > 0)

if __name__ == '__main__':
    unittest.main()
