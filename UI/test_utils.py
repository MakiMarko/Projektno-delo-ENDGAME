import unittest
from utils import combine_boxes, intersection_over_union

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

if __name__ == '__main__':
    unittest.main()
