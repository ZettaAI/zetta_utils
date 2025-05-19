import numpy as np
import pytest

from zetta_utils.geometry.mask_center import center_pole, centroid, interior_point


class TestMaskCenter:
    def test_centroid_simple_square(self):
        # Test with a simple 3x3 square mask
        mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        result = centroid(mask)
        np.testing.assert_almost_equal(result, np.array([1.0, 1.0]))

    def test_centroid_rectangle(self):
        # Test with a rectangular mask
        mask = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
        result = centroid(mask)
        np.testing.assert_almost_equal(result, np.array([0.5, 1.5]))

    def test_centroid_irregular_shape(self):
        # Test with an irregular shape
        mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        result = centroid(mask)
        np.testing.assert_almost_equal(result, np.array([1.0, 1.0]))

    def test_centroid_empty_mask(self):
        # Test with an empty mask (should return an empty array or NaN values)
        mask = np.zeros((3, 3), dtype=int)
        result = centroid(mask)
        # Check if result contains NaN (which would happen if dividing by zero)
        assert np.isnan(result).all() or len(result) == 0

    def test_center_pole_square(self):
        # Test center_pole with a square
        mask = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        result = center_pole(mask)
        assert result in ((1, 1), (1, 2), (2, 1), (2, 2))

    def test_center_pole_rectangle(self):
        # Test with a rectangular mask (with a border of zeros).
        # Note that with this shape, there are a whole bunch of points that are
        # equally inaccessible (1 away from the border in the X dimension).
        # It is undefined which will be returned.  But a unit test requires
        # that we pick one, so for now, the one below is correct.
        mask = np.zeros((5, 17), dtype=int)
        mask[1:4, 1:16] = 1
        result = center_pole(mask)
        print(f"result: {result}")
        assert result == (2, 2)

    def test_center_pole_donut(self):
        # Test with a donut shape (center pole should be on the ring)
        mask = np.array(
            [[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]]
        )
        result = center_pole(mask)
        # The center pole should be one of the points on the ring
        valid_points = [(0, 2), (1, 1), (1, 3), (2, 0), (2, 4), (3, 1), (3, 3), (4, 2)]
        assert result in valid_points

    def test_center_pole_empty_mask(self):
        # Test with an empty mask (should raise error)
        mask = np.zeros((3, 3), dtype=int)
        with pytest.raises(Exception):  # Expecting an exception due to empty array
            center_pole(mask)

    def test_interior_point_centroid_inside(self):
        # Test when centroid is inside the mask
        mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        result = interior_point(mask)
        assert result == (1, 1)

    def test_interior_point_centroid_outside(self):
        # Test when centroid is outside the mask (donut shape)
        mask = np.array(
            [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
        )
        result = interior_point(mask)
        # Should use center_pole since centroid is not in the mask
        # One of these points should be the result
        valid_points = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 4),
            (2, 0),
            (2, 4),
            (3, 0),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
        ]
        assert result in valid_points

    def test_interior_point_empty_mask(self):
        # Test with an empty mask
        mask = np.zeros((3, 3), dtype=int)
        result = interior_point(mask)
        assert result is None

    def test_interior_point_single_pixel(self):
        # Test with a single pixel mask
        mask = np.zeros((3, 3), dtype=int)
        mask[1, 1] = 1
        result = interior_point(mask)
        assert result == (1, 1)

    def test_interior_point_thin_line(self):
        # Test with a thin line mask
        mask = np.zeros((5, 5), dtype=int)
        mask[2, :] = 1  # Horizontal line
        result = interior_point(mask)
        assert result is not None
        assert result[0] == 2  # y-coordinate should be 2
        assert 0 <= result[1] <= 4  # x-coordinate should be within range
