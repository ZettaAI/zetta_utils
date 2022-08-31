# pylint: disable=missing-docstring,invalid-name
import numpy as np

from zetta_utils.tensor_ops import mask


def test_filter_cc_small():
    a = np.array(
        [
            [1, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
        ]
    )

    expected = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
        ]
    )

    result = mask.filter_cc(
        a,
        mode="keep_small",
        thr=2,
    )
    np.testing.assert_array_equal(result, expected)


def test_filter_cc_big():
    a = np.array(
        [
            [1, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
        ]
    )

    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    result = mask.filter_cc(
        a,
        mode="keep_large",
        thr=2,
    )
    np.testing.assert_array_equal(result, expected)
