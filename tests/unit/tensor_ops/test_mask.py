# pylint: disable=missing-docstring,invalid-name
import numpy as np
import pytest
import skimage
import torch

from zetta_utils.tensor_ops import mask

from ..helpers import assert_array_equal


def test_skip_on_empty_data(mocker):
    wrapie = mocker.MagicMock()

    def wrapie_call(data: torch.Tensor) -> torch.Tensor:
        return data + 1

    wrapie.__call__ = wrapie_call

    wrapped = mask.skip_on_empty_data(wrapie)
    assert wrapped(torch.Tensor([0])) == torch.Tensor([0])


def test_filter_cc_small():
    a = torch.Tensor(
        [
            [
                [1, 1, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 1],
            ]
        ]
    ).unsqueeze(-1)

    expected = torch.Tensor(
        [
            [
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 1],
            ]
        ]
    ).unsqueeze(-1)

    result = mask.filter_cc(
        a,
        mode="keep_small",
        thr=2,
    )
    assert_array_equal(result, expected)


def test_filter_cc_big():
    a = torch.Tensor(
        [
            [
                [[1], [1], [0], [1]],
                [[1], [1], [0], [0]],
                [[0], [0], [0], [0]],
                [[0], [0], [1], [1]],
            ]
        ]
    )

    expected = torch.Tensor(
        [
            [
                [[1], [1], [0], [0]],
                [[1], [1], [0], [0]],
                [[0], [0], [0], [0]],
                [[0], [0], [0], [0]],
            ]
        ]
    )

    result = mask.filter_cc(
        a,
        mode="keep_large",
        thr=2,
    )
    assert_array_equal(result, expected)


def test_filter_cc3d_small():
    a = torch.Tensor(
        [
            [
                [1, 1, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 1],
            ],
            [
                [1, 1, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
            ],
        ]
    ).unsqueeze(0)

    expected = torch.Tensor(
        [
            [
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 1],
            ],
            [
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
            ],
        ]
    ).unsqueeze(0)

    result = mask.filter_cc3d(
        a,
        mode="keep_small",
        thr=2,
    )
    assert_array_equal(result, expected)


def test_filter_cc3d_large():
    a = torch.Tensor(
        [
            [
                [1, 1, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 1],
            ],
            [
                [1, 1, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
            ],
        ]
    ).unsqueeze(0)

    expected = torch.Tensor(
        [
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ]
    ).unsqueeze(0)

    result = mask.filter_cc3d(
        a,
        mode="keep_large",
        thr=2,
    )
    assert_array_equal(result, expected)


def test_kornia_closing():
    a = np.expand_dims(
        np.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 1],
                    [0, 1, 0, 1],
                    [0, 1, 1, 1],
                ]
            ],
            dtype=np.uint8,
        ),
        -1,
    )

    expected = np.expand_dims(
        np.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ]
            ],
            dtype=np.uint8,
        ),
        -1,
    )

    result = mask.kornia_closing(
        a, "square", width=3, border_type="constant", border_value=0, device="cpu"
    )
    assert_array_equal(result, expected)


def test_kornia_opening():
    a = np.expand_dims(
        np.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 1],
                    [0, 0, 1, 1],
                    [0, 1, 1, 1],
                ]
            ]
        ),
        -1,
    )

    expected = np.expand_dims(
        np.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                ]
            ]
        ),
        -1,
    )

    result = mask.kornia_opening(a, torch.ones(3, 3), border_type="geodesic")
    assert_array_equal(result, expected)


def test_kornia_dilation():
    a = np.expand_dims(
        np.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ]
        ),
        -1,
    )

    expected = np.expand_dims(
        np.array(
            [
                [
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [0, 0, 0, 0],
                ]
            ]
        ),
        -1,
    )

    result = mask.kornia_dilation(a, torch.ones(3, 3), border_type="geodesic")
    assert_array_equal(result, expected)


def test_kornia_erosion():
    a = np.expand_dims(
        np.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                ]
            ]
        ),
        -1,
    )

    expected = np.expand_dims(
        np.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                ]
            ]
        ),
        -1,
    )

    result = mask.kornia_erosion(a, torch.ones(3, 3), border_type="geodesic")
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "kernel, width, expected_kernel",
    [
        ["square", 7, torch.ones(7, 7)],
        ["diamond", 7, torch.tensor(skimage.morphology.diamond(7))],
        ["star", 5, torch.tensor(skimage.morphology.star(5))],
        ["disk", 7, torch.tensor(skimage.morphology.disk(7))],
        [torch.ones(5, 3), None, torch.ones(5, 3)],
        [np.ones((5, 3)), None, torch.ones(5, 3)],
    ],
)
def test_normalize_kernel(kernel, width, expected_kernel):
    result = mask._normalize_kernel(kernel, width, device=None)  # pylint: disable=protected-access
    assert_array_equal(result, expected_kernel)


@pytest.mark.parametrize(
    "kernel, width, expected_exc",
    [
        ["ball", 7, ValueError],
        ["square", 2.5, TypeError],
        ["square", -1, ValueError],
        [torch.ones(5, 3, 2), None, ValueError],
        [np.ones((5, 3, 2)), None, ValueError],
    ],
)
def test_normalize_kernel_exc(kernel, width, expected_exc):
    with pytest.raises(expected_exc):
        mask._normalize_kernel(kernel, width, device=None)  # pylint: disable=protected-access


def test_combine_mask_fns():
    data = torch.Tensor([0, 1, 2, 3])
    expected = torch.Tensor([1, 1, 1, 0]).bool()

    result = mask.combine_mask_fns(
        data,
        fns=[
            lambda x: x % 2 == 0,
            lambda x: x == 1,
        ],
    )
    assert_array_equal(result, expected)


def test_combine_mask_fns_exc():
    with pytest.raises(ValueError):
        mask.combine_mask_fns(data=torch.zeros((10, 10)), fns=[])
