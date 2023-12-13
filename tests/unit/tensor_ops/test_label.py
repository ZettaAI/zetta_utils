import torch

from zetta_utils.tensor_ops.label import seg_to_aff

from ..helpers import assert_array_equal


def test_convert_seg_to_aff():
    seg = torch.Tensor(
        [
            [
                [1, 1, 1],
                [1, 2, 1],
                [1, 2, 1],
            ]
        ]
        * 3
    )
    correct_aff = torch.Tensor(
        [
            [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        ]
    )
    correct_aff_mask = torch.Tensor(
        [
            [[False, False, False], [False, True, False]],
            [[False, False, False], [False, True, False]],
            [[False, False, False], [False, True, False]],
        ],
    )
    seg_mask = seg == 2
    aff, aff_mask = seg_to_aff(seg, edge=[0, 1, 0], mask=seg_mask)
    assert_array_equal(aff, correct_aff)
    assert_array_equal(aff_mask, correct_aff_mask)
