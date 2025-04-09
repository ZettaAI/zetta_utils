import math

import pytest
import torch
import torchfields

from ...alignment.field import get_rigidity_map_zcxy


def rotation_tensor(degrees):
    rads = math.radians(degrees)
    return torch.Tensor(
        [[math.cos(rads), -math.sin(rads), 0], [math.sin(rads), math.cos(rads), 0]]
    )


def mask_test_field():
    """Generate a displacement field that has a mild rotation, except for one
    point where there is an extreme displacement vector.  With this we should
    see a significant difference when that extreme vector is masked out."""
    fld = torchfields.Field.affine_field(rotation_tensor(45), size=(1, 2, 16, 16))
    fld[0, 0, 4, 4] = 5  # setting the vector at (4,4) to [5, -1]
    fld[0, 1, 4, 4] = -1
    return fld


def mask_test_weight_map():
    """A mask with a zero at the point of the extreme vector in mask_test_field."""
    weight_map = torch.ones(1, 16, 16)
    weight_map[0, 4, 4] = 0
    return weight_map


@pytest.mark.parametrize(
    "field, weight_map, expected_rig_low, expected_rig_high",
    [
        # No movement
        [torch.zeros(1, 2, 32, 32), None, 0, 0],
        # Translation
        [torch.ones(1, 2, 32, 32), None, 0, 0],
        # 45° rotation (mean will be slightly larger)
        [
            torchfields.Field.affine_field(rotation_tensor(45), size=(1, 2, 16, 16)),
            None,
            0,
            0.001,
        ],
        # non rigid displacement at transition for 1 to 2
        [torch.Tensor([[[[1, 1, 1, 2, 2]] * 5] * 2]), None, 0.019, 0.02],
        # 90° Rotation. Large field is required to get small penalty due to precision
        [
            torchfields.Field.affine_field(
                torch.Tensor([[[0, 1, 0], [1, 0, 0]]]), size=(1, 2, 1024, 1024)
            ),
            None,
            0,
            1e-5,
        ],
        # our mask test field (mild rotation but with one wild displacement vector)
        [mask_test_field(), None, 0.10, 0.11],
        # and same as above, but with a weight map that masks out the wild vector
        [mask_test_field(), mask_test_weight_map(), 0, 0.001],
    ],
)
def test_get_rigidity_map_zcxy_correctness(
    field: torch.Tensor,
    weight_map: torch.Tensor,
    expected_rig_low: float,
    expected_rig_high: float,
):
    # get the rigidity map for the given displacement field
    rigidity_map = get_rigidity_map_zcxy(field, weight_map=weight_map)
    # check that the mean is within the expected range
    assert rigidity_map.mean() >= expected_rig_low
    assert rigidity_map.mean() <= expected_rig_high
    # verify that the output size matches the field size
    assert rigidity_map.size()[0] == field.size()[0]  # batches
    assert rigidity_map.size()[1:] == field.size()[2:]  # width and height
