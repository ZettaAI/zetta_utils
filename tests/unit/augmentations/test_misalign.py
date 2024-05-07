from typing import Literal

import pytest
import torch

from zetta_utils.augmentations.misalign import MisalignProcessor
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.geometry.vec import Vec3D
from zetta_utils.layer.volumetric.index import VolumetricIndex

from ..helpers import assert_array_equal


def test_write_exc(mocker):
    idx = mocker.MagicMock()
    data = mocker.MagicMock()
    proc = MisalignProcessor(
        prob=1.0, disp_min_in_unit=1, disp_max_in_unit=1, disp_in_unit_must_be_divisible_by=1
    )
    proc.prepared_disp_fraction = mocker.MagicMock()
    with pytest.raises(RuntimeError):
        proc.process_index(idx, mode="write")

    with pytest.raises(RuntimeError):
        proc.process_data(data, mode="write")


def test_tensor_process_data_slip_pos(mocker):
    data_padded = torch.zeros((1, 5, 5, 5))
    for x in range(5):
        for y in range(5):
            for z in range(5):
                data_padded[0, x, y, z] = 100 * z + 10 * y + x

    proc = MisalignProcessor(
        prob=1.0,
        disp_min_in_unit=1,
        disp_max_in_unit=1,
        disp_in_unit_must_be_divisible_by=1,
        mode="slip",
    )
    proc.prepared_disp_fraction = (1 / 5, 2 / 5)
    chosen_z = 3
    mocker.patch("zetta_utils.augmentations.misalign._select_z", return_value=chosen_z)
    result = proc.process_data(data_padded.clone(), mode="read")
    for x in range(4):
        for y in range(3):
            for z in range(4):
                if z == chosen_z:
                    assert result[0, x, y, z] == 100 * z + 10 * y + x
                else:
                    assert result[0, x, y, z] == 100 * z + 10 * (y + 2) + (x + 1)


def test_tensor_process_data_slip_neg(mocker):
    data_padded = torch.zeros((1, 5, 5, 5))
    for x in range(5):
        for y in range(5):
            for z in range(5):
                data_padded[0, x, y, z] = 100 * z + 10 * y + x

    proc = MisalignProcessor(
        prob=1.0,
        disp_min_in_unit=1,
        disp_max_in_unit=1,
        disp_in_unit_must_be_divisible_by=1,
        mode="slip",
    )
    proc.prepared_disp_fraction = (-1 / 5, -2 / 5)
    chosen_z = 3
    mocker.patch("zetta_utils.augmentations.misalign._select_z", return_value=chosen_z)
    result = proc.process_data(data_padded.clone(), mode="read")
    for x in range(4):
        for y in range(3):
            for z in range(4):
                if z == chosen_z:
                    assert result[0, x, y, z] == 100 * z + 10 * (y + 2) + (x + 1)
                else:
                    assert result[0, x, y, z] == 100 * z + 10 * y + x


def test_tensor_process_data_step_pos(mocker):
    data_padded = torch.zeros((1, 5, 5, 5))
    for x in range(5):
        for y in range(5):
            for z in range(5):
                data_padded[0, x, y, z] = 100 * z + 10 * y + x

    proc = MisalignProcessor(
        prob=1.0,
        disp_min_in_unit=1,
        disp_max_in_unit=1,
        disp_in_unit_must_be_divisible_by=1,
        mode="step",
    )
    proc.prepared_disp_fraction = (1 / 5, 2 / 5)
    chosen_z = 3
    mocker.patch("zetta_utils.augmentations.misalign._select_z", return_value=chosen_z)
    result = proc.process_data(data_padded.clone(), mode="read")
    for x in range(4):
        for y in range(3):
            for z in range(4):
                if z >= chosen_z:
                    assert result[0, x, y, z] == 100 * z + 10 * y + x
                else:
                    assert result[0, x, y, z] == 100 * z + 10 * (y + 2) + (x + 1)


def test_dict_process_data_slip_pos(mocker):
    tensor_data_padded = torch.zeros((1, 5, 5, 5))
    for x in range(5):
        for y in range(5):
            for z in range(5):
                tensor_data_padded[0, x, y, z] = 100 * z + 10 * y + x
    data = {
        "key0": 100,
        "key1": tensor_data_padded.clone(),
        "key2": tensor_data_padded.clone(),
        "key3": tensor_data_padded.clone(),
    }
    keys_to_apply = ["key1", "key2"]
    proc = MisalignProcessor[dict[str, torch.Tensor]](
        prob=1.0,
        disp_min_in_unit=1,
        disp_max_in_unit=1,
        disp_in_unit_must_be_divisible_by=1,
        mode="slip",
        keys_to_apply=keys_to_apply,
    )
    proc.prepared_disp_fraction = (1 / 5, 2 / 5)
    chosen_z = 3
    mocker.patch("zetta_utils.augmentations.misalign._select_z", return_value=chosen_z)

    result = proc.process_data(data, mode="read")
    assert result.keys() == data.keys()
    for x, y, z in zip(range(4), range(3), range(4)):
        for k, v in data.items():
            if k in keys_to_apply:
                if z == chosen_z:
                    assert result[k][0, x, y, z] == 100 * z + 10 * y + x
                else:
                    assert result[k][0, x, y, z] == 100 * z + 10 * (y + 2) + (x + 1)
            elif isinstance(v, torch.Tensor):
                assert_array_equal(result[k], v)
            else:
                assert result[k] == v


def test_dict_process_slip_no_keys_exc():
    data = {"key": torch.ones((1, 5, 5, 5))}
    proc = MisalignProcessor[dict[str, torch.Tensor]](
        prob=1.0,
        disp_min_in_unit=1,
        disp_max_in_unit=1,
        disp_in_unit_must_be_divisible_by=1,
        mode="slip",
    )
    proc.prepared_disp_fraction = (1 / 5, 1 / 5)
    with pytest.raises(ValueError):
        proc.process_data(data, mode="read")


def test_dict_process_step_no_keys():
    data = {"key": torch.ones((1, 5, 5, 5))}
    proc = MisalignProcessor[dict[str, torch.Tensor]](
        prob=1.0,
        disp_min_in_unit=1,
        disp_max_in_unit=1,
        disp_in_unit_must_be_divisible_by=1,
        mode="step",
    )
    proc.prepared_disp_fraction = (1 / 5, 1 / 5)
    result = proc.process_data(data, mode="read")
    assert result["key"].shape == [1, 4, 4, 5]


@pytest.mark.parametrize(
    "misalign_mode, cropped_region, expect_change",
    [
        ["slip", "upper", False],
        ["slip", "lower", False],
        ["step", "upper", False],
        ["step", "lower", True],
    ],
)
def test_dict_process_diff_size(
    misalign_mode: Literal["slip", "step"],
    cropped_region: Literal["upper", "lower"],
    expect_change: bool,
    mocker,
):
    data_in = {"key0": torch.ones((1, 5, 5, 5)), "key1": torch.rand((1, 5, 5, 3))}
    proc = MisalignProcessor[dict[str, torch.Tensor]](
        prob=1.0,
        disp_min_in_unit=1,
        disp_max_in_unit=1,
        disp_in_unit_must_be_divisible_by=1,
        mode=misalign_mode,
        keys_to_apply=["key0", "key1"],
    )
    proc.prepared_disp_fraction = (1 / 5, 1 / 5)
    mocker.patch(
        "zetta_utils.augmentations.misalign._choose_pos_or_neg_misalignent", return_value=1
    )
    if cropped_region == "lower":
        mocker.patch("zetta_utils.augmentations.misalign._select_z", return_value=0)
    else:
        assert cropped_region == "upper"
        mocker.patch("zetta_utils.augmentations.misalign._select_z", return_value=4)

    result = proc.process_data({k: v.clone() for k, v in data_in.items()}, mode="read")

    diff = (result["key1"] != data_in["key1"][:, 1:, 1:]).sum() != 0
    if expect_change:
        assert diff
    else:
        assert not diff


def test_dict_process_diff_xy_size_exc():
    data = {"key0": torch.ones((1, 5, 5, 5)), "key1": torch.ones((1, 10, 10, 5))}
    proc = MisalignProcessor[dict[str, torch.Tensor]](
        prob=1.0,
        disp_min_in_unit=1,
        disp_max_in_unit=1,
        mode="slip",
        keys_to_apply=["key0", "key1"],
    )
    proc.prepared_disp_fraction = (1 / 5, 1 / 5)
    with pytest.raises(ValueError):
        proc.process_data(data, mode="read")


def test_process_index_pos(mocker):
    idx_in = VolumetricIndex(
        resolution=Vec3D(1, 1, 1), bbox=BBox3D(bounds=((1, 2), (10, 20), (100, 200)))
    )
    proc = MisalignProcessor(
        prob=1.0,
        disp_min_in_unit=1,
        disp_max_in_unit=1,
        disp_in_unit_must_be_divisible_by=1,
    )
    mocker.patch(
        "zetta_utils.augmentations.misalign._choose_pos_or_neg_misalignent", return_value=1
    )
    idx_out = proc.process_index(idx_in, mode="read")
    assert idx_out == VolumetricIndex(
        resolution=Vec3D(1, 1, 1), bbox=BBox3D(bounds=((0, 2), (9, 20), (100, 200)))
    )


def test_process_index_neg(mocker):
    idx_in = VolumetricIndex(
        resolution=Vec3D(1, 1, 1), bbox=BBox3D(bounds=((1, 2), (10, 20), (100, 200)))
    )
    proc = MisalignProcessor(
        prob=1.0, disp_min_in_unit=1, disp_max_in_unit=1, disp_in_unit_must_be_divisible_by=1
    )
    mocker.patch(
        "zetta_utils.augmentations.misalign._choose_pos_or_neg_misalignent", return_value=-1
    )
    idx_out = proc.process_index(idx_in, mode="read")
    assert idx_out == VolumetricIndex(
        resolution=Vec3D(1, 1, 1), bbox=BBox3D(bounds=((1, 3), (10, 21), (100, 200)))
    )
