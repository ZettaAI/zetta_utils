# pylint: disable=missing-docstring,invalid-name
import einops
import pytest
import torch

from zetta_utils.tensor_ops import generators


@pytest.mark.parametrize(
    "data, kwargs, expected",
    [
        [
            torch.tensor(
                [
                    [1, 1],
                    [1, 1],
                ],
                dtype=torch.float,
            ),
            {"trans_x_px": 1},
            torch.tensor(
                [
                    [0, 1],
                    [0, 1],
                ],
                dtype=torch.float,
            ),
        ],
        [
            torch.tensor(
                [
                    [1, 1],
                    [1, 1],
                ],
                dtype=torch.float,
            ),
            {
                "trans_x_px": 1,
                "trans_y_px": 1,
            },
            torch.tensor(
                [
                    [0, 0],
                    [0, 1],
                ],
                dtype=torch.float,
            ),
        ],
        [
            torch.tensor(
                [
                    [1, 1],
                    [1, 1],
                ],
                dtype=torch.float,
            ),
            {"trans_x_px": 1, "trans_y_px": 1, "rot_deg": 90},
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                ],
                dtype=torch.float,
            ),
        ],
        [
            torch.tensor(
                [
                    [1, 1],
                    [1, 1],
                ],
                dtype=torch.float,
            ),
            {
                "shear_x_deg": 45,
            },
            torch.tensor(
                [
                    [1, 0.5],
                    [0.5, 1],
                ],
                dtype=torch.float,
            ),
        ],
    ],
)
def test_get_affine_field(data, kwargs, expected):
    field = generators.get_affine_field(size=data.shape[-1], **kwargs)
    field = einops.rearrange(field, "C X Y Z -> Z C X Y")
    result = field.from_pixels().sample(data)  # type: ignore
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "shape, res, device, expected_shape, expected_device",
    [
        [(1, 16, 16, 3), [4, 4], "cpu", torch.Size([1, 16, 16, 3]), torch.device("cpu")],
        [[2, 16, 16, 2], [4, 4], "cpu", torch.Size([2, 16, 16, 2]), torch.device("cpu")],
        [(3, 16, 16, 1), [4, 4], None, torch.Size([3, 16, 16, 1]), torch.device("cpu")],
        [(2, 16, 24, 1), [4, 6], None, torch.Size([2, 16, 24, 1]), torch.device("cpu")],
    ],
)
def test_rand_perlin_2d(shape, res, device, expected_shape, expected_device):
    result = generators.rand_perlin_2d(shape, res, device=device)
    assert result.shape == expected_shape
    assert result.device == expected_device


@pytest.mark.parametrize(
    "shape, res, device, expected_shape, expected_device",
    [
        [(1, 16, 16, 3), [4, 4], "cpu", torch.Size([1, 16, 16, 3]), torch.device("cpu")],
        [[2, 16, 16, 2], [4, 4], "cpu", torch.Size([2, 16, 16, 2]), torch.device("cpu")],
        [(3, 16, 16, 1), [4, 4], None, torch.Size([3, 16, 16, 1]), torch.device("cpu")],
        [(2, 16, 24, 1), [4, 6], None, torch.Size([2, 16, 24, 1]), torch.device("cpu")],
    ],
)
def test_rand_perlin_2d_octaves(shape, res, device, expected_shape, expected_device):
    result = generators.rand_perlin_2d_octaves(shape, res, octaves=2, device=device)
    assert result.shape == expected_shape
    assert result.device == expected_device


@pytest.mark.parametrize(
    "shape, res",
    [
        [(1, 16, 16), [8, 8]],
        [[2, 16, 16, 2], (4,)],
    ],
)
def test_rand_perlin_2d_failure(shape, res):
    with pytest.raises(ValueError):
        generators.rand_perlin_2d(shape, res)


@pytest.mark.parametrize(
    "shape, res",
    [
        [(1, 16, 16), [8, 8]],
        [[2, 16, 16, 2], (4,)],
    ],
)
def test_rand_perlin_2d_octaves_failure(shape, res):
    with pytest.raises(ValueError):
        generators.rand_perlin_2d_octaves(shape, res)
