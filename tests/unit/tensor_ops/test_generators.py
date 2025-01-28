# pylint: disable=missing-docstring,invalid-name
import einops
import numpy as np
import pytest
import torch
from torch import nan

from zetta_utils.db_annotations.annotation import AnnotationDBEntry
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric.index import VolumetricIndex
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


@pytest.mark.parametrize(
    "mat, size, expected_shape, expected_dtype",
    [
        [
            torch.tensor(
                [[1, 0, 0], [0, 1, 0]],
                dtype=torch.float,
            ),
            16,
            (1, 2, 16, 16),
            torch.float,
        ],
        [
            np.array(
                [
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                ],
                dtype=np.float32,
            ),
            (16, 16),
            (2, 2, 16, 16),
            torch.float,
        ],
        [
            torch.tensor(
                [
                    [[1, 0, 0], [0, 1, 0]],
                    [[1, 0, 0], [0, 1, 0]],
                    [[1, 0, 0], [0, 1, 0]],
                ],
                dtype=torch.double,
            ),
            torch.Size((16, 16)),
            (3, 2, 16, 16),
            torch.double,
        ],
        [
            torch.tensor(
                [
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                ],
                dtype=torch.float16,
            ),
            [16, 16],
            (3, 2, 16, 16),
            torch.float16,
        ],
    ],
)
def test_get_field_from_matrix(mat, size, expected_shape, expected_dtype):
    result = generators.get_field_from_matrix(mat, size)
    assert result.shape == expected_shape
    assert result.dtype == expected_dtype


@pytest.mark.parametrize(
    "mat, size",
    [
        [
            torch.tensor(
                [
                    [[[1, 0, 0], [0, 1, 0]]],
                ],
                dtype=torch.float,
            ),
            (16, 16),
        ],
        [
            torch.tensor(
                [
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                ],
                dtype=torch.float,
            ),
            (3, 16, 16),
        ],
    ],
)
def test_get_field_from_matrix_exceptions(mat, size):
    with pytest.raises(ValueError):
        generators.get_field_from_matrix(mat, size)


@pytest.mark.parametrize(
    "annotations, index, expected",
    [
        # Single vector
        [
            [
                AnnotationDBEntry.from_dict(
                    "1", {"type": "line", "pointA": [1.0, 0.0, 0.0], "pointB": [0.0, 0.0, 1.0]}
                ),
            ],
            VolumetricIndex.from_coords([0, 0, 0], [2, 2, 1], Vec3D(*[1, 1, 1])),
            torch.tensor(
                [[[[nan, nan], [0.0, nan]], [[nan, nan], [-1.0, nan]]]], dtype=torch.float32
            ),
        ],
        # Two vectors, same target pixel -- should average
        [
            [
                AnnotationDBEntry.from_dict(
                    "1", {"type": "line", "pointA": [0.4, 0.0, 0.0], "pointB": [0.0, 0.0, 1.0]}
                ),
                AnnotationDBEntry.from_dict(
                    "2", {"type": "line", "pointA": [0.6, 0.0, 0.0], "pointB": [1.0, 0.0, 1.0]}
                ),
            ],
            VolumetricIndex.from_coords([0, 0, 0], [2, 2, 1], Vec3D(*[1, 1, 1])),
            torch.tensor(
                [[[[0.0, nan], [nan, nan]], [[0.0, nan], [nan, nan]]]], dtype=torch.float32
            ),
        ],
        # Annotation outside ROI
        [
            [
                AnnotationDBEntry.from_dict(
                    "1", {"type": "line", "pointA": [0.4, 0.0, 0.0], "pointB": [0.0, 0.0, 1.0]}
                ),
            ],
            VolumetricIndex.from_coords([0, 0, 1], [2, 2, 2], Vec3D(*[1, 1, 1])),
            torch.tensor(
                [[[[nan, nan], [nan, nan]], [[nan, nan], [nan, nan]]]], dtype=torch.float32
            ),
        ],
    ],
)
def test_get_field_from_annotations(annotations, index, expected):
    result = generators.get_field_from_annotations(annotations, index, device="cpu")
    torch.testing.assert_close(result, expected, equal_nan=True)
    assert result.device == torch.device("cpu")


def test_get_field_from_annotations_exception():
    with pytest.raises(ValueError):
        generators.get_field_from_annotations(
            [AnnotationDBEntry.from_dict("1", {"type": "point", "point": [0.0, 0.0, 0.0]})],
            VolumetricIndex.from_coords([0, 0, 0], [2, 2, 1], Vec3D(*[1, 1, 1])),
        )
