# pylint: disable=missing-docstring,invalid-name
import pytest
import torch

from zetta_utils.tensor_ops import transform


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
    field = transform.get_affine_field(size=data.shape[-1], **kwargs)
    result = field.from_pixels().sample(data)  # type: ignore
    torch.testing.assert_close(result, expected)
