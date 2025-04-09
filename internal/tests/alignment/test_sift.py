import numpy as np
import pytest
import torch

from zetta_utils.internal.alignment.sift import Transform2D


def test_transform_2d_invalid_estimate_mode():
    with pytest.raises(ValueError):
        Transform2D(estimate_mode="invalid_mode")  # type: ignore[arg-type]


def test_transform_2d_invalid_transformation_mode():
    with pytest.raises(ValueError):
        Transform2D(transformation_mode="invalid_mode")  # type: ignore[arg-type]


def test_transform_2d_rigid_transformation():
    src = (torch.rand(1, 101, 101, 1) + 1.0) * 127.0
    tgt = torch.rot90(src, dims=(1, 2))
    expected_transform = torch.Tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).reshape(1, 3, 3, 1)
    transform = Transform2D(transformation_mode="rigid")
    estimated_transform = transform(src.to(torch.uint8), tgt.to(torch.uint8))
    assert np.allclose(estimated_transform, expected_transform, atol=1e-4)
