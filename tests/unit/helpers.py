# pylint: disable=invalid-name
"""Common testing utilities."""
import numpy as np
import torch


def assert_array_equal(a, b):
    """Generic assertion for equality testing."""
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    else:
        raise ValueError(f"Invalid input type '{type(a)}'")
