import pickle
from functools import partial

import pytest
import torch

from zetta_utils.convnet.architecture.primitives import (
    AvgPool2DFlatten,
    CenterCrop,
    Clamp,
    Crop,
    Flatten,
    MaxPool2DFlatten,
    MultiHeaded,
    MultiHeadedOutput,
    RescaleValues,
    SplitTuple,
    Unflatten,
    UpConv,
    View,
)


def test_multihead_output():
    mho = MultiHeadedOutput(
        in_channels=3,
        heads={"first": 1, "second": 2},
        conv=partial(torch.nn.Conv2d, kernel_size=3),
        preactivation=torch.nn.ELU(),
        activation=torch.nn.ReLU(),
    )
    assert len(list(mho.heads["first"].children())) == 3
    assert len(list(mho.heads["second"].children())) == 3
    first_children = list(mho.heads["first"].children())
    second_children = list(mho.heads["second"].children())
    assert isinstance(first_children[0], torch.nn.Conv2d)
    assert isinstance(second_children[0], torch.nn.Conv2d)
    assert first_children[0].out_channels == 1
    assert second_children[0].out_channels == 2
    assert isinstance(first_children[1], torch.nn.ELU)
    assert isinstance(second_children[1], torch.nn.ELU)
    assert isinstance(first_children[2], torch.nn.ReLU)
    assert isinstance(second_children[2], torch.nn.ReLU)


@pytest.mark.parametrize(
    "primitive_class,args,kwargs",
    [
        (SplitTuple, (), {}),
        (Flatten, (), {}),
        (Unflatten, (), {}),
        (MaxPool2DFlatten, (), {}),
        (AvgPool2DFlatten, (), {}),
        (RescaleValues, ((0, 255), (0, 1)), {}),
        (View, ((-1, 300),), {}),
        (Clamp, (), {"min": 0, "max": 1}),
        (Crop, ([5, 5],), {}),
        (CenterCrop, ([10, 10],), {}),
        (UpConv, (), {"in_channels": 64, "out_channels": 32, "kernel_size": 3}),
        (
            MultiHeaded,
            (),
            {"heads": {"first": torch.nn.Linear(100, 1), "second": torch.nn.Linear(100, 2)}},
        ),
        (
            MultiHeadedOutput,
            (),
            {
                "in_channels": 3,
                "heads": {"first": 1, "second": 2},
                "conv": partial(torch.nn.Conv2d, kernel_size=3, padding=1),
                "preactivation": torch.nn.ReLU(),
                "activation": torch.nn.Sigmoid(),
            },
        ),
    ],
)
def test_primitive_pickle(primitive_class, args, kwargs):
    """Test that all primitive classes can be pickled and unpickled correctly."""
    primitive = primitive_class(*args, **kwargs)

    try:
        pickled = pickle.dumps(primitive)
        unpickled = pickle.loads(pickled)
    except Exception as e:  # pylint: disable=broad-exception-caught
        pytest.fail(f"Failed to pickle/unpickle {primitive_class.__name__}: {e}")

    assert isinstance(
        unpickled, primitive_class
    ), f"Unpickled object is not instance of {primitive_class.__name__}"
