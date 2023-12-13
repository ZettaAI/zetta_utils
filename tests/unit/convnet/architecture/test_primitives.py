from functools import partial

import torch

from zetta_utils.convnet.architecture.primitives import MultiHeadedOutput


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
