# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
import torch
import numpy as np

from zetta_utils.training.datasets import LayerDataset


def test_layer_dataset(mocker):
    layer_m = mocker.Mock()
    indexer_m = mocker.Mock()

    indexer_m.__call__ = mocker.Mock(return_value="index")
    indexer_m.__len__ = mocker.Mock(return_value=10)
    data = {
        "np": np.ones(
            (
                2,
                2,
            )
        ),
        "str": "abc",
        "list": [1, 2, 3],
        "torch": torch.ones(
            (
                2,
                2,
            )
        ),
    }
    data_expected = {
        "np": torch.ones(
            (
                2,
                2,
            )
        ),
        "str": "abc",
        "list": [1, 2, 3],
        "torch": torch.ones(
            (
                2,
                2,
            )
        ),
    }
    layer_m.__getitem__ = mocker.Mock(return_value=data)

    lds = LayerDataset(layer_m, indexer_m)
    assert len(lds) == 10
    assert lds[0] == data_expected
