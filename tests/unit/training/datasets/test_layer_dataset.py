# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
import torch
import numpy as np

from zetta_utils.training.datasets import LayerDataset

from ...helpers import assert_array_equal


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
            ),
            dtype=torch.double,
        ),
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
    result = lds[0]
    assert_array_equal(result["np"], data_expected["np"])
    assert_array_equal(result["torch"], data_expected["torch"])
    assert result["list"] == data_expected["list"]
