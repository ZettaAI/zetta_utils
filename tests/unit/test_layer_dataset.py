# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
from zetta_utils.training.datasets import LayerDataset


def test_layer_dataset(mocker):
    layer_m = mocker.Mock()
    indexer_m = mocker.Mock()
    indexer_m.__call__ = mocker.Mock(return_value="index")
    indexer_m.__len__ = mocker.Mock(return_value=10)
    layer_m.__getitem__ = mocker.Mock(return_value="data")
    lds = LayerDataset(layer_m, indexer_m)
    assert len(lds) == 10
    assert lds[0] == "data"
