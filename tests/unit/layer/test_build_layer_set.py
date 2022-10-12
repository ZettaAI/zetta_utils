# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
from zetta_utils.layer import build_layer_set


def test_layer_set(mocker):
    kwargs = {
        "layers": {
            "layer1": mocker.Mock(),
            "layer2": mocker.Mock(),
        }
    }
    layer_set = build_layer_set(**kwargs)
    assert layer_set.io_backend.layer == kwargs["layers"]
