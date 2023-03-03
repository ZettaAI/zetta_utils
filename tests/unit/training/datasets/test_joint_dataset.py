# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,unsubscriptable-object
from zetta_utils.training.datasets import JointDataset


def test_joint_dataset_constructor(mocker):
    layer_dataset1_m = mocker.Mock()
    layer_dataset2_m = mocker.Mock()

    layer_dataset1_m.__len__ = mocker.Mock(return_value=2)
    layer_dataset2_m.__len__ = mocker.Mock(return_value=3)

    JointDataset("horizontal", {"lds1": layer_dataset1_m, "lds2": layer_dataset2_m})
    JointDataset("vertical", {"lds1": layer_dataset1_m, "lds2": layer_dataset2_m})
    JointDataset("vertical", {"lds1": layer_dataset1_m, "lds2": layer_dataset2_m}, sampling_order="shuffle")


def test_joint_dataset_len(mocker):
    layer_dataset1_m = mocker.Mock()
    layer_dataset2_m = mocker.Mock()

    layer_dataset1_m.__len__ = mocker.Mock(return_value=2)
    layer_dataset2_m.__len__ = mocker.Mock(return_value=3)

    jds_h = JointDataset("horizontal", {"lds1": layer_dataset1_m, "lds2": layer_dataset2_m})
    jds_v = JointDataset("vertical", {"lds1": layer_dataset1_m, "lds2": layer_dataset2_m})
    jds_vs = JointDataset("vertical", {"lds1": layer_dataset1_m, "lds2": layer_dataset2_m}, sampling_order="shuffle")

    assert len(jds_h) == 2
    assert len(jds_v) == 5
    assert len(jds_vs) == 5


def test_joint_dataset_getitem(mocker):
    layer_dataset1_m = mocker.Mock()
    layer_dataset2_m = mocker.Mock()

    layer_dataset1_m.__len__ = mocker.Mock(return_value=2)
    layer_dataset2_m.__len__ = mocker.Mock(return_value=3)

    layer_dataset1_m.__getitem__ = mocker.Mock(return_value=42)
    layer_dataset2_m.__getitem__ = mocker.Mock(return_value=57)

    mocker.patch("random.shuffle", lambda x: x.reverse())

    jds_h = JointDataset("horizontal", {"lds1": layer_dataset1_m, "lds2": layer_dataset2_m})
    jds_v = JointDataset("vertical", {"lds1": layer_dataset1_m, "lds2": layer_dataset2_m})
    jds_vs = JointDataset("vertical", {"lds1": layer_dataset1_m, "lds2": layer_dataset2_m}, sampling_order="shuffle")

    assert jds_h[0] == {"lds1": 42, "lds2": 57}
    assert [jds_v[i] for i in range(5)] == [42, 42, 57, 57, 57]
    assert [jds_vs[i] for i in range(5)] == [57, 57, 57, 42, 42]
