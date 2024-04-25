import pytest
import torch

from zetta_utils.mazepa_layer_processing.segmentation import watershed


@pytest.mark.parametrize(
    "fragments_in_xy",
    [False, True],
)
def test_ws_dummy_data_lsd(fragments_in_xy):
    affs = torch.zeros(3, 8, 8, 8)
    ret = watershed.watershed_from_affinities(affs, method="lsd", fragments_in_xy=fragments_in_xy)
    assert ret.shape == (1, 8, 8, 8)


@pytest.mark.parametrize(
    "size_threshold",
    [0, 200],
)
def test_ws_dummy_data_abiss(size_threshold):
    affs = torch.zeros(3, 8, 8, 8)
    ret = watershed.watershed_from_affinities(affs, method="abiss", size_threshold=size_threshold)
    assert ret.shape == (1, 8, 8, 8)
