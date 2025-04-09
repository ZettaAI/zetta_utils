import copy

import numpy as np
import pytest
import torch

from zetta_utils.internal.segmentation.agglomeration import (
    extract_region_graph_mean_affinity,
    extract_segments,
    get_connected_components_from_edges,
    remap_np_array,
    run_agglomeration_aff,
    run_agglomeration_rag,
)


def test_aff_agglomeration_dummy():
    # Just making sure that things are not broken
    # TODO: use real data
    affs = torch.rand(3, 8, 8, 8)
    supervoxels = (torch.rand(1, 8, 8, 8) * 2000).to(torch.int64)
    run_agglomeration_aff(
        affs=affs,
        supervoxels=supervoxels,
        threshold=0.0,
    )


def test_region_graph_agglomeration_dummy():
    # Just making sure that things are not broken
    # TODO: use real data
    affs = torch.rand(3, 8, 8, 8)
    supervoxels = (torch.rand(1, 8, 8, 8) * 2000).to(torch.int64)
    rag, rag_meta = extract_region_graph_mean_affinity(
        affs=affs,
        supervoxels=supervoxels,
    )

    supervoxels_ = copy.deepcopy(supervoxels)

    merge_history, seg_1 = run_agglomeration_rag(
        region_graph=rag,
        region_graph_metadata=rag_meta,
        threshold=0.2,
        supervoxels=supervoxels,
    )
    assert seg_1 is not None

    supervoxels_1 = copy.deepcopy(supervoxels_)
    seg_2 = extract_segments(supervoxels_1, merge_history, threshold=0.2)

    # same thresholds should give the same # of segments
    assert len(np.unique(seg_1)) == len(np.unique(seg_2))

    supervoxels_2 = copy.deepcopy(supervoxels_)
    seg_3 = extract_segments(supervoxels_2, merge_history, threshold=0.9)

    # different thresholds should give the different # of segments
    assert len(np.unique(seg_1)) != len(np.unique(seg_3))


def test_remap_np_array():
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 888, 888, 9999])
    mapping = {
        2: 234,
        3: 4,
        4: 3,
        5: 567,
        888: 8888,
        9999: 10,
    }
    result = remap_np_array(arr, mapping)
    answer = np.array([0, 1, 234, 4, 3, 567, 6, 7, 8888, 8888, 10])
    for i, j in zip(result, answer):
        assert i == j


@pytest.mark.parametrize(
    "edges,num_components",
    [
        [[], 0],
        [[(1, 2)], 1],
        [[(1, 2), (1, 3)], 1],
        [[(1, 2), (1, 3), (2, 3)], 1],
        [[(1, 2), (1, 3), (2, 3), (4, 4)], 2],
        [[(1, 2), (2, 3), (4, 5), (4, 6), (4, 7), (8, 9)], 3],
    ],
)
def test_get_connected_components_from_edges(edges, num_components):
    res = get_connected_components_from_edges(edges)
    assert len(list(res)) == num_components
