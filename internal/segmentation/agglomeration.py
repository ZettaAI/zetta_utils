from __future__ import annotations

import pickle
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import einops
import networkx
import numpy as np
import numpy.typing as npt
import tensorstore as ts
import typeguard
import waterz

from zetta_utils import builder, tensor_ops
from zetta_utils.tensor_ops.common import rearrange, squeeze, unsqueeze
from zetta_utils.tensor_typing import TensorTypeVar

RegionGraphType = Sequence[MutableMapping[str, Any]]  # entry: {u, v, score}
RegionGraphMetadataType = Sequence[Any]
MergeHistoryType = Sequence[MutableMapping[str, Any]]  # entry: {a, b, c, score}


@typeguard.typechecked
def run_agglomeration_aff(
    affs: TensorTypeVar,
    supervoxels: TensorTypeVar,
    threshold: float,
    discretize_queue: int = 256,
    affs_in_xyz: bool = True,
    tensor_in_xyz: bool = True,
    **kwargs,
) -> tuple[TensorTypeVar, dict[str, Any]]:
    """
    Run agglomeration using waterz.

    :param affs, supervoxels: Input tensors
    :param threshold: Threshold to agglomerate down to.
    :param discretize_queue: Whether to discretize waterz's merge queues and improve
        performance (usually safe)
    :param affs_in_xyz: Indicating whether `affs` is ordered xyz or zyx.
    :param tensor_in_xyz: Indicating whether `supervoxels` is ordered cxyz or czyx.
    :return: Relabeled `supervoxels`
    """

    if tensor_in_xyz:
        affs = rearrange(affs, pattern="C X Y Z -> C Z Y X")
    affs_np = tensor_ops.convert.to_np(affs)
    if affs_np.shape[0] != 3:
        raise RuntimeError(
            f"Affinity channel should only have 3 components, receiving {affs_np.shape[0]}"
        )
    if affs_in_xyz:
        affs_np = np.flip(affs_np, 0)
    if affs_np.dtype == "uint8":
        # Assume affs are in range 0-255. Convert to 0.0-1.0 range.
        affs_np = affs_np.astype("float32") / 255
    assert affs_np.dtype == "float32"

    if tensor_in_xyz:
        supervoxels = rearrange(supervoxels, pattern="1 X Y Z -> Z Y X")
    supervoxels_np = tensor_ops.convert.to_np(supervoxels).astype("uint64")

    generator = waterz.agglomerate(
        affs=affs_np,
        thresholds=[0.0, 1.0 - threshold],
        fragments=supervoxels_np,
        discretize_queue=discretize_queue,
        return_merge_history=True,
        return_region_graph=True,
        return_region_graph_metadata=True,
        # scoring_function=merge_function,
        **kwargs,
    )

    _, _, rag, rag_meta = next(generator)
    seg, merge_history, _, _ = next(generator)
    metadata = {
        "merge_history": merge_history,
        "region_graph": rag,
        "region_graph_metadata": rag_meta,
    }

    for _ in generator:
        pass  # cleanup waterz internal states

    seg = einops.rearrange(seg, "Z Y X -> 1 X Y Z")
    seg_ret = tensor_ops.convert.astype(seg, supervoxels)

    return seg_ret, metadata


@typeguard.typechecked
def run_agglomeration_rag(
    region_graph: RegionGraphType,
    region_graph_metadata: RegionGraphMetadataType,
    threshold: float,
    supervoxels: TensorTypeVar | None = None,
    discretize_queue: int = 256,
    **kwargs,
) -> tuple[MergeHistoryType, TensorTypeVar | None]:
    """
    Run agglomeration on a precomputed region graph.

    :param region_graph: List of edges of the form {u, v, score} with score in [0.0, 1.0].
        In term of affinities, score = 1.0 - aff.
    :param region_graph_metadata: Metadata for the region graph. For mean affinity, it is
        a list of size of each edge in `region_graph`.
    :param threshold: Threshold to agglomerate down to.
    :param supervoxels: Optional tensor to be relabeled.
    :param discretize_queue: Whether to discretize waterz's merge queues and improve
        performance (usually safe)
    :return: Merge history and optionally relabeled `supervoxels`
    """

    if supervoxels is not None:
        # for simple relabeling, we don't actually need to worry about xyz or zyx dim ordering.
        supervoxels_np = tensor_ops.convert.to_np(squeeze(supervoxels, dim=0)).astype("uint64")

    generator = waterz.agglomerate(
        input_rag=region_graph,
        input_rag_metadata=region_graph_metadata,
        thresholds=[1.0 - threshold],
        fragments=supervoxels_np,
        discretize_queue=discretize_queue,
        # scoring_function=merge_function,
        # force_rebuild=True,
        **kwargs,
    )

    merge_history, seg = next(generator)

    for _ in generator:
        pass  # cleanup waterz internal states

    if seg is not None:
        assert supervoxels is not None
        seg = tensor_ops.convert.astype(unsqueeze(seg, dim=0), supervoxels)

    return merge_history, seg


@builder.register("test_agg_rg_seg")
@typeguard.typechecked
def test_agg_rg_seg(
    supervoxels: TensorTypeVar,
    threshold: float,
    path: str,
    discretize_scores: int = 255,
) -> TensorTypeVar:

    db = ts.KvStore.open(
        {
            "driver": "file",
            "path": path,
        }
    ).result()
    rag = pickle.loads(db["rag"])
    rag_meta = pickle.loads(db["rag_meta"])

    if discretize_scores > 0:
        for e in rag:
            e["score"] = float(e["score"]) / discretize_scores
        rag_meta = [float(k) / discretize_scores for k in rag_meta]

    merge_history, seg = run_agglomeration_rag(
        region_graph=rag,
        region_graph_metadata=rag_meta,
        threshold=threshold,
        supervoxels=supervoxels,
    )
    assert seg is not None

    db = ts.KvStore.open(
        {
            "driver": "file",
            "path": path,
        }
    ).result()
    db["merge_history"] = pickle.dumps(merge_history)

    return tensor_ops.convert.astype(seg, supervoxels)


@builder.register("extract_region_graph_mean_affinity")
@typeguard.typechecked
def extract_region_graph_mean_affinity(
    affs: TensorTypeVar,
    supervoxels: TensorTypeVar,
    **kwargs,
) -> tuple[RegionGraphType, RegionGraphMetadataType]:

    _, metadata = run_agglomeration_aff(
        affs=affs, supervoxels=supervoxels, threshold=1.0, **kwargs
    )
    return metadata["region_graph"], metadata["region_graph_metadata"]


@builder.register("test_extract_region_graph_mean_affinity")
@typeguard.typechecked
def test_extract_region_graph_mean_affinity(
    affs: TensorTypeVar,  # in CXYZ
    supervoxels: TensorTypeVar,
    path: str,
    discretize_scores: int = 255,
) -> None:
    rag, rag_meta = extract_region_graph_mean_affinity(affs, supervoxels)

    if discretize_scores > 0:
        for e in rag:
            e["score"] = int(e["score"] * discretize_scores)
        rag_meta = [int(k * discretize_scores) for k in rag_meta]

    db = ts.KvStore.open(
        {
            "driver": "file",
            "path": path,
        }
    ).result()
    db["rag"] = pickle.dumps(rag)
    db["rag_meta"] = pickle.dumps(rag_meta)


@typeguard.typechecked
def remap_np_array(
    arr: TensorTypeVar,
    mapping: Mapping[int, int],
) -> npt.NDArray:
    """
    Remap an array with given mapping.

    :param arr: Input tensor.
    :param mapping: Mapping from old to new values. Values not found in the map will
        be preserved.
    :return: Remapped tensor.
    """
    # from https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
    # TODO: check if funlib.segment is faster
    values, inverse = np.unique(arr, return_inverse=True)
    values = np.array([mapping.get(x, x) for x in values])  # Any remapping method works.
    return values[inverse].reshape(arr.shape)


@typeguard.typechecked
def get_connected_components_from_edges(
    edges: Sequence[tuple[int, int]],
) -> Iterable[Iterable[int]]:
    merge_graph = networkx.Graph()
    for a, b in edges:
        merge_graph.add_edge(a, b)
    return networkx.connected_components(merge_graph)


@typeguard.typechecked
def extract_segments(
    supervoxels: TensorTypeVar,
    merge_history: MergeHistoryType,
    threshold: float = 0.0,
) -> TensorTypeVar:
    """
    Relabel `supervoxels` given a merge history.

    :param supervoxels: Volume to be relabeled.
    :param merge_history: List of scored edges. Each entry is a dict of the form {a, b, c, score}.
    :param threshold: Optional threshold where if specified edges with affinity below
        this threshold will be ignored.
    :return: Relabeled supervoxels.
    """
    cc_edges = []
    for e in merge_history:
        a, b, score = e["a"], e["b"], e["score"]
        edge_aff = 1.0 - score
        if edge_aff > threshold:
            cc_edges.append((a, b))

    components_list = get_connected_components_from_edges(cc_edges)

    def get_first_in_set(s):
        # No method to get one element with set :/
        for e in s:
            return e

    mapper = {
        k: get_first_in_set(components) for components in components_list for k in components
    }
    seg = remap_np_array(supervoxels, mapper)
    return tensor_ops.convert.astype(seg, supervoxels)


@builder.register("test_label_segments")
@typeguard.typechecked
def test_label_segments(
    supervoxels: TensorTypeVar,
    path: str,
) -> TensorTypeVar:

    db = ts.KvStore.open(
        {
            "driver": "file",
            "path": path,
        }
    ).result()
    merge_history = pickle.loads(db["merge_history"])

    seg = extract_segments(
        supervoxels=supervoxels,
        merge_history=merge_history,
    )
    return tensor_ops.convert.astype(seg, supervoxels)


@builder.register("agglomerate_supervoxels_from_affinities")
@typeguard.typechecked
def agglomerate_supervoxels_from_affinities(
    affs: TensorTypeVar,
    supervoxels: TensorTypeVar,
    threshold: float,
    **kwargs,
) -> TensorTypeVar:
    """
    Agglomerate supevoxels using affinity graph. Convenience function that returns only seg.

    :param affs, supervoxels: Input tensors
    :param threshold: Threshold to agglomerate down to.
    :return: Relabeled `supervoxels`
    """
    seg, _ = run_agglomeration_aff(
        affs=affs,
        supervoxels=supervoxels,
        threshold=threshold,
        **kwargs,
    )
    return tensor_ops.convert.astype(seg, supervoxels)
