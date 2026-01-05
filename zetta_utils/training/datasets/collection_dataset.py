import json
import os
from typing import Sequence

import fsspec
from neuroglancer.viewer_state import AxisAlignedBoundingBoxAnnotation
from typeguard import typechecked

from zetta_utils import builder, db_annotations
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.layer.layer_set.build import build_layer_set
from zetta_utils.layer.tools_base import DataProcessor
from zetta_utils.layer.volumetric.cloudvol.build import build_cv_layer
from zetta_utils.layer.volumetric.layer import VolumetricLayer
from zetta_utils.training.datasets.joint_dataset import JointDataset
from zetta_utils.training.datasets.layer_dataset import LayerDataset
from zetta_utils.training.datasets.sample_indexers.volumetric_strided_indexer import (
    VolumetricStridedIndexer,
)


def _get_z_resolution(layers: dict[str, VolumetricLayer]) -> float:
    z_resolutions = {}
    for layer_name, layer in layers.items():
        layer_path = layer.backend.name.removeprefix("precomputed://")
        layer_path = layer_path.removesuffix("|neuroglancer-precomputed:")
        info_path = os.path.join(layer_path.strip("/"), "info")
        with fsspec.open(info_path) as f:
            info = json.loads(f.read())
            z_resolutions[layer_name] = {e["resolution"][-1] for e in info["scales"]}
    result = min(set.intersection(*z_resolutions.values()))
    return result


@builder.register("build_collection_dataset")
@typechecked
def build_collection_dataset(  # pylint: disable=too-many-locals
    collection_name: str,
    resolution: Sequence[float],
    chunk_size: Sequence[int],
    chunk_stride: Sequence[int],
    layer_rename_map: dict[str, str],
    per_layer_read_procs: dict[str, Sequence[DataProcessor]] | None = None,
    shared_read_procs: Sequence[DataProcessor] = tuple(),
    tags: list[str] | None = None,
    flexible_z: bool = True,
) -> JointDataset:
    datasets = {}
    annotations = db_annotations.read_annotations(
        collection_ids=[collection_name], tags=tags, union=False
    )
    # layer group->layer_name->layer
    layer_group_map: dict[str, dict[str, VolumetricLayer]] = {}

    per_layer_read_procs_dict = {}
    if per_layer_read_procs is not None:
        per_layer_read_procs_dict = per_layer_read_procs

    for i, annotation in enumerate(annotations.values()):
        if annotation.layer_group not in layer_group_map:
            layer_group = db_annotations.read_layer_group(annotation.layer_group)
            db_layers = db_annotations.read_layers(layer_ids=layer_group.layers)
            layers = {}
            for layer in db_layers:
                name = layer.name
                if name in layer_rename_map:
                    name = layer_rename_map[name]
                read_procs = per_layer_read_procs_dict.get(name, [])
                layers[name] = build_cv_layer(
                    path=layer.source, read_procs=read_procs, allow_slice_rounding=True
                )
            layer_group_map[annotation.layer_group] = layers
        else:
            layers = layer_group_map[annotation.layer_group]

        z_resolution = resolution[-1]
        if flexible_z:
            z_resolution = _get_z_resolution(layers)

        this_resolution = [resolution[0], resolution[1], z_resolution]
        if isinstance(annotation.ng_annotation, AxisAlignedBoundingBoxAnnotation):
            bbox = BBox3D.from_ng_bbox(annotation.ng_annotation, (1, 1, 1)).snapped(
                (0, 0, 0), this_resolution, "shrink"
            )

            this_dset = LayerDataset(
                layer=build_layer_set(layers=layers, read_procs=shared_read_procs),
                sample_indexer=VolumetricStridedIndexer(
                    resolution=this_resolution,
                    chunk_size=chunk_size,
                    stride=chunk_stride,
                    mode="shrink",
                    bbox=bbox,
                ),
            )
            if len(this_dset) == 0:
                raise RuntimeError(
                    f"The following annotation indicates bounding box {bbox} "
                    f"which is smaller than the chunk size {chunk_size}"
                )
            datasets[str(i)] = this_dset
    dset = JointDataset(mode="vertical", datasets=datasets)
    return dset
