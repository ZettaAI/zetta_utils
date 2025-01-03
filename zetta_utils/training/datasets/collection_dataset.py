import json
import os
from collections import defaultdict
from typing import Sequence

import fsspec
from neuroglancer.viewer_state import AxisAlignedBoundingBoxAnnotation
from typeguard import typechecked

from zetta_utils import builder, db_annotations
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.geometry.vec import Vec3D
from zetta_utils.layer.layer_base import Layer
from zetta_utils.layer.layer_set.build import build_layer_set
from zetta_utils.layer.tools_base import DataProcessor
from zetta_utils.layer.volumetric.cloudvol.build import build_cv_layer
from zetta_utils.training.datasets.joint_dataset import JointDataset
from zetta_utils.training.datasets.layer_dataset import LayerDataset
from zetta_utils.training.datasets.sample_indexers.volumetric_strided_indexer import (
    VolumetricStridedIndexer,
)


@builder.register("build_collection_dataset")
@typechecked
def build_collection_dataset(
    collection_name: str,
    resolution: Sequence[float],
    chunk_size: Sequence[int],
    chunk_stride: Sequence[int],
    layer_rename_map: dict[str, str],
    per_layer_read_procs: dict[str, Sequence[DataProcessor]] | None = None,
    shared_read_procs: Sequence[DataProcessor] = tuple(),
    tags: list[str] | None = None,
) -> JointDataset:
    datasets = {}
    annotations = db_annotations.read_annotations(
        collection_ids=[collection_name], tags=tags, union=False
    )
    layer_group_map: dict[str, dict[str, Layer]] = {}
    if per_layer_read_procs is None:
        per_layer_read_procs_dict = defaultdict(tuple)
    else:
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
                read_procs = per_layer_read_procs_dict[name]
                layers[name] = build_cv_layer(path=layer.source, read_procs=read_procs)
            layer_group_map[annotation.layer_group] = layers
        else:
            layers = layer_group_map[annotation.layer_group]

        z_resolution = resolution[-1]
        for layer in layers.values():
            info_path = os.path.join(layer.backend.name.strip("precomputed://"), "info")
            with fsspec.open(info_path) as f:
                info = json.loads(f.read())
                z_resolutions = {e["resolution"][-1] for e in info["scales"]}
                if len(z_resolutions) != 1:
                    raise RuntimeError("Only layers with single z resolution are supported")
                z_resolution = list(z_resolutions)[0]
        this_resolution = [resolution[0], resolution[1], z_resolution]
        if isinstance(annotation.ng_annotation, AxisAlignedBoundingBoxAnnotation):
            ng_bbox = annotation.ng_annotation
            point_a_nm = Vec3D(*ng_bbox.pointA).int()
            point_b_nm = Vec3D(*ng_bbox.pointB).int()

            start_coord = [
                int((min(point_a_nm[i], point_b_nm[i]) / this_resolution[i])) * this_resolution[i]
                for i in range(3)
            ]
            end_coord = [
                int(max(point_a_nm[i], point_b_nm[i]) / this_resolution[i]) * this_resolution[i]
                for i in range(3)
            ]
            bbox = BBox3D.from_coords(start_coord=start_coord, end_coord=end_coord)
            datasets[str(i)] = LayerDataset(
                layer=build_layer_set(layers=layers, read_procs=shared_read_procs),
                sample_indexer=VolumetricStridedIndexer(
                    resolution=this_resolution,
                    chunk_size=chunk_size,
                    stride=chunk_stride,
                    mode="shrink",
                    bbox=bbox,
                ),
            )
    dset = JointDataset(mode="vertical", datasets=datasets)
    return dset
