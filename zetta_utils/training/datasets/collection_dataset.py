from typing import Sequence

from neuroglancer.viewer_state import AxisAlignedBoundingBoxAnnotation
from typeguard import typechecked

from zetta_utils import builder, db_annotations
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.geometry.vec import Vec3D
from zetta_utils.layer.layer_base import Layer
from zetta_utils.layer.layer_set.build import build_layer_set
from zetta_utils.layer.tools_base import DataProcessor
from zetta_utils.layer.volumetric.cloudvol.build import build_cv_layer
from zetta_utils.layer.volumetric.layer import VolumetricLayer
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
    base_resolution: Sequence[float],
    layer_rename_map: dict[str, str],
    shared_read_procs: Sequence[DataProcessor],
    per_layer_read_procs: dict[str, Sequence[DataProcessor]] ,
) -> JointDataset:

    datasets = {}
    annotations = db_annotations.read_annotations(collection_ids=[collection_name])
    layer_group_map: dict[str, dict[str, Layer]]= {}

    for i, annotation in enumerate(annotations.values()):
        if annotation.layer_group_id not in layer_group_map:
            layer_group = db_annotations.read_layer_group(annotation.layer_group_id)
            db_layers = db_annotations.read_layers(layer_ids=layer_group.layers)
            layers = {}
            for layer in db_layers:
                name = layer.name
                if name in layer_rename_map:
                    name = layer_rename_map[name]
                read_procs = per_layer_read_procs[name]
                layers[name] = build_cv_layer(path=layer.source, read_procs=read_procs)
            layer_group_map[annotation.layer_group_id] = layers
        else:
            layers = layer_group_map[annotation.layer_group_id]

        if isinstance(annotation.ng_annotation, AxisAlignedBoundingBoxAnnotation):
            bbox = BBox3D.from_ng_bbox(
                ng_bbox=annotation.ng_annotation,
                base_resolution=base_resolution
            ).snapped([0, 0, 0], resolution, "shrink")
            datasets[str(i)] = (
                LayerDataset(
                    layer=build_layer_set(
                        layers=layers,
                        read_procs=shared_read_procs
                    ),
                    sample_indexer=VolumetricStridedIndexer(
                        resolution=resolution,
                        chunk_size=chunk_size,
                        stride=chunk_stride,
                        mode="shrink",
                        bbox=bbox,
                    )

                )
            )
    dset = JointDataset(mode="vertical", datasets=datasets)
    return dset
