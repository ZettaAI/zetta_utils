from __future__ import annotations

from typing import Sequence, Union, cast

import attrs
import tqdm
from neuroglancer.viewer_state import AxisAlignedBoundingBoxAnnotation
from numpy import typing as npt

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.common import ComparablePartial
from zetta_utils.db_annotations.annotation import read_annotations
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric.index import VolumetricIndex
from zetta_utils.layer.volumetric.layer import VolumetricLayer
from zetta_utils.layer.volumetric.protocols import VolumetricBasedLayerProtocol
from zetta_utils.mazepa.flows import sequential_flow
from zetta_utils.mazepa_layer_processing.common.subchunkable_apply_flow import (
    build_subchunkable_apply_flow,
)


@mazepa.taskable_operation_cls
@attrs.frozen
class CopyAnnotatedOp:
    def __call__(
        self,
        idx: VolumetricIndex,
        src: VolumetricLayer,
        dst: VolumetricLayer,
    ):
        dst[idx] = src[idx]


@builder.register("build_copy_annotated_flow")
def build_copy_annotated_flow(
    src: VolumetricBasedLayerProtocol,
    dst: VolumetricBasedLayerProtocol,
    collection_name: str,
    layer_group_name: str,
    resolution: Sequence[float],
    base_resolution: Sequence[float],
    # processing_chunk_size: Sequence[int],
) -> mazepa.Flow:
    annotations = read_annotations(
        collection_ids=[collection_name], layer_group_ids=[layer_group_name]
    )
    bboxes = [
        BBox3D.from_ng_bbox(
            ng_bbox=annotation.ng_annotation, base_resolution=base_resolution
        ).snapped([0, 0, 0], resolution, "shrink")
        for annotation in annotations.values()
        if isinstance(annotation.ng_annotation, AxisAlignedBoundingBoxAnnotation)
    ]
    resolution = Vec3D(*resolution)
    # TODO: speed this up later
    for bbox in tqdm.tqdm(bboxes):
        dst[resolution, bbox] = src[resolution, bbox]
    result = sequential_flow(stages=[])
    return result
