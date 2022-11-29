"""neuroglancer state parsing."""

from enum import Enum
from os import environ
from typing import List, Union

import numpy as np
from cloudfiles import CloudFiles
from neuroglancer.viewer_state import (
    AnnotationLayer,
    AxisAlignedBoundingBoxAnnotation,
    PointAnnotation,
    make_layer,
)

from zetta_utils.bcube import BoundingCube
from zetta_utils.log import get_logger
from zetta_utils.typing import Vec3D

logger = get_logger("zetta_utils")
remote_path = environ.get("REMOTE_LAYERS_PATH", "gs://remote-annotations")


class NGL_LAYER_KEYS(Enum):
    ANNOTATION_COLOR = "annotationColor"
    ANNOTATIONS = "annotations"
    NAME = "name"
    RESOLUTION = "voxelSize"
    TOOL = "tool"
    TYPE = "type"


class DEFAULT_LAYER_VALUES(Enum):
    COLOR = "#ff0000"
    TYPE = "annotation"
    TOOL = "annotateBoundingBox"


def read_remote_annotations(layer_name: str) -> List[Union[BoundingCube, Vec3D]]:
    logger.info(f"Remote layer: {remote_path}/{layer_name}.")

    cf = CloudFiles(remote_path)
    layer_json = cf.get_json(layer_name)
    logger.debug(layer_json)
    layer: AnnotationLayer = make_layer(layer_json)

    logger.info(f"Layer type: {layer.type}; Total: {len(layer.annotations)}.")
    return _parse_annotations(layer)


def _parse_annotations(layer: AnnotationLayer) -> List[Union[BoundingCube, Vec3D]]:
    result: List[Union[BoundingCube, Vec3D]] = []
    resolution: Vec3D = layer.to_json()[NGL_LAYER_KEYS.RESOLUTION.value]
    for annotation in layer.annotations:
        assert isinstance(
            annotation, (AxisAlignedBoundingBoxAnnotation, PointAnnotation)
        ), "Annotation type not supported."
        try:
            bcube = BoundingCube.from_coords(
                annotation.point_a,
                annotation.point_b,
                resolution=resolution,
            )
            result.append(bcube)
        except AttributeError:
            point: Vec3D = annotation.point
            result.append(point)
    return result


def write_remote_annotations(
    layer_name: str,
    resolution: Vec3D,
    bcubes_or_points: List[Union[BoundingCube, Vec3D]],
) -> None:
    layer = {
        NGL_LAYER_KEYS.NAME.value: layer_name,
        NGL_LAYER_KEYS.RESOLUTION.value: resolution,
        NGL_LAYER_KEYS.TOOL.value: DEFAULT_LAYER_VALUES.TOOL.value,
        NGL_LAYER_KEYS.TYPE.value: DEFAULT_LAYER_VALUES.TYPE.value,
        NGL_LAYER_KEYS.ANNOTATION_COLOR.value: DEFAULT_LAYER_VALUES.COLOR.value,
    }
    annotations: List[Union[AxisAlignedBoundingBoxAnnotation, PointAnnotation]] = []

    res = np.array(resolution)

    for bcubes_or_point in bcubes_or_points:
        try:
            bcube: BoundingCube = bcubes_or_point
            x,y,z = bcube.bounds
            point_a = np.array([x[0], y[0], z[0]])
            point_b = np.array([x[1], y[1], z[1]])
        except AttributeError:
            ...
