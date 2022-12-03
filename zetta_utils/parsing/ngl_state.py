"""neuroglancer state parsing."""

from enum import Enum
from enum import Enum
from os import environ
from typing import Dict, List, Union

import numpy as np
import numpy as np
from cloudfiles import CloudFiles
from neuroglancer.viewer_state import (
    AnnotationLayer,
    AxisAlignedBoundingBoxAnnotation,
    PointAnnotation,
    make_layer,
)

from zetta_utils.bcube import BoundingCube
from zetta_utils.bcube.bcube import BoundingBoxND
from zetta_utils.bcube.bcube import BoundingBoxND
from zetta_utils.log import get_logger
from zetta_utils.typing import Vec3D

logger = get_logger("zetta_utils")
remote_path = environ.get("REMOTE_LAYERS_PATH", "gs://remote-annotations")


class NglLayerKeys(Enum):
    ANNOTATION_COLOR = "annotationColor"
    ANNOTATIONS = "annotations"
    NAME = "name"
    RESOLUTION = "voxelSize"
    TOOL = "tool"
    TYPE = "type"


class AnnotationKeys(Enum):
    ID = "id"
    POINT = "point"
    POINT_A = "pointA"
    POINT_B = "pointB"
    TYPE = "type"


class DefaultLayerValues(Enum):
    COLOR = "#ff0000"
    TYPE = "annotation"
    TOOL = "annotateBoundingBox"


def load(layer_name: str) -> AnnotationLayer:
    from cloudfiles import CloudFiles
    from neuroglancer.viewer_state import make_layer


class NGL_LAYER_KEYS(Enum):
    ANNOTATION_COLOR = "annotationColor"
    ANNOTATIONS = "annotations"
    NAME = "name"
    RESOLUTION = "voxelSize"
    TOOL = "tool"
    TYPE = "type"


class ANNOTATION_KEYS(Enum):
    ID = "id"
    POINT = "point"
    POINT_A = "pointA"
    POINT_B = "pointB"
    TYPE = "type"


class DEFAULT_LAYER_VALUES(Enum):
    COLOR = "#ff0000"
    TYPE = "annotation"
    TOOL = "annotateBoundingBox"


def load(layer_name: str) -> AnnotationLayer:
    from cloudfiles import CloudFiles
    from neuroglancer.viewer_state import make_layer


class NGL_LAYER_KEYS(Enum):
    ANNOTATION_COLOR = "annotationColor"
    ANNOTATIONS = "annotations"
    NAME = "name"
    RESOLUTION = "voxelSize"
    TOOL = "tool"
    TYPE = "type"


class ANNOTATION_KEYS(Enum):
    ID = "id"
    POINT = "point"
    POINT_A = "pointA"
    POINT_B = "pointB"
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
    resolution: Vec3D = layer.to_json()[NglLayerKeys.RESOLUTION.value]
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
            point: Vec3D = annotation.point * resolution
            point: Vec3D = annotation.point * resolution
            result.append(point)
    return result


def write_remote_annotations(
    layer_name: str,
    resolution: Vec3D,
    bcubes_or_points: List[Union[BoundingCube, Vec3D]],
) -> None:
    annotations: List[Dict] = []
    layer_d = {
        NglLayerKeys.NAME.value: layer_name,
        NglLayerKeys.RESOLUTION.value: resolution,
        NglLayerKeys.TOOL.value: DefaultLayerValues.TOOL.value,
        NglLayerKeys.TYPE.value: DefaultLayerValues.TYPE.value,
        NglLayerKeys.ANNOTATION_COLOR.value: DefaultLayerValues.COLOR.value,
        NglLayerKeys.ANNOTATIONS.value: annotations,
    }

    for i, bcubes_or_point in enumerate(bcubes_or_points):
        if isinstance(bcubes_or_point, BoundingBoxND):
            x, y, z = bcubes_or_point.bounds
            point_a = np.array([x[0], y[0], z[0]])
            point_b = np.array([x[1], y[1], z[1]])
            annotation = {
                AnnotationKeys.ID.value: str(i),
                AnnotationKeys.POINT_A.value: point_a / resolution,
                AnnotationKeys.POINT_B.value: point_b / resolution,
                AnnotationKeys.TYPE.value: AxisAlignedBoundingBoxAnnotation().type,
            }
            annotations.append(annotation)
        else:
            annotation = {
                AnnotationKeys.ID.value: str(i),
                AnnotationKeys.POINT.value: np.array(bcubes_or_point) / resolution,
                AnnotationKeys.TYPE.value: PointAnnotation().type,
            }
            annotations.append(annotation)
    layer = make_layer(layer_d)
    cf = CloudFiles(remote_path)
    cf.put_json(layer_name, layer.to_json())
