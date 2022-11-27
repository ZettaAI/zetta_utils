"""neuroglancer state parsing."""

from os import environ
from typing import List, Union

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
RESOLUTION_KEY = "voxelSize"


def read_remote_annotations(layer_name: str) -> AnnotationLayer:
    logger.info(f"Remote layer: {remote_path}/{layer_name}.")

    cf = CloudFiles(remote_path)
    layer_json = cf.get_json(layer_name)
    logger.debug(layer_json)
    layer: AnnotationLayer = make_layer(layer_json)

    logger.info(f"Layer type: {layer.type}; Total: {len(layer.annotations)}.")
    return _parse_annotations(layer)


def _parse_annotations(layer: AnnotationLayer) -> List[Union[BoundingCube, Vec3D]]:
    result: List[Union[BoundingCube, Vec3D]] = []
    resolution: Vec3D = layer.to_json()[RESOLUTION_KEY]
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


def write_remote_annotations():
    ...
