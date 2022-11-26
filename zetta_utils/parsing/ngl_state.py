"""neuroglancer state parsing."""

from os import environ
from typing import List

from neuroglancer.viewer_state import AnnotationLayer

from ..bcube import BoundingCube
from ..log import get_logger


logger = get_logger("zetta_utils")
remote_path = environ.get("REMOTE_LAYERS_PATH", "gs://remote-annotations")
RESOLUTION_KEY = "voxelSize"


def load(layer_name: str) -> AnnotationLayer:
    from cloudfiles import CloudFiles
    from neuroglancer.viewer_state import make_layer

    logger.info(f"Remote layer: {remote_path}/{layer_name}.")

    cf = CloudFiles(remote_path)
    layer_json = cf.get_json(layer_name)
    logger.debug(layer_json)
    return make_layer(layer_json)


def get_bcubes_from_annotations(layer: AnnotationLayer) -> List[BoundingCube]:
    from ..typing import Vec3D

    bcubes = []
    resolution: Vec3D = layer.to_json()[RESOLUTION_KEY]
    for annotation in layer.annotations:
        try:
            bcube = BoundingCube.from_coords(
                annotation.point_a,
                annotation.point_b,
                resolution=resolution,
            )
            bcubes.append(bcube)
        except AttributeError:
            ...
    return bcubes
