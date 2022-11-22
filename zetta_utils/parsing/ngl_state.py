"""neuroglancer state parsing."""

from os import environ

from cloudfiles import CloudFiles
from neuroglancer.viewer_state import Layer
from neuroglancer.viewer_state import make_layer

from ..log import get_logger


logger = get_logger("zetta_utils")
remote_path = environ.get("REMOTE_LAYERS_PATH", "gs://remote-annotations")

def load(layer_name: str) -> Layer:
    logger.info(f"Remote layer: {remote_path}/{layer_name}.")

    cf = CloudFiles(remote_path)
    layer_json = cf.get_json(layer_name)
    logger.debug(layer_json)
    return make_layer(layer_json)

