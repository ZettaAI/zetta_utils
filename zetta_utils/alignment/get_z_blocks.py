from typing import List

from zetta_utils import builder, log
from zetta_utils.bcube import BoundingCube

from ..parsing.ngl_state import get_bcubes_from_annotations
from ..parsing.ngl_state import load as load_ngl_layer

logger = log.get_logger("zetta_utils")


@builder.register("get_z_blocks")
def get_z_blocks(
    remote_layer: str,
) -> List[BoundingCube]:
    layer = load_ngl_layer(remote_layer)
    logger.info(f"Layer type: {layer.type}; Total: {len(layer.annotations)}.")

    result = get_bcubes_from_annotations(layer)
    logger.info(f"Final number of blocks: {len(result)}")
    return result
