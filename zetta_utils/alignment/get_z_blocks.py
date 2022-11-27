from typing import List, Union

from zetta_utils import builder, log
from zetta_utils.bcube import BoundingCube
from zetta_utils.parsing import ngl_state
from zetta_utils.typing import Vec3D

logger = log.get_logger("zetta_utils")


@builder.register("get_z_blocks")
def get_z_blocks(
    remote_layer: str
) -> List[Union[BoundingCube, Vec3D]]:
    result = ngl_state.read_remote_annotations(remote_layer)
    logger.info(f"Final number of blocks/points: {len(result)}")
    return result
