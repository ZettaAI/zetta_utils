from typing import List

from zetta_utils import builder, log
from zetta_utils.bcube import BoundingCube
from ..parsing.ngl_state import load as load_ngl_layer

logger = log.get_logger("zetta_utils")


@builder.register("get_z_blocks")
def get_z_blocks(
    bcube: BoundingCube,
    block_size: int,
    block_size_res: float,
) -> List[BoundingCube]:
    logger.info(f"Breaking bcube {bcube} into Z blocks.")
    logger.info(f"Block size {block_size}, block size resolution {block_size_res}.")

    layer = load_ngl_layer("remote/zfish_10132022_cutout_misalignments")
    logger.info(f"Layer type: {layer.type}; Total: {len(layer.annotations)}.")

    z_start = bcube.bounds[-1][0]
    z_end = bcube.bounds[-1][1]
    result = []  # type: List[BoundingCube]
    interval = block_size * block_size_res

    curr_start = z_start
    while curr_start < z_end:
        curr_end = min(z_end, curr_start + interval)
        block_bcube = BoundingCube(
            bounds=[bcube.bounds[0], bcube.bounds[1], (curr_start, curr_end)]
        )
        logger.info(f"Adding block with bcube {block_bcube}")
        result.append(block_bcube)
        curr_start = curr_end

    logger.info(f"Final number of blocks: {len(result)}")
    return result
