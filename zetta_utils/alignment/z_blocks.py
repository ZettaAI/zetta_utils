from typing import List, Union

from zetta_utils import builder, log
from zetta_utils.bcube import BoundingCube
from zetta_utils.parsing import ngl_state
from zetta_utils.typing import Vec3D

logger = log.get_logger("zetta_utils")


@builder.register("get_z_blocks")
def get_z_blocks(remote_layer: str) -> List[Union[BoundingCube, Vec3D]]:
    result = ngl_state.read_remote_annotations(remote_layer)
    logger.info(f"Number of bcubes/points: {len(result)}")
    return result


# TODO: Handle case where bcubes_or_points takes in a BoundingCube
@builder.register("put_z_blocks", cast_to_vec3d=["bcubes_or_points"])
def put_z_blocks(
    remote_layer: str,
    resolution: Vec3D,
    bcubes_or_points: List[Union[BoundingCube, Vec3D]],
) -> None:
    ngl_state.write_remote_annotations(remote_layer, resolution, bcubes_or_points)
