from typing import Sequence

from zetta_utils import builder
from zetta_utils.db_annotations.annotation import read_annotations
from zetta_utils.layer.volumetric.layer import VolumetricLayer


@builder.register("copy_annotated_data")
def CopyAnnotatedFlow(
    src: VolumetricLayer,
    dst: VolumetricLayer,
    collection_name: str,
    layer_group_name: str,
    resolution: Sequence[float],
):
    annotations = read_annotations(
        collection_ids=[collection_name], layer_group_ids=[layer_group_name]
    )
    breakpoint()
