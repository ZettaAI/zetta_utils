from typing import Any, Dict, Callable
import copy
import attrs
import mazepa

from zetta_utils.typing import Vec3D
from zetta_utils.layer import Layer
from zetta_utils.layer.volumetric import VolumetricIndex
from .. import LayerProcessor


@mazepa.task_maker
@attrs.mutable
class ComputeFieldProcessor(LayerProcessor):
    compute_field_method: Callable
    tgt_offset: Vec3D

    def __call__(self, layers: Dict[str, Layer[Any, VolumetricIndex]], idx: VolumetricIndex):
        src_data = layers["src"][idx]
        tgt_idx = copy.deepcopy(idx)
        tgt_idx.bcube.translate(
            offset=self.tgt_offset,
            resolution=tgt_idx.resolution,
        )
        tgt_data = layers["tgt"][idx]
        result = self.compute_field_method(src_data, tgt_data)
        layers["dst"][idx] = result
