from typing import Optional
import attrs
import mazepa
from typeguard import typechecked
from zetta_utils import builder, tensor_ops
from zetta_utils.typing import Vec3D
from zetta_utils.layer import Layer
from zetta_utils.layer.volumetric import VolumetricIndex, RawVolumetricIndex
from .. import LayerProcessor

@builder.register("ChunkedLayerProcessJob")
@mazepa.job
@typechecked
@attrs.frozen()
class ChunkedLayerProcessJob:
    # preserve_zeros: bool = False
    pass


@builder.register("ChunkedLayerProcessor")
@typechecked
@attrs.frozen()
class ChunkedLayerProcessor(LayerProcessor):
    chunker = None
    inner_processor = LayerProcessor
    exec_queue: Optional[mazepa.ExecutionQUeue] = None
    batch_gap_sleep_sec: float = 4.0
    max_batch_len: int = 10000
    default_state_constructor: Callable[..., mazepa.ExecutionState] = mazepa.InMemoryExecutionState

@builder.register("ChunkedProcessLayerJob")
@mazepa.job
@typechecked
@attrs.frozen()
class ChunkedProcessLayerJob(LayerProcessor):
    # preserve_zeros: bool = False

    def __call__(
        self,
        src: Layer[RawVolumetricIndex, VolumetricIndex],
        idx: VolumetricIndex,
        dst: Optional[Layer[RawVolumetricIndex, VolumetricIndex]] = None,
    ):
        if dst is None:
            dst = src
        data_src = src[idx]
        scale_factor = tuple(idx.resolution[i] / self.resolution[i] for i in range(3))
        data_dst = tensor_ops.interpolate(
            data=data_src,
            scale_factor=scale_factor,
            mode=self.mode,
            unsqueeze_input_to=5, # Only 3D data is allowed here -- no 2D!
        )
        idx_dst = VolumetricIndex(
            bcube=idx.bcube,
            resolution=self.resolution,
        )
        dst[idx_dst] = data_dst
