from __future__ import annotations

from typing import Sequence, Union

import attrs

from zetta_utils import builder, mazepa
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric.annotation.backend import (
    Annotation,
    AnnotationLayerBackend,
)
from zetta_utils.layer.volumetric.index import VolumetricIndex

from ... import DataProcessor, IndexProcessor, JointIndexDataProcessor, Layer
from ..conversion import UserVolumetricIndex, convert_idx

AnnotationDataProcT = Union[
    DataProcessor[Sequence[Annotation]],
    JointIndexDataProcessor[Sequence[Annotation], VolumetricIndex],
]
AnnotationDataWriteProcT = Union[
    DataProcessor[Sequence[Annotation]],
    JointIndexDataProcessor[Sequence[Annotation], VolumetricIndex],
]


@attrs.frozen
class VolumetricAnnotationLayer(
    Layer[VolumetricIndex, Sequence[Annotation], Sequence[Annotation]]
):
    backend: AnnotationLayerBackend
    index_resolution: Vec3D | None = None
    default_desired_resolution: Vec3D | None = None
    allow_slice_rounding: bool = False
    readonly: bool = False

    index_procs: tuple[IndexProcessor[VolumetricIndex], ...] = ()
    read_procs: tuple[AnnotationDataProcT, ...] = ()
    write_procs: tuple[AnnotationDataWriteProcT, ...] = ()

    def _convert_annotations(
        self, annotations: Sequence[Annotation], from_res: Vec3D, to_res: Vec3D
    ) -> list[Annotation]:
        return [
            annotation.with_converted_coordinates(from_res=from_res, to_res=to_res)
            for annotation in annotations
        ]

    def __getitem__(self, idx: UserVolumetricIndex) -> Sequence[Annotation]:
        idx_backend = convert_idx(
            idx,
            self.index_resolution,
            self.default_desired_resolution,
            self.allow_slice_rounding,
        )

        annotations = self.read_with_procs(idx=idx_backend)

        backend_index = self.backend.index
        assert backend_index is not None, "Backend index should be initialized"
        return self._convert_annotations(
            annotations, from_res=backend_index.resolution, to_res=idx_backend.resolution
        )

    def __setitem__(self, idx: UserVolumetricIndex, data: Sequence[Annotation]):
        idx_backend = convert_idx(
            idx,
            self.index_resolution,
            self.default_desired_resolution,
            self.allow_slice_rounding,
        )

        self.write_with_procs(idx=idx_backend, data=data)

    def pformat(self) -> str:  # pragma: no cover
        return self.backend.pformat()

    def with_changes(
        self,
        **kwargs,
    ):
        return attrs.evolve(self, **kwargs)  # pragma: no cover


@mazepa.taskable_operation
def post_process_annotation_layer_op(target: VolumetricAnnotationLayer):  # pragma: no cover
    target.backend.post_process()


@builder.register("post_process_annotation_layer_flow")
@mazepa.flow_schema
def post_process_annotation_layer_flow(target: VolumetricAnnotationLayer):  # pragma: no cover
    yield post_process_annotation_layer_op.make_task(target)
