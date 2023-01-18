import copy
from typing import Optional, Protocol

import attrs
import einops
import torch
import torchfields  # pylint: disable=unused-import

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.bcube import BoundingCube
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricIndexTranslator,
    VolumetricLayer,
)
from zetta_utils.mazepa_layer_processing.alignment.common import (
    translation_adjusted_download,
)
from zetta_utils.mazepa_layer_processing.common import build_chunked_apply_flow
from zetta_utils.typing import IntVec3D, Vec3D


class ComputeFieldFn(Protocol):
    def __call__(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_field: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ...


@builder.register(
    "ComputeFieldOperation", cast_to_vec3d=["crop_pad"], cast_to_intvec3d=["chunk_size"]
)
@mazepa.taskable_operation_cls
@attrs.mutable
class ComputeFieldOperation:
    fn: ComputeFieldFn
    crop_pad: IntVec3D = IntVec3D(0, 0, 0)
    res_change_mult: Vec3D = Vec3D(1, 1, 1)
    output_crop_px: IntVec3D = attrs.field(init=False)

    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        return dst_resolution / self.res_change_mult

    def __attrs_post_init__(self):
        output_crop_px = self.crop_pad / self.res_change_mult

        for e in output_crop_px:
            if not e.is_integer():
                raise ValueError(
                    f"Destination layer crop pad of {self.crop_pad} with resolution change "
                    f"multiplier of {self.res_change_mult} results in non-integer "
                    f"output crop of {output_crop_px}."
                )
        self.output_crop_px = IntVec3D(*(int(e) for e in output_crop_px))

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        src: VolumetricLayer,
        tgt: VolumetricLayer,
        src_field: Optional[VolumetricLayer],
        tgt_field: Optional[VolumetricLayer],
    ):
        idx_input = copy.deepcopy(idx)
        idx_input.resolution = self.get_input_resolution(idx.resolution)
        idx_input_padded = idx_input.pad(self.crop_pad)
        src_data, src_field_data, src_translation = translation_adjusted_download(
            src=src,
            field=src_field,
            idx=idx_input_padded,
        )
        tgt_data, tgt_field_data, _ = translation_adjusted_download(
            src=tgt, field=tgt_field, idx=idx_input_padded
        )

        if tgt_field_data is not None:
            tgt_field_data_zcxy = einops.rearrange(tgt_field_data, "C X Y Z -> Z C X Y")
            tgt_data_zcxy = einops.rearrange(tgt_data, "C X Y Z -> Z C X Y")
            tgt_nonz_zcxy = tgt_data_zcxy != 0
            tgt_data_warped = tgt_field_data_zcxy.field().from_pixels()( # type: ignore
                tgt_data_zcxy
            )
            tgt_nonz_warped = tgt_field_data_zcxy.field().from_pixels()( # type: ignore
                tgt_nonz_zcxy.float()
            )
            tgt_data_warped[tgt_nonz_warped < 0.1] = 0
            tgt_data_final = einops.rearrange(
                tgt_data_warped,
                "Z C X Y -> C X Y Z",
            )
        else:
            tgt_data_final = tgt_data
        result_raw = self.fn(
            src=src_data,
            tgt=tgt_data_final,
            src_field=src_field_data,
        )
        result = tensor_ops.crop(result_raw, crop=self.output_crop_px)

        result[0] += src_translation[0]
        result[1] += src_translation[1]
        dst[idx] = result


@builder.register(
    "ComputeFieldFlowSchema",
    cast_to_vec3d=["src_offset", "tgt_offset"],
    cast_to_intvec3d=["chunk_size"],
)
@mazepa.flow_schema_cls
@attrs.mutable
class ComputeFieldFlowSchema:
    chunk_size: IntVec3D
    operation: ComputeFieldOperation
    chunker: VolumetricIndexChunker = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.chunker = VolumetricIndexChunker(chunk_size=self.chunk_size)

    def flow(
        self,
        bcube: BoundingCube,
        dst_resolution: Vec3D,
        dst: VolumetricLayer,
        src: VolumetricLayer,
        tgt: Optional[VolumetricLayer] = None,
        src_field: Optional[VolumetricLayer] = None,
        tgt_field: Optional[VolumetricLayer] = None,
        tgt_offset: Vec3D = Vec3D(0, 0, 0),
        src_offset: Vec3D = Vec3D(0, 0, 0),
    ):
        if tgt is None:
            tgt = src

        tgt = copy.deepcopy(tgt)
        input_resolution = self.operation.get_input_resolution(dst_resolution)
        tgt.index_adjs.insert(
            0, VolumetricIndexTranslator(offset=tgt_offset, resolution=input_resolution)
        )

        src = copy.deepcopy(src)
        src.index_adjs.insert(
            0, VolumetricIndexTranslator(offset=src_offset, resolution=input_resolution)
        )

        if tgt_field is not None:
            tgt_field = copy.deepcopy(tgt_field)
            tgt_field.index_adjs.insert(
                0,
                VolumetricIndexTranslator(offset=tgt_offset, resolution=input_resolution),
            )

        if src_field is not None:
            src_field = copy.deepcopy(src_field)
            src_field.index_adjs.insert(
                0,
                VolumetricIndexTranslator(offset=src_offset, resolution=input_resolution),
            )
        # breakpoint()
        cf_flow = build_chunked_apply_flow(
            operation=self.operation,  # type: ignore
            chunker=self.chunker,
            idx=VolumetricIndex(bcube=bcube, resolution=dst_resolution),
            dst=dst,  # type: ignore
            src=src,  # type: ignore
            tgt=tgt,  # type: ignore
            src_field=src_field,  # type: ignore
            tgt_field=tgt_field,  # type: ignore
        )

        yield cf_flow


@builder.register(
    "build_compute_field_flow",
    cast_to_vec3d=["dst_resolution", "tgt_offset", "src_offset"],
    cast_to_intvec3d=["chunk_size"],
)
def build_compute_field_flow(
    chunk_size: IntVec3D,
    operation: ComputeFieldOperation,
    bcube: BoundingCube,
    dst_resolution: Vec3D,
    dst: VolumetricLayer,
    src: VolumetricLayer,
    tgt: Optional[VolumetricLayer] = None,
    src_field: Optional[VolumetricLayer] = None,
    tgt_offset: Vec3D = Vec3D(0, 0, 0),
    src_offset: Vec3D = Vec3D(0, 0, 0),
) -> mazepa.Flow:
    flow_schema = ComputeFieldFlowSchema(chunk_size=chunk_size, operation=operation)
    flow = flow_schema(
        bcube=bcube,
        dst_resolution=dst_resolution,
        dst=dst,
        src=src,
        tgt=tgt,
        src_field=src_field,
        tgt_offset=tgt_offset,
        src_offset=src_offset,
    )
    return flow
