from __future__ import annotations

from typing import Literal, Optional, Sequence

import attrs

from zetta_utils import alignment, builder, mazepa, tensor_ops
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricLayer,
    VolumetricLayerSet,
)
from zetta_utils.mazepa.semaphores import semaphore

from ..common import build_chunked_volumetric_callable_flow_schema


@builder.register("build_get_match_offsets_naive_flow")
def build_get_match_offsets_naive_flow(
    chunk_size: Sequence[int],
    bbox: BBox3D,
    dst_resolution: Sequence[float],
    non_tissue: VolumetricLayer,
    dst: VolumetricLayer,
    misalignment_mask_zm1: VolumetricLayer,
    misalignment_mask_zm2: Optional[VolumetricLayer] = None,
    misalignment_mask_zm3: Optional[VolumetricLayer] = None,
) -> mazepa.Flow:
    flow_schema = build_chunked_volumetric_callable_flow_schema(
        fn=alignment.aced_relaxation.get_aced_match_offsets_naive,
        chunker=VolumetricIndexChunker(chunk_size=Vec3D[int](*chunk_size)),
    )
    flow = flow_schema(
        idx=VolumetricIndex(bbox=bbox, resolution=Vec3D[float](*dst_resolution)),
        non_tissue=non_tissue,
        dst=dst,
        misalignment_mask_zm1=misalignment_mask_zm1,
        misalignment_mask_zm2=misalignment_mask_zm2,
        misalignment_mask_zm3=misalignment_mask_zm3,
    )
    return flow


@builder.register("AcedMatchOffsetOp")
@mazepa.taskable_operation_cls
@attrs.frozen
class AcedMatchOffsetOp:
    crop_pad: Sequence[int] = (0, 0, 0)

    def get_input_resolution(  # pylint: disable=no-self-use
        self, dst_resolution: Vec3D[float]
    ) -> Vec3D[float]:
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> AcedMatchOffsetOp:
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayerSet,
        tissue_mask: VolumetricLayer,
        misalignment_masks: dict[str, VolumetricLayer],
        pairwise_fields: dict[str, VolumetricLayer],
        pairwise_fields_inv: dict[str, VolumetricLayer],
        max_dist: int,
    ):
        idx_padded = idx.padded(self.crop_pad)
        with semaphore("read"):
            tissue_mask_data = tissue_mask[idx_padded]
        with semaphore("cuda"):
            if (tissue_mask_data != 0).sum() > 0:
                result = alignment.aced_relaxation.get_aced_match_offsets(
                    tissue_mask=tissue_mask_data,
                    misalignment_masks={k: v[idx_padded] for k, v in misalignment_masks.items()},
                    pairwise_fields={k: v[idx_padded] for k, v in pairwise_fields.items()},
                    pairwise_fields_inv={k: v[idx_padded] for k, v in pairwise_fields_inv.items()},
                    max_dist=max_dist,
                )
            result = {k: tensor_ops.crop(v, self.crop_pad) for k, v in result.items()}
        with semaphore("write"):
            dst[idx] = result


@builder.register("AcedRelaxationOp")
@mazepa.taskable_operation_cls
@attrs.frozen
class AcedRelaxationOp:
    crop_pad: Sequence[int] = (0, 0, 0)

    def get_input_resolution(  # pylint: disable=no-self-use
        self, dst_resolution: Vec3D[float]
    ) -> Vec3D[float]:
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> AcedRelaxationOp:
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        match_offsets: VolumetricLayer,
        pfields: dict[str, VolumetricLayer],
        rigidity_masks: VolumetricLayer | None = None,
        first_section_fix_field: VolumetricLayer | None = None,
        last_section_fix_field: VolumetricLayer | None = None,
        num_iter: int = 100,
        lr: float = 0.3,
        rigidity_weight: float = 10.0,
        fix: Literal["first", "last", "both"] | None = "first",
    ):
        idx_padded = idx.padded(self.crop_pad)
        first_section_idx_padded = idx_padded.translated_end((0, 0, 1 - idx_padded.shape[-1]))
        last_section_idx_padded = idx_padded.translated_start((0, 0, idx_padded.shape[-1] - 1))

        with semaphore("read"):
            match_offsets_data = match_offsets[idx_padded]

        with semaphore("cuda"):
            if (match_offsets_data != 0).sum() > 0:
                result = alignment.aced_relaxation.perform_aced_relaxation(
                    match_offsets=match_offsets_data,
                    pfields={k: v[idx_padded] for k, v in pfields.items()},
                    rigidity_masks=rigidity_masks[idx_padded] if rigidity_masks else None,
                    first_section_fix_field=(
                        first_section_fix_field[first_section_idx_padded]
                        if first_section_fix_field
                        else None
                    ),
                    last_section_fix_field=(
                        last_section_fix_field[last_section_idx_padded]
                        if last_section_fix_field
                        else None
                    ),
                    num_iter=num_iter,
                    lr=lr,
                    rigidity_weight=rigidity_weight,
                    fix=fix,
                )
                result_cropped = tensor_ops.crop(result, self.crop_pad)

        with semaphore("write"):
            dst[idx] = result_cropped


@builder.register("build_aced_relaxation_flow")
def build_aced_relaxation_flow(
    chunk_size: Sequence[int],
    bbox: BBox3D,
    dst_resolution: Sequence[float],
    dst: VolumetricLayer,
    match_offsets: VolumetricLayer,
    field_zm1: VolumetricLayer,
    crop_pad: Sequence[int],
    rigidity_masks: Optional[VolumetricLayer] = None,
    field_zm2: Optional[VolumetricLayer] = None,
    field_zm3: Optional[VolumetricLayer] = None,
    num_iter: int = 100,
    lr: float = 0.3,
    rigidity_weight: float = 10.0,
    fix: Optional[Literal["first", "last", "both"]] = "first",
) -> mazepa.Flow:
    flow_schema = build_chunked_volumetric_callable_flow_schema(
        fn=alignment.aced_relaxation.perform_aced_relaxation,
        chunker=VolumetricIndexChunker(chunk_size=Vec3D[int](*chunk_size)),
        crop_pad=Vec3D[int](*crop_pad),
    )
    flow = flow_schema(
        idx=VolumetricIndex(bbox=bbox, resolution=Vec3D[float](*dst_resolution)),
        dst=dst,
        match_offsets=match_offsets,
        rigidity_masks=rigidity_masks,
        field_zm1=field_zm1,
        field_zm2=field_zm2,
        field_zm3=field_zm3,
        num_iter=num_iter,
        lr=lr,
        rigidity_weight=rigidity_weight,
        fix=fix,
    )
    return flow
