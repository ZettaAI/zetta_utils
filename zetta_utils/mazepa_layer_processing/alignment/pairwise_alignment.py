from __future__ import annotations

import copy
import os
from functools import partial
from typing import Any, Callable, Mapping, Sequence

import attrs
import torch

from zetta_utils import builder, mazepa
from zetta_utils.geometry import BBox3D, IntVec3D
from zetta_utils.layer.volumetric import VolumetricIndexTranslator, VolumetricLayer
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.mazepa_layer_processing.common import (
    VolumetricCallableOperation,
    build_subchunkable_apply_flow,
)
from zetta_utils.tensor_ops.common import compare
from zetta_utils.tensor_ops.convert import to_uint8
from zetta_utils.tensor_ops.mask import (
    filter_cc,
    kornia_closing,
    kornia_dilation,
    kornia_opening,
)

from .compute_field_multistage_flow import (
    ComputeFieldMultistageFlowSchema,
    ComputeFieldStage,
)
from .warp_operation import WarpOperation


def _default_layer_factory(
    path: str,
    resolution_list: Sequence[Sequence[int]] | None = None,
    per_scale_config: Mapping[str, Any] | None = None,
    **build_cv_kwargs,
):
    if resolution_list is None:
        resolution_list = []
    if per_scale_config is None:
        per_scale_config = {}

    info_add_scales = None
    if resolution_list is not None:
        assert isinstance(resolution_list, Sequence)
        info_add_scales = []
        for res in resolution_list:
            scale = {
                "resolution": res,
            } | per_scale_config
            info_add_scales.append(scale)
    info_add_scales = build_cv_kwargs.pop("info_add_scales", info_add_scales)

    return build_cv_layer(
        path=path,
        info_add_scales=build_cv_kwargs.pop("info_add_scales", info_add_scales),
        info_add_scales_mode=build_cv_kwargs.pop("info_add_scales_mode", "replace"),
        on_info_exists=build_cv_kwargs.pop("on_info_exists", "overwrite"),
        **build_cv_kwargs,
    )


def _pad_crop_pads(crop_pad, length):
    return [[0, 0, 0]] * (length - 1) + [crop_pad]


def _set_volumetric_callable_default_op_kwargs(
    res_change_mult=None,
    fn_uses_cuda=False,
    task_name=None,
    overrides=None,
):
    if overrides is None:
        overrides = {}
    kwargs = {}
    if task_name is not None:
        kwargs["operation_name"] = task_name
    if fn_uses_cuda:
        kwargs["fn_semaphores"] = ["cuda"]
    if res_change_mult is not None:
        kwargs["res_change_mult"] = res_change_mult
    kwargs |= overrides
    return kwargs


@mazepa.flow_schema_cls
@attrs.mutable
class SubchunkableFnFlowSchema:
    op: Callable
    subchunkable_kwargs: Mapping[str, Any]

    def __init__(
        self,
        dst_resolution: Sequence[int],
        processing_chunk_sizes: Sequence[Sequence[int]],
        task_name: str = None,
        op=None,
        op_kwargs=None,
        fn=None,
        fn_kwargs: Mapping[str, Any] | None = None,
        fn_uses_cuda: bool = False,
        model_res_change_mult: Sequence[int] | None = None,
        model_max_processing_chunk_size: Sequence[int] | None = None,
        src_path: str | None = None,
        src_layer: VolumetricLayer | None = None,
        dst_path: str | None = None,
        dst_layer: VolumetricLayer | None = None,
        dst_factory: Callable | None = None,
        dst_factory_kwargs: Mapping[str, Any] | None = None,
        crop_pad: Sequence[int] | None = None,
        subchunkable_kwargs=None,
    ):
        if src_layer is None:
            src_layer = build_cv_layer(src_path)
        if dst_factory_kwargs is None:
            dst_factory_kwargs = {}
        if subchunkable_kwargs is None:
            subchunkable_kwargs = {}
        if op_kwargs is None:
            op_kwargs = {}
        if fn_kwargs is None:
            fn_kwargs = {}

        if dst_layer is None:
            if dst_factory is None:
                dst_factory = _default_layer_factory
                # Provide some essential arguments
                dst_factory_kwargs = {
                    "info_reference_path": src_path,
                    "resolution_list": [dst_resolution],
                } | dst_factory_kwargs
            dst_layer = dst_factory(dst_path, **dst_factory_kwargs)

        if crop_pad is None:
            crop_pad = [0, 0, 0]

        if model_max_processing_chunk_size is not None:
            max_chunk_size = list(model_max_processing_chunk_size)
            assert len(max_chunk_size) == 2 or len(max_chunk_size) == 3
            if len(max_chunk_size) == 2:
                # pad dimension
                max_chunk_size.append(1)
            processing_chunk_sizes = copy.deepcopy(processing_chunk_sizes)
            processing_chunk_sizes[-1] = [
                min(a, b) for a, b in zip(max_chunk_size, processing_chunk_sizes[-1])
            ]

        self.subchunkable_kwargs = {
            "processing_chunk_sizes": processing_chunk_sizes,
            "processing_crop_pads": _pad_crop_pads(crop_pad, len(processing_chunk_sizes)),
            "skip_intermediaries": True,
            "level_intermediaries_dirs": None,
            "dst": dst_layer,
            "dst_resolution": dst_resolution,
            "op_kwargs": {"src": src_layer},
        } | subchunkable_kwargs

        if op is None:
            # wrap provided fn with VolumetricCallableOperation
            if len(fn_kwargs):
                fn = partial(fn, **fn_kwargs)
            op_kwargs = _set_volumetric_callable_default_op_kwargs(
                res_change_mult=model_res_change_mult,
                fn_uses_cuda=fn_uses_cuda,
                task_name=task_name,
                overrides=op_kwargs,
            )
            self.op = VolumetricCallableOperation(fn, **op_kwargs)
        else:
            if len(op_kwargs):
                op = partial(op, **op_kwargs)
            self.op = op

    def flow(self, bbox):
        flow = build_subchunkable_apply_flow(
            bbox=bbox,
            op=self.op,
            **self.subchunkable_kwargs,
        )
        yield flow


@mazepa.flow_schema_cls
@attrs.mutable
class EncodingFlowSchema:
    subchunkable_flows: Sequence[SubchunkableFnFlowSchema]

    def __init__(
        self,
        models: Sequence[Any],
        dst_path: str | None = None,
        dst_factory_kwargs: Mapping[str, Any] | None = None,
        subchunkable_kwargs=None,
        op_kwargs=None,
        fn_kwargs=None,
        **kwargs,
    ):
        if subchunkable_kwargs is None:
            subchunkable_kwargs = {}
        if op_kwargs is None:
            op_kwargs = {}
        if fn_kwargs is None:
            fn_kwargs = {}
        if dst_factory_kwargs is None:
            dst_factory_kwargs = {}

        dst_factory_kwargs = {
            "resolution_list": [model["dst_resolution"] for model in models],
            "info_field_overrides": {"data_type": "int8"},
        } | dst_factory_kwargs

        self.subchunkable_flows = []
        for model in models:

            fn = model["fn"]
            model_fn_kwargs = fn_kwargs | model.get("fn_kwargs", {})
            model_subchunkable_kwargs = subchunkable_kwargs | model.get("subchunkable_kwargs", {})
            model_op_kwargs = op_kwargs | model.get("op_kwargs", {})
            dst_resolution = model["dst_resolution"]
            dst_path_ = model.get("dst_path", dst_path)

            flow = SubchunkableFnFlowSchema(
                fn=fn,
                fn_kwargs=model_fn_kwargs,
                task_name=f"ImageEncoding-{dst_resolution[0]}",
                fn_uses_cuda=True,
                dst_resolution=dst_resolution,
                model_res_change_mult=model.get("res_change_mult", [1, 1, 1]),
                model_max_processing_chunk_size=model.get("max_processing_chunk_size", None),
                dst_path=dst_path_,
                # dst_factory_kwargs=model_dst_factory_kwargs,
                dst_factory_kwargs=dst_factory_kwargs,
                subchunkable_kwargs=model_subchunkable_kwargs,
                op_kwargs=model_op_kwargs,
                **kwargs,
            )
            self.subchunkable_flows.append(flow)

    def flow(self, bbox):
        for flow in self.subchunkable_flows:
            yield flow(bbox)


@mazepa.flow_schema_cls
@attrs.mutable
class DefectFlowSchema:
    subchunkable_flow: SubchunkableFnFlowSchema

    def __init__(
        self,
        **kwargs,
    ):
        self.subchunkable_flow = SubchunkableFnFlowSchema(
            task_name="Defect",
            fn_uses_cuda=True,
            **kwargs,
        )

    def flow(self, bbox):
        yield self.subchunkable_flow(bbox)


@builder.register("binarize_defect_prediction")
def binarize_defect_prediction(
    src: torch.Tensor,
    threshold,
    kornia_opening_width: int = 0,
    kornia_dilation_width: int = 0,
    filter_cc_threshold: int = 0,
    kornia_closing_width: int = 0,
):
    pred = compare(src, mode=">=", value=threshold, binarize=True)

    mask = to_uint8(pred)  # kornia errors out with `bool`?
    if kornia_opening_width:
        # remove thin line from mask
        mask = kornia_opening(mask, width=kornia_opening_width)
    if kornia_dilation_width:
        # grow mask a little
        mask = kornia_dilation(mask, width=kornia_dilation_width)

    pred = torch.where(mask > 0, 0, pred)

    if filter_cc_threshold:
        # remove small islands that are likely FPs
        pred = filter_cc(pred, mode="keep_large", thr=filter_cc_threshold)
    if kornia_closing_width:
        # connect disconnected folds
        pred = kornia_closing(pred, width=kornia_closing_width)

    return to_uint8(pred)


@mazepa.flow_schema_cls
@attrs.mutable
class BinarizeDefectFlowSchema:
    subchunkable_flow: SubchunkableFnFlowSchema

    def __init__(
        self,
        **kwargs,
    ):
        self.subchunkable_flow = SubchunkableFnFlowSchema(
            task_name="BinarizeDefect",
            **kwargs,
        )

    def flow(self, bbox):
        yield self.subchunkable_flow(bbox)


@builder.register("zero_out_src_with_mask")
def zero_out_src_with_mask2(src, mask, opening_width=0, dilation_width=0):
    # opening_width=2 finds and removes >=2px wide masks
    # opening_width=3 finds and removes >=3px wide masks
    # dilation_width=2 grows mask by 1px
    # dilation_width=3 grows mask by 2px
    if opening_width > 0:
        mask0 = mask
        exclusion_from_dilation = kornia_opening(mask, width=opening_width)
        mask = mask & torch.logical_not(exclusion_from_dilation)
    if dilation_width > 0:
        mask = kornia_dilation(mask, width=dilation_width)
    if opening_width > 0:
        mask |= mask0
    return torch.where(mask > 0, 0, src)  # where(cond, true, false)


@mazepa.flow_schema_cls
@attrs.mutable
class MaskEncodingsFlowSchema:
    subchunkable_flows: Sequence[SubchunkableFnFlowSchema]

    def __init__(
        self,
        dst_resolution_list: Sequence[Sequence[int] | Mapping[str, Any]],
        fn_kwargs: Mapping[str, Any] | None = None,
        src_path: str | None = None,
        src_layer: VolumetricLayer | None = None,
        mask_path: str | None = None,
        mask_layer: VolumetricLayer | None = None,
        mask_resolution: Sequence[int] | None = None,
        dst_factory_kwargs: Mapping[str, Any] | None = None,
        subchunkable_kwargs=None,
        **kwargs,
    ):
        if fn_kwargs is None:
            fn_kwargs = {}
        if subchunkable_kwargs is None:
            subchunkable_kwargs = {}
        if dst_factory_kwargs is None:
            dst_factory_kwargs = {}

        # assume fn takes src & mask as inputs, add mask src to op_kwargs
        if mask_layer is None:
            assert mask_path is not None
            mask_layer = build_cv_layer(
                mask_path,
                data_resolution=mask_resolution,
                interpolation_mode="mask",
            )
        op_kwargs_mask = {"op_kwargs": {"mask": mask_layer}}
        subchunkable_kwargs = op_kwargs_mask | subchunkable_kwargs

        dst_factory_kwargs = {
            "resolution_list": [model["dst_resolution"] for model in dst_resolution_list],
            "info_field_overrides": {"data_type": "int8"},
        } | dst_factory_kwargs

        self.subchunkable_flows = []
        for model in dst_resolution_list:
            dst_resolution = model["dst_resolution"]
            fn_kwargs_ = fn_kwargs | model.get("fn_kwargs", {})

            # add src to op_kwargs in subchunkable_kwargs
            src_path_ = model.get("src_path", src_path)
            src_layer_ = model.get("src_layer", src_layer)
            if src_layer_ is None:
                src_layer_ = build_cv_layer(src_path_)
            subchunkable_kwargs_ = copy.deepcopy(subchunkable_kwargs)
            subchunkable_kwargs_["op_kwargs"]["src"] = src_layer_

            flow = SubchunkableFnFlowSchema(
                fn_kwargs=fn_kwargs_,
                task_name=f"MaskEncodings-{dst_resolution[0]}",
                fn_uses_cuda=False,
                dst_resolution=dst_resolution,
                model_res_change_mult=(1, 1, 1),
                src_path=src_path_,
                src_layer=src_layer_,
                dst_factory_kwargs=dst_factory_kwargs,
                subchunkable_kwargs=subchunkable_kwargs_,
                **kwargs,
            )
            self.subchunkable_flows.append(flow)

    def flow(self, bbox):
        for flow in self.subchunkable_flows:
            yield flow(bbox)


@mazepa.flow_schema_cls
@attrs.mutable
class ComputeFieldFlowSchema:
    flows: Sequence[ComputeFieldMultistageFlowSchema]
    z_offsets: Sequence[int]
    shrink_bbox_to_z_offsets: bool
    z_offset_resolution: int

    def __init__(
        self,
        stages: Sequence[Mapping[str, Any]],
        z_offsets: Sequence[int],
        # resume_path: str | None = None,
        # resume_resolution: Sequence[int] | None = None,
        processing_chunk_sizes: Sequence[Sequence[int]],
        shrink_bbox_to_z_offsets: bool = False,
        src_path: str | None = None,
        src_layer: VolumetricLayer | None = None,
        tgt_path: str | None = None,
        tgt_layer: VolumetricLayer | None = None,
        dst_path: str | None = None,
        dst_factory: Callable | None = None,
        dst_factory_kwargs: Mapping[str, Any] | None = None,
        crop_pad: Sequence[int] | None = None,
        compute_field_multistage_kwargs: Mapping[str, Any] | None = None,
        compute_field_stage_kwargs: Mapping[str, Any] | None = None,
    ):
        if len(stages) == 0:
            raise RuntimeError("Input `stages` is empty")

        if compute_field_multistage_kwargs is None:
            compute_field_multistage_kwargs = {}
        if compute_field_stage_kwargs is None:
            compute_field_stage_kwargs = {}
        if crop_pad is None:
            crop_pad = [0, 0, 0]

        self.z_offsets = z_offsets
        self.shrink_bbox_to_z_offsets = shrink_bbox_to_z_offsets

        if tgt_path is None:
            tgt_path = src_path
        if src_layer is None:
            src_layer = build_cv_layer(src_path)
        if tgt_layer is None:
            tgt_layer = build_cv_layer(tgt_path)
        if dst_factory is None:
            dst_factory = _default_layer_factory

        z_offset_resolution = {stage["dst_resolution"][2] for stage in stages}
        if len(z_offset_resolution) > 1:
            raise RuntimeError("Inconsistent z resolutions between stages!")
        z_offset_resolution = z_offset_resolution.pop()
        self.z_offset_resolution = z_offset_resolution

        cf_stages = []
        for stage in stages:
            cf_kwargs = {}
            cf_kwargs["dst_resolution"] = stage["dst_resolution"]
            cf_kwargs["processing_chunk_sizes"] = processing_chunk_sizes
            cf_kwargs["processing_crop_pads"] = _pad_crop_pads(
                crop_pad, len(processing_chunk_sizes)
            )
            cf_kwargs["expand_bbox_processing"] = True
            cf_kwargs["shrink_processing_chunk"] = False
            cf_kwargs["fn"] = partial(stage["fn"], **stage.get("fn_kwargs", {}))
            if "path" in stage:
                layer = build_cv_layer(stage["path"])
                cf_kwargs["src"] = layer
                cf_kwargs["tgt"] = layer
            # override with user values
            cf_kwargs |= compute_field_stage_kwargs
            cf_kwargs |= stage.get("cf_kwargs", {})
            cf_stage = ComputeFieldStage(**cf_kwargs)
            cf_stages.append(cf_stage)

        self.flows = []
        for z_offset in z_offsets:

            dst_path_ = os.path.join(dst_path, str(z_offset))
            dst_kwargs = {
                "path": dst_path_,
                "info_reference_path": src_path,
                "resolution_list": [stage["dst_resolution"] for stage in stages],
                "info_field_overrides": {
                    "type": "image",
                    "data_type": "float32",
                    "num_channels": 2,
                },
            } | dst_factory_kwargs
            dst_layer = dst_factory(**dst_kwargs)
            default_tmp_layer_factory = partial(
                build_cv_layer,
                info_reference_path=dst_path_,
                on_info_exists="overwrite",
            )
            default_tmp_layer_dir = os.path.join(dst_path_, "tmp")

            ms_kwargs = {}
            ms_kwargs["stages"] = cf_stages
            ms_kwargs["tmp_layer_dir"] = default_tmp_layer_dir
            ms_kwargs["tmp_layer_factory"] = default_tmp_layer_factory
            ms_kwargs["src"] = src_layer
            ms_kwargs["tgt"] = tgt_layer
            ms_kwargs["dst"] = dst_layer
            ms_kwargs["tgt_offset"] = [0, 0, z_offset]
            ms_kwargs["offset_resolution"] = [1, 1, z_offset_resolution]
            # override with user values
            ms_kwargs |= compute_field_multistage_kwargs

            flow = ComputeFieldMultistageFlowSchema(
                stages=ms_kwargs.pop("stages"),
                tmp_layer_dir=ms_kwargs.pop("tmp_layer_dir"),
                tmp_layer_factory=ms_kwargs.pop("tmp_layer_factory"),
            )
            self.flows.append(partial(flow, **ms_kwargs))

    def flow(self, bbox):
        for flow, z_offset in zip(self.flows, self.z_offsets):
            bbox_ = bbox
            if self.shrink_bbox_to_z_offsets:
                bbox_ = _shrink_bbox_to_z_offset(bbox, z_offset, self.z_offset_resolution)
            yield flow(bbox=bbox_)


def _shrink_bbox_to_z_offset(bbox, z_offset, z_offset_resolution):
    if z_offset < 0:
        bbox_ = BBox3D.from_coords(
            bbox.start + IntVec3D(0, 0, -z_offset * z_offset_resolution), bbox.end
        )
    else:
        bbox_ = BBox3D.from_coords(
            bbox.start, bbox.end - IntVec3D(0, 0, z_offset * z_offset_resolution)
        )
    return bbox_


@mazepa.flow_schema_cls
@attrs.mutable
class InvertFieldFlowSchema:
    flows: Sequence[SubchunkableFnFlowSchema]
    z_offsets: Sequence[int]
    shrink_bbox_to_z_offsets: bool
    z_offset_resolution: int

    def __init__(
        self,
        dst_resolution: Sequence[int],
        z_offsets: Sequence[int],
        shrink_bbox_to_z_offsets: bool = False,
        src_path: str | None = None,
        dst_path: str | None = None,
        **kwargs,
    ):
        self.z_offsets = z_offsets
        self.shrink_bbox_to_z_offsets = shrink_bbox_to_z_offsets
        self.z_offset_resolution = dst_resolution[2]
        self.flows = []
        for z_offset in z_offsets:
            src_path_ = os.path.join(src_path, str(z_offset))
            dst_path_ = os.path.join(dst_path, str(z_offset))
            flow = SubchunkableFnFlowSchema(
                task_name=f"InvertField_z{z_offset}",
                dst_resolution=dst_resolution,
                src_path=src_path_,
                dst_path=dst_path_,
                **kwargs,
            )
            self.flows.append(flow)

    def flow(self, bbox):
        for flow, z_offset in zip(self.flows, self.z_offsets):
            bbox_ = bbox
            if self.shrink_bbox_to_z_offsets:
                bbox_ = _shrink_bbox_to_z_offset(bbox, z_offset, self.z_offset_resolution)
            yield flow(bbox=bbox_)


@mazepa.flow_schema_cls
@attrs.mutable
class WarpFlowSchema:
    flows: Sequence[SubchunkableFnFlowSchema]
    z_offsets: Sequence[int]
    shrink_bbox_to_z_offsets: bool
    z_offset_resolution: int

    def __init__(
        self,
        dst_resolution: Sequence[int],
        z_offsets: Sequence[int],
        src_path: str,
        field_path: str,
        dst_path: str,
        shrink_bbox_to_z_offsets: bool = False,
        field_resolution: Sequence[int] = None,
        subchunkable_kwargs=None,
        **kwargs,
    ):
        if subchunkable_kwargs is None:
            subchunkable_kwargs = {}

        self.z_offsets = z_offsets
        self.shrink_bbox_to_z_offsets = shrink_bbox_to_z_offsets
        self.z_offset_resolution = dst_resolution[2]

        self.flows = []
        for z_offset in z_offsets:

            field_path_ = os.path.join(field_path, str(z_offset))
            dst_path_ = os.path.join(dst_path, str(z_offset))
            src_idx_translator = VolumetricIndexTranslator(
                offset=[0, 0, z_offset], resolution=dst_resolution
            )
            src_layer = build_cv_layer(src_path, index_procs=[src_idx_translator])
            field_layer = build_cv_layer(
                field_path_, data_resolution=field_resolution, interpolation_mode="field"
            )
            subchunkable_kwargs_ = {
                "op_kwargs": {
                    "src": src_layer,
                    "field": field_layer,
                }
            } | subchunkable_kwargs

            flow = SubchunkableFnFlowSchema(
                task_name=f"Warp_{z_offset}",
                op=WarpOperation(mode="img"),
                dst_resolution=dst_resolution,
                fn_uses_cuda=False,
                src_path=src_path,
                src_layer=src_layer,
                dst_path=dst_path_,
                subchunkable_kwargs=subchunkable_kwargs_,
                **kwargs,
            )
            self.flows.append(flow)

    def flow(self, bbox):
        for flow, z_offset in zip(self.flows, self.z_offsets):
            bbox_ = bbox
            if self.shrink_bbox_to_z_offsets:
                bbox_ = _shrink_bbox_to_z_offset(bbox, z_offset, self.z_offset_resolution)
            yield flow(bbox=bbox_)


@mazepa.flow_schema_cls
@attrs.mutable
class EncodeWarpedImgsFlowSchema:
    flows: Sequence[SubchunkableFnFlowSchema]
    z_offsets: Sequence[int]
    shrink_bbox_to_z_offsets: bool
    z_offset_resolution: int

    def __init__(
        self,
        model: Mapping[str, Any],
        z_offsets: Sequence[int],
        src_path: str,
        dst_path: str,
        src_resolution: Sequence[int] | None = None,
        shrink_bbox_to_z_offsets: bool = False,
        dst_factory_kwargs: Mapping[str, Any] | None = None,
        reencode_tgt: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        if dst_factory_kwargs is None:
            dst_factory_kwargs = {}

        fn = model["fn"]
        dst_resolution = model["dst_resolution"]

        self.z_offsets = copy.deepcopy(z_offsets)
        self.shrink_bbox_to_z_offsets = shrink_bbox_to_z_offsets
        self.z_offset_resolution = dst_resolution[2]

        dst_factory_kwargs = {
            "resolution_list": [dst_resolution],
            "info_field_overrides": {"data_type": "int8"},
        } | dst_factory_kwargs

        self.flows = []
        for z_offset in self.z_offsets:

            src_path_ = os.path.join(src_path, str(z_offset))
            dst_path_ = os.path.join(dst_path, str(z_offset))
            src_layer = build_cv_layer(
                src_path_, data_resolution=src_resolution, interpolation_mode="img"
            )

            flow = SubchunkableFnFlowSchema(
                fn=fn,
                task_name=f"EncodeWarpedImg_z{z_offset}",
                fn_uses_cuda=True,
                dst_resolution=dst_resolution,
                model_res_change_mult=model.get("res_change_mult", [1, 1, 1]),
                model_max_processing_chunk_size=model.get("max_processing_chunk_size", None),
                dst_factory_kwargs=dst_factory_kwargs,
                src_path=src_path_,
                src_layer=src_layer,
                dst_path=dst_path_,
                **kwargs,
            )
            self.flows.append(flow)

        if reencode_tgt is not None:
            # re-encode tgt with the given encoder
            src_path_ = reencode_tgt["src_path"]
            dst_path_ = reencode_tgt["dst_path"]
            flow = SubchunkableFnFlowSchema(
                fn=fn,
                task_name="EncodeWarpedImg_tgt",
                fn_uses_cuda=True,
                dst_resolution=dst_resolution,
                model_res_change_mult=model.get("res_change_mult", [1, 1, 1]),
                model_max_processing_chunk_size=model.get("max_processing_chunk_size", None),
                dst_factory_kwargs=dst_factory_kwargs,
                src_path=src_path_,
                dst_path=dst_path_,
                **kwargs,
            )
            self.flows.append(flow)
            self.z_offsets.append(0)

    def flow(self, bbox):
        for flow, z_offset in zip(self.flows, self.z_offsets):
            bbox_ = bbox
            if self.shrink_bbox_to_z_offsets:
                bbox_ = _shrink_bbox_to_z_offset(bbox, z_offset, self.z_offset_resolution)
            yield flow(bbox=bbox_)


@mazepa.flow_schema_cls
@attrs.mutable
class MisalignmentDetectorFlowSchema:
    flows: Sequence[SubchunkableFnFlowSchema]
    z_offsets: Sequence[int]
    shrink_bbox_to_z_offsets: bool
    z_offset_resolution: int

    def __init__(
        self,
        dst_resolution: Sequence[int],
        models: Sequence[Mapping[str, Any]],  # one per z_offset
        z_offsets: Sequence[int],
        src_path: str,
        dst_path: str,
        tgt_path: str | None = None,
        tgt_layer: VolumetricLayer | None = None,
        shrink_bbox_to_z_offsets: bool = False,
        dst_factory_kwargs: Mapping[str, Any] | None = None,
        subchunkable_kwargs=None,
        **kwargs,
    ):
        if dst_factory_kwargs is None:
            dst_factory_kwargs = {}
        if subchunkable_kwargs is None:
            subchunkable_kwargs = {}

        self.z_offsets = z_offsets
        self.shrink_bbox_to_z_offsets = shrink_bbox_to_z_offsets
        self.z_offset_resolution = dst_resolution[2]

        dst_factory_kwargs = {
            "resolution_list": [dst_resolution],
            "info_field_overrides": {"data_type": "uint8"},
        } | dst_factory_kwargs

        # add tgt to op_kwargs
        if tgt_layer is None:
            assert tgt_path is not None
            tgt_layer = build_cv_layer(tgt_path)
        op_kwargs_mask = {"op_kwargs": {"tgt": tgt_layer}}
        subchunkable_kwargs = op_kwargs_mask | subchunkable_kwargs

        if len(models) == 1:
            models = models * len(z_offsets)
        assert len(models) == len(z_offsets)

        self.flows = []
        for z_offset, model in zip(z_offsets, models):

            fn = model["fn"]
            dst_resolution_ = model.get("dst_resolution", dst_resolution)

            src_path_ = os.path.join(src_path, str(z_offset))
            dst_path_ = os.path.join(dst_path, str(z_offset))
            src_layer = build_cv_layer(src_path_)

            subchunkable_kwargs_ = copy.deepcopy(subchunkable_kwargs)
            subchunkable_kwargs_["op_kwargs"]["src"] = src_layer

            flow = SubchunkableFnFlowSchema(
                fn=fn,
                task_name=f"Misd_z{z_offset}",
                fn_uses_cuda=True,
                dst_resolution=dst_resolution_,
                model_max_processing_chunk_size=model.get("max_processing_chunk_size", None),
                dst_factory_kwargs=dst_factory_kwargs,
                src_path=src_path_,
                src_layer=src_layer,
                dst_path=dst_path_,
                subchunkable_kwargs=subchunkable_kwargs_,
                **kwargs,
            )
            self.flows.append(flow)

    def flow(self, bbox):
        for flow, z_offset in zip(self.flows, self.z_offsets):
            bbox_ = bbox
            if self.shrink_bbox_to_z_offsets:
                bbox_ = _shrink_bbox_to_z_offset(bbox, z_offset, self.z_offset_resolution)
            yield flow(bbox=bbox_)


@builder.register("binarize_misd_mask")
def binarize_misd_mask(src, threshold):
    src = compare(src, mode=">=", value=threshold)
    return to_uint8(src)


@mazepa.flow_schema_cls
@attrs.mutable
class BinarizeMisalignmentFlowSchema:
    flows: Sequence[SubchunkableFnFlowSchema]
    z_offsets: Sequence[int]
    shrink_bbox_to_z_offsets: bool
    z_offset_resolution: int

    def __init__(
        self,
        dst_resolution: Sequence[int],
        models: Sequence[Mapping[str, Any]],  # one per z_offset
        z_offsets: Sequence[int],
        src_path: str,
        dst_path: str,
        shrink_bbox_to_z_offsets: bool = False,
        **kwargs,
    ):
        self.z_offsets = z_offsets
        self.shrink_bbox_to_z_offsets = shrink_bbox_to_z_offsets
        self.z_offset_resolution = dst_resolution[2]

        if len(models) == 1:
            models = models * len(z_offsets)
        assert len(models) == len(z_offsets)

        self.flows = []
        for z_offset, model in zip(z_offsets, models):

            fn = model["fn"]
            dst_resolution_ = model.get("dst_resolution", dst_resolution)

            src_path_ = os.path.join(src_path, str(z_offset))
            dst_path_ = os.path.join(dst_path, str(z_offset))

            flow = SubchunkableFnFlowSchema(
                fn=fn,
                task_name=f"BinarizeMisd_z{z_offset}",
                dst_resolution=dst_resolution_,
                src_path=src_path_,
                dst_path=dst_path_,
                **kwargs,
            )
            self.flows.append(flow)

    def flow(self, bbox):
        for flow, z_offset in zip(self.flows, self.z_offsets):
            bbox_ = bbox
            if self.shrink_bbox_to_z_offsets:
                bbox_ = _shrink_bbox_to_z_offset(bbox, z_offset, self.z_offset_resolution)
            yield flow(bbox=bbox_)


@builder.register("PairwiseAlignmentFlowSchema")
@mazepa.flow_schema_cls
@attrs.mutable
class PairwiseAlignmentFlowSchema:
    encoding_flow: EncodingFlowSchema | None
    defect_flow: DefectFlowSchema | None
    binarize_defect_flow: BinarizeDefectFlowSchema | None
    mask_encodings_flow: MaskEncodingsFlowSchema | None
    compute_field_flow: ComputeFieldFlowSchema | None
    invert_field_flow: InvertFieldFlowSchema | None
    warp_flow: WarpFlowSchema | None
    enc_warped_imgs_flow: EncodeWarpedImgsFlowSchema | None
    misd_flow: MisalignmentDetectorFlowSchema | None
    binarize_misd_flow: MisalignmentDetectorFlowSchema | None

    def flow(self, bbox):
        encoding_task = []
        if self.encoding_flow is not None:
            encoding_task = self.encoding_flow(bbox)
            yield encoding_task

        defect_task = []
        if self.defect_flow is not None:
            defect_task = self.defect_flow(bbox)
            yield defect_task

        binarized_defect_task = []
        if self.binarize_defect_flow is not None:
            yield mazepa.Dependency(defect_task)
            binarized_defect_task = self.binarize_defect_flow(bbox)
            yield binarized_defect_task

        mask_encodings_task = []
        if self.mask_encodings_flow is not None:
            yield mazepa.Dependency(encoding_task)
            yield mazepa.Dependency(binarized_defect_task)
            mask_encodings_task = self.mask_encodings_flow(bbox)
            yield mask_encodings_task

        compute_field_task = []
        if self.compute_field_flow is not None:
            yield mazepa.Dependency(encoding_task)
            yield mazepa.Dependency(mask_encodings_task)
            compute_field_task = self.compute_field_flow(bbox)
            yield compute_field_task

        invert_field_task = []
        if self.invert_field_flow is not None:
            yield mazepa.Dependency(compute_field_task)
            invert_field_task = self.invert_field_flow(bbox)
            yield invert_field_task

        warp_task = []
        if self.warp_flow is not None:
            yield mazepa.Dependency(invert_field_task)
            warp_task = self.warp_flow(bbox)
            yield warp_task

        enc_warped_imgs_task = []
        if self.enc_warped_imgs_flow is not None:
            yield mazepa.Dependency(warp_task)
            enc_warped_imgs_task = self.enc_warped_imgs_flow(bbox)
            yield enc_warped_imgs_task

        misd_task = []
        if self.misd_flow is not None:
            yield mazepa.Dependency(enc_warped_imgs_task)
            misd_task = self.misd_flow(bbox)
            yield misd_task

        binarize_misd_task = []
        if self.binarize_misd_flow is not None:
            yield mazepa.Dependency(misd_task)
            binarize_misd_task = self.binarize_misd_flow(bbox)
            yield binarize_misd_task


@builder.register("build_pairwise_alignment_flow")
def build_pairwise_alignment_flow(
    bbox: BBox3D | None = None,
    bbox_list: Sequence[BBox3D] | None = None,
    src_image_path: str = "",
    project_folder: str = "",
    z_offsets: Sequence[int] = (-1,),
    run_encoding: bool = False,
    encoding_flow_kwargs: Mapping[str, Any] | None = None,
    run_defect: bool = False,
    skipped_defect: bool = False,
    defect_flow_kwargs: Mapping[str, Any] | None = None,
    run_binarize_defect: bool = False,
    binarize_defect_flow_kwargs: Mapping[str, Any] | None = None,
    run_mask_encodings: bool = False,
    mask_encodings_flow_kwargs: Mapping[str, Any] | None = None,
    run_compute_field: bool = False,
    compute_field_subproject: str | None = None,
    compute_field_flow_kwargs: Mapping[str, Any] | None = None,
    run_invert_field: bool = False,
    invert_field_subproject: str | None = None,
    invert_field_flow_kwargs: Mapping[str, Any] | None = None,
    run_warp: bool = False,
    warp_subproject: str | None = None,
    warp_flow_kwargs: Mapping[str, Any] | None = None,
    run_enc_warped_imgs: bool = False,
    enc_warped_imgs_flow_kwargs: Mapping[str, Any] | None = None,
    run_misd: bool = False,
    misd_flow_kwargs: Mapping[str, Any] | None = None,
    run_binarize_misd: bool = False,
    binarize_misd_flow_kwargs: Mapping[str, Any] | None = None,
) -> mazepa.Flow:  # pylint: disable=too-many-statements)

    if bbox is None and bbox_list is None:
        raise RuntimeError("Either `bbox` and `bbox_list` must be provided")
    if bbox is not None and bbox_list is not None:
        raise RuntimeError("`bbox` and `bbox_list` cannot be both specified")
    if bbox_list is None:
        bbox_list = [bbox]

    def resolve_path(path, default=None):
        if path is None:
            path = default
        if "gs://" in path:
            # path is absolute
            return path
        # otherwise path is relative
        return os.path.join(project_folder, path)

    def set_path(config, key, default, subproject=None):
        if subproject is not None:
            default = os.path.join(subproject, default)
        config[key] = resolve_path(config.get(key, None), default)

    encoding_flow = None
    if encoding_flow_kwargs is None:
        encoding_flow_kwargs = {}
    set_path(encoding_flow_kwargs, "src_path", default=src_image_path)
    set_path(encoding_flow_kwargs, "dst_path", default="encodings")
    if run_encoding:
        encoding_flow = EncodingFlowSchema(**encoding_flow_kwargs)

    defect_flow = None
    if defect_flow_kwargs is None:
        defect_flow_kwargs = {}
    set_path(defect_flow_kwargs, "src_path", default=src_image_path)
    set_path(defect_flow_kwargs, "dst_path", default="defect")
    if run_defect:
        defect_flow = DefectFlowSchema(**defect_flow_kwargs)

    binarize_defect_flow = None
    if binarize_defect_flow_kwargs is None:
        binarize_defect_flow_kwargs = {}
    set_path(binarize_defect_flow_kwargs, "src_path", default=defect_flow_kwargs["dst_path"])
    set_path(binarize_defect_flow_kwargs, "dst_path", default="defect_binarized")
    if run_binarize_defect:
        binarize_defect_flow = BinarizeDefectFlowSchema(**binarize_defect_flow_kwargs)

    mask_encodings_flow = None
    if mask_encodings_flow_kwargs is None:
        mask_encodings_flow_kwargs = {}
    set_path(
        mask_encodings_flow_kwargs, "mask_path", default=binarize_defect_flow_kwargs["dst_path"]
    )
    set_path(mask_encodings_flow_kwargs, "src_path", default=encoding_flow_kwargs["dst_path"])
    set_path(mask_encodings_flow_kwargs, "dst_path", default="encodings_masked")
    if run_mask_encodings:
        mask_encodings_flow = MaskEncodingsFlowSchema(**mask_encodings_flow_kwargs)

    compute_field_flow = None
    if compute_field_flow_kwargs is None:
        compute_field_flow_kwargs = {}
    cf_default_src = (
        encoding_flow_kwargs["dst_path"]
        if skipped_defect
        else mask_encodings_flow_kwargs["dst_path"]
    )
    set_path(compute_field_flow_kwargs, "src_path", default=cf_default_src)
    set_path(compute_field_flow_kwargs, "dst_path", "fields_fwd", compute_field_subproject)
    compute_field_flow_kwargs = {"z_offsets": z_offsets} | compute_field_flow_kwargs
    if run_compute_field:
        compute_field_flow = ComputeFieldFlowSchema(**compute_field_flow_kwargs)

    invert_field_flow = None
    if invert_field_flow_kwargs is None:
        invert_field_flow_kwargs = {}
    set_path(invert_field_flow_kwargs, "src_path", default=compute_field_flow_kwargs["dst_path"])
    set_path(invert_field_flow_kwargs, "dst_path", "fields_inv", invert_field_subproject)
    if "stages" in compute_field_flow_kwargs and len(compute_field_flow_kwargs["stages"]):
        # try to get field_resolution from the previous step
        cf_dst_resolution = compute_field_flow_kwargs["stages"][-1]["dst_resolution"]
        invert_field_flow_kwargs = {"dst_resolution": cf_dst_resolution} | invert_field_flow_kwargs
    invert_field_flow_kwargs = {"z_offsets": z_offsets} | invert_field_flow_kwargs
    if run_invert_field:
        invert_field_flow = InvertFieldFlowSchema(**invert_field_flow_kwargs)

    warp_flow = None
    if warp_flow_kwargs is None:
        warp_flow_kwargs = {}
    set_path(warp_flow_kwargs, "src_path", default=src_image_path)
    set_path(warp_flow_kwargs, "field_path", default=invert_field_flow_kwargs["dst_path"])
    set_path(warp_flow_kwargs, "dst_path", "imgs_warped", warp_subproject)
    if "dst_resolution" in invert_field_flow_kwargs:
        # try to get field_resolution from the previous step
        warp_flow_kwargs = {
            "field_resolution": invert_field_flow_kwargs["dst_resolution"]
        } | warp_flow_kwargs
    warp_flow_kwargs = {"z_offsets": z_offsets} | warp_flow_kwargs
    if run_warp:
        warp_flow = WarpFlowSchema(**warp_flow_kwargs)

    enc_warped_imgs_flow = None
    if enc_warped_imgs_flow_kwargs is None:
        enc_warped_imgs_flow_kwargs = {}
    set_path(enc_warped_imgs_flow_kwargs, "src_path", default=warp_flow_kwargs["dst_path"])
    set_path(enc_warped_imgs_flow_kwargs, "dst_path", default="imgs_warped_encoded")
    if "dst_resolution" in warp_flow_kwargs:
        # try to get resolution from the previous step
        enc_warped_imgs_flow_kwargs = {
            "src_resolution": warp_flow_kwargs["dst_resolution"]
        } | enc_warped_imgs_flow_kwargs
    enc_warped_imgs_flow_kwargs = {"z_offsets": z_offsets} | enc_warped_imgs_flow_kwargs
    if "reencode_tgt" in enc_warped_imgs_flow_kwargs:
        set_path(enc_warped_imgs_flow_kwargs["reencode_tgt"], "src_path", default=src_image_path)
        set_path(enc_warped_imgs_flow_kwargs["reencode_tgt"], "dst_path", default="encodings_misd")
    if run_enc_warped_imgs:
        enc_warped_imgs_flow = EncodeWarpedImgsFlowSchema(**enc_warped_imgs_flow_kwargs)

    misd_flow = None
    if misd_flow_kwargs is None:
        misd_flow_kwargs = {}
    set_path(misd_flow_kwargs, "src_path", default=enc_warped_imgs_flow_kwargs["dst_path"])
    set_path(misd_flow_kwargs, "dst_path", default="misalignments")
    # determine tgt
    if "reencode_tgt" in enc_warped_imgs_flow_kwargs:
        set_path(
            misd_flow_kwargs,
            "tgt_path",
            default=enc_warped_imgs_flow_kwargs["reencode_tgt"]["dst_path"],
        )
    else:
        set_path(misd_flow_kwargs, "tgt_path", default=encoding_flow_kwargs["dst_path"])
    misd_flow_kwargs = {"z_offsets": z_offsets} | misd_flow_kwargs
    if run_misd:
        misd_flow = MisalignmentDetectorFlowSchema(**misd_flow_kwargs)

    binarize_misd_flow = None
    if binarize_misd_flow_kwargs is None:
        binarize_misd_flow_kwargs = {}
    set_path(binarize_misd_flow_kwargs, "src_path", default=misd_flow_kwargs["dst_path"])
    set_path(binarize_misd_flow_kwargs, "dst_path", default="misalignments_binarized")
    binarize_misd_flow_kwargs = {"z_offsets": z_offsets} | binarize_misd_flow_kwargs
    if run_binarize_misd:
        binarize_misd_flow = BinarizeMisalignmentFlowSchema(**binarize_misd_flow_kwargs)

    flow_schema = PairwiseAlignmentFlowSchema(
        encoding_flow=encoding_flow,
        defect_flow=defect_flow,
        binarize_defect_flow=binarize_defect_flow,
        mask_encodings_flow=mask_encodings_flow,
        compute_field_flow=compute_field_flow,
        invert_field_flow=invert_field_flow,
        warp_flow=warp_flow,
        enc_warped_imgs_flow=enc_warped_imgs_flow,
        misd_flow=misd_flow,
        binarize_misd_flow=binarize_misd_flow,
    )

    @mazepa.flow_schema
    def run_multi(bbox_list):
        yield [flow_schema(bbox) for bbox in bbox_list]

    return run_multi(bbox_list)
