from __future__ import annotations

# pylint: disable=unused-import

import copy
import os
from typing import Callable, Literal, Sequence, Union, Any, Mapping

import attrs
from functools import partial

import torch

from zetta_utils import builder, mazepa
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import (
    DataResolutionInterpolator,
    VolumetricIndexTranslator,
    VolumetricLayer,
)

# from zetta_utils.alignment.base_coarsener import BaseCoarsener
from zetta_utils.alignment import BaseCoarsener
from zetta_utils.alignment import DefectDetector
from zetta_utils.mazepa_layer_processing.common import (
    VolumetricCallableOperation,
    build_subchunkable_apply_flow,
)
from zetta_utils.layer.volumetric.cloudvol import (
    build_cv_layer,
)

from zetta_utils.tensor_ops.convert import to_uint8
from zetta_utils.tensor_ops.common import (
    compare,
)
from zetta_utils.tensor_ops.mask import (
    kornia_opening,
    kornia_dilation,
    filter_cc,
    kornia_closing,
)

from .compute_field_multistage_flow import (
    ComputeFieldMultistageFlowSchema,
    ComputeFieldStage,
    build_compute_field_multistage_flow,
)

from .warp_operation import WarpOperation


def _make_dst_layer(path, info_reference_path, data_type, resolution_list, chunk_size=None):
    return build_cv_layer(
        path=path,
        info_reference_path=info_reference_path,
        info_add_scales=resolution_list,
        info_add_scales_mode="replace",
        info_field_overrides={
            "type": "image",
            "num_channels": 1,
            "data_type": data_type,
        },
        info_chunk_size=chunk_size,
        on_info_exists="overwrite",
    )

def _default_layer_factory(
        path,
        resolution_list=None,
        add_zfpc=False,
        **kwargs):
    if resolution_list is None:
        resolution_list = []

    info_add_scales = None
    if resolution_list is not None:
        assert isinstance(resolution_list, list)
        info_add_scales = []
        for res in resolution_list:
            scale = {
                "resolution": res,
            }
            if add_zfpc:
                scale |= {
                    "encoding": "zfpc",
                    "zfpc_correlated_dims": [True, True, False, False],
                    "zfpc_tolerance": 0.001953125,
                }
            info_add_scales.append(scale)
    info_add_scales = kwargs.pop("info_add_scales", info_add_scales)

    # info_field_overrides = kwargs.pop("info_field_overrides", {})
    # info_field_overrides = {
    #         "type": "image",
    #         "num_channels": 2,
    #         "data_type": data_type,
    # } | info_field_overrides
    return build_cv_layer(
        path=path,
        info_add_scales=kwargs.pop("info_add_scales", info_add_scales),
        info_add_scales_mode=kwargs.pop("info_add_scales_mode", "replace"),
        on_info_exists=kwargs.pop("on_info_exists", "overwrite"),
        **kwargs
    )

def _make_dst_layer_field(path, data_type, resolution_list, **kwargs):
    scales = []
    for res in resolution_list:
        scale = {
            "resolution": res,
            "encoding": "zfpc",
            "zfpc_correlated_dims": [True, True, False, False],
            "zfpc_tolerance": 0.001953125,
        }
        scales.append(scale)

    info_field_overrides = kwargs.pop("info_field_overrides", {})
    info_field_overrides = {
            "type": "image",
            "num_channels": 2,
            "data_type": data_type,
    } | info_field_overrides

    return build_cv_layer(
        path=path,
        info_add_scales=scales,
        info_add_scales_mode=kwargs.pop("info_add_scales_mode", "replace"),
        info_field_overrides=info_field_overrides,
        on_info_exists="overwrite",
        **kwargs
    )


def _set_tiled_fn_kwargs(model_path, ds_factor=None, overrides=None):
    if overrides is None:
        overrides = {}
    fn_kwargs = {}
    fn_kwargs["model_path"] = model_path
    if ds_factor is not None:
        fn_kwargs["ds_factor"] = ds_factor
    # fn_kwargs["tile_pad_in"] = ds_factor * crop_pad[0]
    # fn_kwargs["tile_size"] = ds_factor * processing_chunk_sizes[-1][0]
    fn_kwargs |= overrides
    return fn_kwargs


def _get_dst_layer(dst: str | VolumetricLayer, layer_factory: Callable):
    if isinstance(dst, VolumetricLayer):
        return dst
    return layer_factory(dst)


def _pad_crop_pads(crop_pad, length):
    return [[0, 0, 0]] * (length - 1) + [crop_pad]


def _set_subchunkable_kwargs(
    src_layer,
    dst_layer,
    processing_chunk_sizes,
    crop_pad,
    dst_resolution=None,
    overrides=None,
):
    kwargs = {}
    kwargs["processing_chunk_sizes"] = processing_chunk_sizes
    kwargs["processing_crop_pads"] = _pad_crop_pads(crop_pad, len(processing_chunk_sizes))
    kwargs["skip_intermediaries"] = True
    kwargs["level_intermediaries_dirs"] = None
    kwargs["dst"] = dst_layer
    kwargs["dst_resolution"] = dst_resolution

    if overrides is not None:
        kwargs |= overrides

    # need to merge instead of override `op_kwargs`
    # if "op_kwargs" not in kwargs or "src" not in kwargs["op_kwargs"]:
    kwargs["op_kwargs"] = {"src": src_layer} | kwargs.get("op_kwargs", {})

    return kwargs


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
    # fn_kwargs: Mapping[str, Any]
    op_kwargs: Mapping[str, Any]
    subchunkable_kwargs: Mapping[str, Any]
    fn: Callable[..., Any]

    def __init__(
        self,
        fn,
        dst_resolution: Sequence[int],
        processing_chunk_sizes: Sequence[Sequence[int]],
        task_name: str = None,
        fn_uses_cuda: bool = False,
        model_res_change_mult: Sequence[int] | None = None,
        model_max_processing_chunk_size: Sequence[int] | None = None,
        src_path: str | None = None,
        src_layer: VolumetricLayer | None = None,
        dst_path: str | None = None,
        dst_layer: VolumetricLayer | None = None,
        dst_factory: Callable | None = None,
        dst_factory_chunk_size: Sequence[int] | None = None,
        dst_factory_resolution_list: Sequence[Sequence[int]] | None = None,
        dst_data_type: str | None = None,
        crop_pad: Sequence[int] | None = None,
        subchunkable_kwargs=None,
        op_kwargs=None,
        # fn_kwargs=None,
    ):
        if src_layer is None:
            assert src_path is not None
            src_layer = build_cv_layer(src_path)

        if dst_layer is None:
            if src_path is None:
                raise RuntimeError(
                    f"If `dst_layer` is not provided, `src_path` must be specified to be used as"
                    f" `info_reference_path` in making the destination layer with `dst_path`={dst_path}"
                )
            if dst_factory is None:
                if dst_factory_resolution_list is None:
                    dst_factory_resolution_list = [dst_resolution]
                dst_factory = partial(
                    _make_dst_layer,
                    resolution_list=dst_factory_resolution_list,
                    chunk_size=dst_factory_chunk_size,
                    info_reference_path=src_path,
                    data_type=dst_data_type,
                )
            dst_layer = dst_factory(dst_path)

        if crop_pad is None:
            crop_pad = [0, 0, 0]

        if model_max_processing_chunk_size is not None:
            max_chunk_size = list(model_max_processing_chunk_size)
            assert len(max_chunk_size) == 2 or len(max_chunk_size) == 3
            if len(max_chunk_size) == 2:
                # pad dimension
                max_chunk_size.append(1)
            processing_chunk_sizes[-1] = [
                min(a, b) for a, b in zip(max_chunk_size, processing_chunk_sizes[-1])
            ]

        self.subchunkable_kwargs = _set_subchunkable_kwargs(
            src_layer=src_layer,
            dst_layer=dst_layer,
            dst_resolution=dst_resolution,
            processing_chunk_sizes=processing_chunk_sizes,
            crop_pad=crop_pad,
            overrides=subchunkable_kwargs,
        )
        self.op_kwargs = _set_volumetric_callable_default_op_kwargs(
            res_change_mult=model_res_change_mult,
            fn_uses_cuda=fn_uses_cuda,
            task_name=task_name,
            overrides=op_kwargs,
        )
        self.fn = fn

    def flow(self, bbox, i):  # pylint: disable=unused-argument
        flow = build_subchunkable_apply_flow(
            bbox=bbox,
            op=VolumetricCallableOperation(
                # fn=self.fn(**self.fn_kwargs),
                fn=self.fn,
                **self.op_kwargs,
            ),
            **self.subchunkable_kwargs,
        )
        yield flow


@mazepa.flow_schema_cls
@attrs.mutable
class SubchunkableFnFlowSchema2:
    # op_kwargs: Mapping[str, Any]
    op: Callable
    subchunkable_kwargs: Mapping[str, Any]
    # fn: Callable[..., Any]

    def __init__(
        self,
        dst_resolution: Sequence[int],
        processing_chunk_sizes: Sequence[Sequence[int]],
        task_name: str = None,
        op=None,
        op_kwargs=None,
        fn=None,
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

        if dst_layer is None:
            if src_path is None:
                raise RuntimeError(
                    f"If `dst_layer` is not provided, `src_path` must be specified to be used as"
                    f" `info_reference_path` in making the destination layer with `dst_path`={dst_path}"
                )
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
            "op_kwargs": {"src": src_layer}
        } | subchunkable_kwargs

        if op is None:
            # wrap provided fn with VolumetricCallableOperation
            op_kwargs = _set_volumetric_callable_default_op_kwargs(
                res_change_mult=model_res_change_mult,
                fn_uses_cuda=fn_uses_cuda,
                task_name=task_name,
                overrides=op_kwargs,
            )
            self.op = VolumetricCallableOperation(fn, **op_kwargs)
        else:
            # self.op = partial(op, **op_kwargs)
            self.op = op

    def flow(self, bbox, i):  # pylint: disable=unused-argument
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
        processing_chunk_sizes: Sequence[Sequence[int]],
        src_path: str | None = None,
        src_layer: VolumetricLayer | None = None,
        dst_path: str | None = None,
        dst_layer: VolumetricLayer | None = None,
        dst_factory_chunk_size: Sequence[int] | None = None,
        dst_factory_resolution_list: Sequence[Sequence[int]] | None = None,
        crop_pad: Sequence[int] | None = None,
        subchunkable_kwargs=None,
        op_kwargs=None,
        fn_kwargs=None,
    ):
        if subchunkable_kwargs is None:
            subchunkable_kwargs = {}
        if op_kwargs is None:
            op_kwargs = {}
        if fn_kwargs is None:
            fn_kwargs = {}
        if dst_factory_resolution_list is None:
            dst_factory_resolution_list = [model["dst_resolution"] for model in models]

        self.subchunkable_flows = []
        for model in models:
            model_proc_chunk_sizes = copy.copy(processing_chunk_sizes)
            max_chunk_size = model.get("max_processing_chunk_size", None)
            # apply per-model overrides
            model_subchunkable_kwargs = copy.copy(subchunkable_kwargs) | model.get(
                "subchunkable_kwargs", {}
            )
            model_op_kwargs = op_kwargs | model.get("op_kwargs", {})
            model_fn_kwargs = fn_kwargs | model.get("fn_kwargs", {})

            res_change_mult = model.get("res_change_mult", [1, 1, 1])

            model_fn_kwargs = _set_tiled_fn_kwargs(
                model_path=model["path"],
                ds_factor=res_change_mult[0],
                overrides=model_fn_kwargs,
            )
            fn = BaseCoarsener(**model_fn_kwargs)

            dst_resolution = model["dst_resolution"]
            flow = SubchunkableFnFlowSchema(
                fn=fn,
                task_name=f"ImageEncoding-{dst_resolution[0]}",
                fn_uses_cuda=True,
                dst_resolution=model["dst_resolution"],
                model_res_change_mult=res_change_mult,
                model_max_processing_chunk_size=max_chunk_size,
                src_path=src_path,
                src_layer=src_layer,
                dst_path=dst_path,
                dst_layer=dst_layer,
                dst_factory_chunk_size=dst_factory_chunk_size,
                dst_factory_resolution_list=dst_factory_resolution_list,
                dst_data_type="int8",
                processing_chunk_sizes=model_proc_chunk_sizes,
                crop_pad=crop_pad,
                subchunkable_kwargs=model_subchunkable_kwargs,
                op_kwargs=model_op_kwargs,
            )
            self.subchunkable_flows.append(flow)

    def flow(self, bbox):
        for i, flow in enumerate(self.subchunkable_flows):
            yield flow(bbox, i)


@mazepa.flow_schema_cls
@attrs.mutable
class DefectFlowSchema:
    subchunkable_flow: SubchunkableFnFlowSchema

    def __init__(
        self,
        model_path: str,
        dst_resolution: Sequence[int],
        processing_chunk_sizes: Sequence[Sequence[int]],
        src_path: str | None = None,
        src_layer: VolumetricLayer | None = None,
        dst_path: str | None = None,
        dst_layer: VolumetricLayer | None = None,
        dst_factory_chunk_size: Sequence[int] | None = None,
        dst_factory_resolution_list: Sequence[Sequence[int]] | None = None,
        crop_pad: Sequence[int] | None = None,
        subchunkable_kwargs=None,
        op_kwargs=None,
        fn_kwargs=None,
    ):

        fn_kwargs = _set_tiled_fn_kwargs(
            model_path=model_path,
            overrides=fn_kwargs,
        )
        fn = DefectDetector(**fn_kwargs)

        self.subchunkable_flow = SubchunkableFnFlowSchema(
            fn=fn,
            task_name="Defect",
            fn_uses_cuda=True,
            dst_resolution=dst_resolution,
            model_res_change_mult=(1, 1, 1),
            src_path=src_path,
            src_layer=src_layer,
            dst_path=dst_path,
            dst_layer=dst_layer,
            dst_factory_chunk_size=dst_factory_chunk_size,
            dst_factory_resolution_list=dst_factory_resolution_list,
            dst_data_type="uint8",
            processing_chunk_sizes=processing_chunk_sizes,
            crop_pad=crop_pad,
            subchunkable_kwargs=subchunkable_kwargs,
            op_kwargs=op_kwargs,
        )

    def flow(self, bbox):
        yield self.subchunkable_flow(bbox, 0)


def _binarize_defect_prediction(
    src: torch.Tensor,
    threshold,
    kornia_opening_width: int = 0,
    kornia_dilation_width: int = 0,
    filter_cc_threshold: int = 0,
    kornia_closing_width: int = 0,
):
    # TODO: add option to convert to ZCYX once instead of at every step

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
        fn_kwargs: Mapping[str, Any],
        dst_resolution: Sequence[int],
        processing_chunk_sizes: Sequence[Sequence[int]],
        src_path: str | None = None,
        src_layer: VolumetricLayer | None = None,
        dst_path: str | None = None,
        dst_layer: VolumetricLayer | None = None,
        dst_factory_chunk_size: Sequence[int] | None = None,
        dst_factory_resolution_list: Sequence[Sequence[int]] | None = None,
        crop_pad: Sequence[int] | None = None,
        subchunkable_kwargs=None,
        op_kwargs=None,
    ):
        fn = partial(_binarize_defect_prediction, **fn_kwargs)
        self.subchunkable_flow = SubchunkableFnFlowSchema(
            fn=fn,
            task_name="BinarizeDefect",
            fn_uses_cuda=False,
            dst_resolution=dst_resolution,
            model_res_change_mult=(1, 1, 1),
            src_path=src_path,
            src_layer=src_layer,
            dst_path=dst_path,
            dst_layer=dst_layer,
            dst_factory_chunk_size=dst_factory_chunk_size,
            dst_factory_resolution_list=dst_factory_resolution_list,
            dst_data_type="uint8",
            processing_chunk_sizes=processing_chunk_sizes,
            crop_pad=crop_pad,
            subchunkable_kwargs=subchunkable_kwargs,
            op_kwargs=op_kwargs,
        )

    def flow(self, bbox):
        yield self.subchunkable_flow(bbox, 0)


def _mask_out(src, mask, grow_mask_width=0):
    if grow_mask_width > 0:
        mask = kornia_dilation(mask, width=grow_mask_width)
    return torch.where(mask > 0, 0, src)  # where(cond, true, false)


@mazepa.flow_schema_cls
@attrs.mutable
class MaskEncodingsFlowSchema:
    subchunkable_flows: Sequence[SubchunkableFnFlowSchema]

    def __init__(
        self,
        dst_resolution_list: Sequence[Sequence[int]],
        processing_chunk_sizes: Sequence[Sequence[int]],
        grow_mask_width: int = 0,
        src_encodings_path: str | None = None,
        src_encodings_layer: VolumetricLayer | None = None,
        src_mask_path: str | None = None,
        src_mask_layer: VolumetricLayer | None = None,
        src_mask_resolution: Sequence[int] | None = None,
        dst_path: str | None = None,
        dst_layer: VolumetricLayer | None = None,
        dst_factory_chunk_size: Sequence[int] | None = None,
        dst_factory_resolution_list: Sequence[Sequence[int]] | None = None,
        crop_pad: Sequence[int] | None = None,
        subchunkable_kwargs=None,
        op_kwargs: Mapping[str, Any] | None = None,
    ):
        if op_kwargs is None:
            op_kwargs = {}
        if subchunkable_kwargs is None:
            subchunkable_kwargs = {}

        # add mask src to op_kwargs
        if src_mask_layer is None:
            assert src_mask_path is not None
            src_mask_layer = build_cv_layer(
                src_mask_path,
                data_resolution=src_mask_resolution,
                interpolation_mode="mask",
            )
        # op_kwargs_src_mask = {"mask": src_mask_layer}
        # op_kwargs = op_kwargs_src_mask | op_kwargs
        op_kwargs_src_mask = {"op_kwargs": {"mask": src_mask_layer}}
        subchunkable_kwargs = op_kwargs_src_mask | subchunkable_kwargs

        if dst_factory_resolution_list is None:
            dst_factory_resolution_list = dst_resolution_list

        self.subchunkable_flows = []
        for dst_resolution in dst_resolution_list:

            src_encodings_path_ = src_encodings_path
            src_encodings_layer_ = src_encodings_layer

            # TODO: process per resolution overrides

            mask_fn = partial(_mask_out, grow_mask_width=grow_mask_width)

            flow = SubchunkableFnFlowSchema(
                fn=mask_fn,
                task_name="MaskEncodings",
                fn_uses_cuda=False,
                dst_resolution=dst_resolution,
                model_res_change_mult=(1, 1, 1),
                src_path=src_encodings_path_,
                src_layer=src_encodings_layer_,
                dst_path=dst_path,
                dst_layer=dst_layer,
                dst_factory_chunk_size=dst_factory_chunk_size,
                dst_factory_resolution_list=dst_factory_resolution_list,
                dst_data_type="int8",
                processing_chunk_sizes=processing_chunk_sizes,
                crop_pad=crop_pad,
                subchunkable_kwargs=subchunkable_kwargs,
                op_kwargs=op_kwargs,
            )
            self.subchunkable_flows.append(flow)

    def flow(self, bbox):
        for i, flow in enumerate(self.subchunkable_flows):
            yield flow(bbox, i)


@mazepa.flow_schema_cls
@attrs.mutable
class ComputeFieldFlowSchema:
    flows: Sequence[ComputeFieldMultistageFlowSchema]

    def __init__(
        self,

        stages: Sequence[Mapping[str, Any]],
        z_offsets: Sequence[int],
        # resume_path: str | None = None,
        # resume_resolution: Sequence[int] | None = None,
        processing_chunk_sizes: Sequence[Sequence[int]],

        src_path: str | None = None,
        src_layer: VolumetricLayer | None = None,
        tgt_path: str | None = None,
        tgt_layer: VolumetricLayer | None = None,

        dst_path: str | None = None,
        # dst_layer: VolumetricLayer | None = None,
        dst_factory_kwargs: Mapping[str, Any] | None = None,

        crop_pad: Sequence[int] | None = None,
        compute_field_multistage_kwargs: Mapping[str, Any] | None = None,
        compute_field_stage_kwargs: Mapping[str, Any] | None = None,
    ):
        if compute_field_multistage_kwargs is None:
            compute_field_multistage_kwargs = {}
        if compute_field_stage_kwargs is None:
            compute_field_stage_kwargs = {}
        if crop_pad is None:
            crop_pad = [0, 0, 0]

        if tgt_path is None:
            tgt_path = src_path
        if src_layer is None:
            src_layer = build_cv_layer(src_path)
        if tgt_layer is None:
            tgt_layer = build_cv_layer(tgt_path)

        # z_offset_resolution = set([stage["dst_resolution"][2] for stage in stages])
        z_offset_resolution = {stage["dst_resolution"][2] for stage in stages}
        if len(z_offset_resolution) > 1:
            raise RuntimeError("Inconsistent z resolutions between stages!")
        z_offset_resolution = z_offset_resolution.pop()

        cf_stages = []
        for stage in stages:
            cf_kwargs = {}
            cf_kwargs["dst_resolution"] = stage["dst_resolution"]
            cf_kwargs["processing_chunk_sizes"] = processing_chunk_sizes
            cf_kwargs["processing_crop_pads"] = _pad_crop_pads(crop_pad, len(processing_chunk_sizes))
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
                "data_type": "float32",
                "resolution_list": [stage["dst_resolution"] for stage in stages],
            } | dst_factory_kwargs
            dst_layer = _make_dst_layer_field(**dst_kwargs)
            default_tmp_layer_factory = partial(
                        build_cv_layer,
                        info_reference_path=dst_path_,
                        on_info_exists= "overwrite",
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
        for i, flow in enumerate(self.flows):
            yield flow(bbox=bbox)


@mazepa.flow_schema_cls
@attrs.mutable
class InvertFieldFlowSchema:
    subchunkable_flows: Sequence[SubchunkableFnFlowSchema2]

    def __init__(
        self,
        # dst_resolution_list: Sequence[Sequence[int]],
        dst_resolution: Sequence[int],
        z_offsets: Sequence[int],
        fn: Callable,
        fn_kwargs: Mapping[str, Any] | None = None,
        src_path: str | None = None,
        dst_path: str | None = None,
        # processing_chunk_sizes: Sequence[Sequence[int]],
        # dst_factory_kwargs: Mapping[str, Any] | None = None,
        # crop_pad: Sequence[int] | None = None,
        # subchunkable_kwargs=None,
        # op_kwargs: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        if fn_kwargs is None:
            fn_kwargs = {}
        self.subchunkable_flows = []
        for z_offset in z_offsets:
            src_path_ = os.path.join(src_path, str(z_offset))
            dst_path_ = os.path.join(dst_path, str(z_offset))
            flow = SubchunkableFnFlowSchema2(
                task_name=f"InvertField_z{z_offset}",
                fn=partial(fn, **fn_kwargs),
                dst_resolution=dst_resolution,
                # fn_uses_cuda=True,
                src_path=src_path_,
                dst_path=dst_path_,
                **kwargs,
            )
            self.subchunkable_flows.append(flow)

    def flow(self, bbox):
        for i, flow in enumerate(self.subchunkable_flows):
            yield flow(bbox, i)


@mazepa.flow_schema_cls
@attrs.mutable
class WarpFlowSchema:
    subchunkable_flows: Sequence[SubchunkableFnFlowSchema2]

    def __init__(
        self,
        dst_resolution: Sequence[int],
        z_offsets: Sequence[int],
        src_path: str,
        field_path: str,
        dst_path: str,
        field_resolution: Sequence[int] = None,
        subchunkable_kwargs=None,
        **kwargs,
    ):
        if subchunkable_kwargs is None:
            subchunkable_kwargs = {}

        self.subchunkable_flows = []
        for z_offset in z_offsets:

            field_path_ = os.path.join(field_path, str(z_offset))
            dst_path_ = os.path.join(dst_path, str(z_offset))
            src_idx_translator = VolumetricIndexTranslator(
                offset=[0, 0, z_offset], resolution=[1, 1, dst_resolution[2]]
            )
            src_layer = build_cv_layer(
                src_path, index_procs=[src_idx_translator]
            )
            field_layer = build_cv_layer(
                field_path_, data_resolution=field_resolution, interpolation_mode="field"
            )
            subchunkable_kwargs = {
                "op_kwargs": {
                    "src": src_layer,
                    "field": field_layer,
                }
            } | subchunkable_kwargs

            flow = SubchunkableFnFlowSchema2(
                task_name=f"Warp_{z_offset}",
                op=WarpOperation(mode="img"),
                dst_resolution=dst_resolution,
                fn_uses_cuda=False,
                src_path=src_path,
                src_layer=src_layer,
                dst_path=dst_path_,
                subchunkable_kwargs=subchunkable_kwargs,
                **kwargs,
            )
            self.subchunkable_flows.append(flow)

    def flow(self, bbox):
        for i, flow in enumerate(self.subchunkable_flows):
            yield flow(bbox, i)

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


@builder.register("build_pairwise_alignment_flow")
def build_pairwise_alignment_flow(
    bbox: BBox3D,
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
    compute_field_flow_kwargs: Mapping[str, Any] | None = None,
    run_invert_field: bool = False,
    invert_field_flow_kwargs: Mapping[str, Any] | None = None,
    run_warp: bool = False,
    warp_flow_kwargs: Mapping[str, Any] | None = None,
) -> mazepa.Flow:  # pylint: disable=too-many-statements)

    def resolve_path(path, default=None):
        if path is None:
            path = default
        if "gs://" in path:
            # path is absolute
            return path
        # otherwise path is relative
        return os.path.join(project_folder, path)

    def set_path(config, key, default):
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
    set_path(mask_encodings_flow_kwargs, "src_mask_path", default=binarize_defect_flow_kwargs["dst_path"])
    set_path(mask_encodings_flow_kwargs, "src_encodings_path", default=encoding_flow_kwargs["dst_path"])
    set_path(mask_encodings_flow_kwargs, "dst_path", default="masked_encodings")
    if run_mask_encodings:
        mask_encodings_flow = MaskEncodingsFlowSchema(**mask_encodings_flow_kwargs)

    compute_field_flow = None
    if compute_field_flow_kwargs is None:
        compute_field_flow_kwargs = {}
    cf_default_src = mask_encodings_flow_kwargs["dst_path"] if skipped_defect else \
                     encoding_flow_kwargs["dst_path"]
    set_path(compute_field_flow_kwargs, "src_path", default=cf_default_src)
    set_path(compute_field_flow_kwargs, "dst_path", default="fields_fwd")
    compute_field_flow_kwargs = {"z_offsets": z_offsets} | compute_field_flow_kwargs
    if run_compute_field:
        compute_field_flow = ComputeFieldFlowSchema(**compute_field_flow_kwargs)

    invert_field_flow = None
    if invert_field_flow_kwargs is None:
        invert_field_flow_kwargs = {}
    set_path(invert_field_flow_kwargs, "src_path", default=compute_field_flow_kwargs["dst_path"])
    set_path(invert_field_flow_kwargs, "dst_path", default="fields_inv")
    # try to determine dst_resolution
    if "stages" in compute_field_flow_kwargs:
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
    set_path(warp_flow_kwargs, "dst_path", default="imgs_warped")
    # try to determine field_resolution
    if "dst_resolution" in invert_field_flow_kwargs:
        warp_flow_kwargs = {"field_resolution": invert_field_flow_kwargs["dst_resolution"]} | warp_flow_kwargs
    warp_flow_kwargs = {"z_offsets": z_offsets} | warp_flow_kwargs
    if run_warp:
        warp_flow = WarpFlowSchema(**warp_flow_kwargs)

    flow_schema = PairwiseAlignmentFlowSchema(
        encoding_flow=encoding_flow,
        defect_flow=defect_flow,
        binarize_defect_flow=binarize_defect_flow,
        mask_encodings_flow=mask_encodings_flow,
        compute_field_flow=compute_field_flow,
        invert_field_flow=invert_field_flow,
        warp_flow=warp_flow,
    )
    flow = flow_schema(bbox=bbox)
    return flow
