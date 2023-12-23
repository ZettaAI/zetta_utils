from __future__ import annotations

#pylint: disable=unused-import

import copy
import os

# import os
from typing import Callable, Literal, Sequence, Union, Any, Mapping

import attrs
from functools import partial

from zetta_utils import builder, mazepa
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import (
    DataResolutionInterpolator,
    VolumetricIndexTranslator,
    VolumetricLayer,
)
from zetta_utils.alignment.base_coarsener import BaseCoarsener
from zetta_utils.mazepa_layer_processing.common import (
    VolumetricCallableOperation,
    build_subchunkable_apply_flow,
)
from zetta_utils.layer.volumetric.cloudvol import (
    build_cv_layer,
)


def _set_tiled_fn_kwargs(model_path, ds_factor, crop_pad, processing_chunk_sizes, overrides=None):
    if overrides is None:
        overrides = {}
    fn_kwargs = {}
    fn_kwargs["model_path"] = model_path
    fn_kwargs["ds_factor"] = ds_factor
    fn_kwargs["tile_pad_in"] = ds_factor * crop_pad[0]
    fn_kwargs["tile_size"] = ds_factor * processing_chunk_sizes[-1][0]
    fn_kwargs |= overrides
    return fn_kwargs


def _get_dst_layer(dst: str | VolumetricLayer, layer_factory: Callable):
    if isinstance(dst, VolumetricLayer):
        return dst
    return layer_factory(dst)


def _set_subchunkable_kwargs(
    src_layer,
    dst_layer,
    processing_chunk_sizes,
    crop_pad,
    dst_resolution=None,
    overrides=None,
):
    if overrides is None:
        overrides = {}
    kwargs = {}
    kwargs["op_kwargs"] = {"src": src_layer}
    kwargs["processing_chunk_sizes"] = processing_chunk_sizes
    kwargs["processing_crop_pads"] = [[0, 0, 0]] * (len(processing_chunk_sizes) - 1) + [crop_pad]
    kwargs["skip_intermediaries"] = True
    kwargs["level_intermediaries_dirs"] = None
    kwargs["dst"] = dst_layer
    kwargs["dst_resolution"] = dst_resolution
    kwargs |= overrides
    return kwargs


def _set_subchunkable_op_kwargs(res_change_mult=None, overrides=None):
    if overrides is None:
        overrides = {}
    kwargs = {}
    kwargs["fn_semaphores"] = ["cuda"]
    if res_change_mult is not None:
        kwargs["res_change_mult"] = res_change_mult
    kwargs |= overrides
    return kwargs


@mazepa.flow_schema_cls
@attrs.mutable
class SubchunkableFnFlowSchema:
    fn_kwargs: Mapping[Any, Any]
    op_kwargs: Mapping[Any, Any]
    subchunkable_kwargs: Mapping[Any, Any]
    callable_fn: Callable[..., Any]

    def __init__(
        self,
        callable_fn,
        model_path: str,
        model_resolution: Sequence[int],
        model_res_change_mult: Sequence[int] | None = None,
        src_path: str | None = None,
        src_layer: VolumetricLayer | None = None,
        dst_path: str | None = None,
        dst_layer: VolumetricLayer | None = None,
        dst_factory: Callable | None = None,
        dst_chunk_size: Sequence[int] | None = None,
        dst_resolution_list: Sequence[Sequence[int]] | None = None,
        processing_chunk_sizes=None,
        crop_pad: Sequence[int] | None = None,
        subchunkable_kwargs=None,
        op_kwargs=None,
        fn_kwargs=None,
    ):
        if src_layer is None:
            src_layer = build_cv_layer(src_path)

        if dst_layer is None:
            if src_path is None:
                raise RuntimeError(f"If `dst_layer` is not provided, `src_path` must be specified to be used as `info_ref` in making the destination layer with `dst_path`='{dst_path}'")
            if dst_factory is None:
                if dst_resolution_list is None:
                    dst_resolution_list = [model_resolution]
                dst_factory = partial(
                    _make_dst_layer, resolution_list=dst_resolution_list, chunk_size=dst_chunk_size,
                    info_ref=src_path
                )
            # dst_layer = _get_dst_layer(dst_path=dst_path, layer_factory=dst_factory)
            dst_layer = dst_factory(dst_path)

        if crop_pad is None:
            crop_pad = [0, 0, 0]

        self.subchunkable_kwargs = _set_subchunkable_kwargs(
            src_layer=src_layer,
            dst_layer=dst_layer,
            dst_resolution=model_resolution,
            processing_chunk_sizes=processing_chunk_sizes,
            crop_pad=crop_pad,
            overrides=subchunkable_kwargs,
        )
        self.op_kwargs = _set_subchunkable_op_kwargs(
            res_change_mult=model_res_change_mult,
            overrides=op_kwargs,
        )
        self.fn_kwargs = _set_tiled_fn_kwargs(
            model_path=model_path,
            ds_factor=model_res_change_mult[0],
            crop_pad=crop_pad,
            processing_chunk_sizes=processing_chunk_sizes,
            overrides=fn_kwargs,
        )
        self.callable_fn = callable_fn

    def flow(self, bbox, i):
        flow = build_subchunkable_apply_flow(
            bbox=bbox,
            op=VolumetricCallableOperation(
                fn=self.callable_fn(**self.fn_kwargs),
                **self.op_kwargs,
            ),
            **self.subchunkable_kwargs,
        )
        yield flow


@mazepa.flow_schema_cls
@attrs.mutable
class EncodingFlowSchema:
    flows: Sequence[SubchunkableFnFlowSchema]

    def __init__(
        self,
        models: Sequence[Any],
        src_path: str | None = None,
        src_layer: VolumetricLayer | None = None,
        dst_path: str | None = None,
        dst_layer: VolumetricLayer | None = None,
        dst_chunk_size: Sequence[int] | None = None,
        dst_resolution_list: Sequence[Sequence[int]] | None = None,
        processing_chunk_sizes=None,
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
        if dst_resolution_list is None:
            dst_resolution_list = [model["dst_resolution"] for model in models]
        self.flows = []
        for model in models:
            # handle max_processing_chunk_size for each model
            model_proc_chunk_sizes = copy.copy(processing_chunk_sizes)
            max_chunk_size = model.get("max_processing_chunk_size", None)
            if max_chunk_size is not None:
                model_proc_chunk_sizes[-1] = [
                    min(a, b) for a, b in zip(max_chunk_size, model_proc_chunk_sizes[-1])
                ]
            # apply per-model overrides
            model_subchunkable_kwargs = copy.copy(subchunkable_kwargs) | model.get("subchunkable_kwargs", {})
            model_op_kwargs = op_kwargs | model.get("op_kwargs", {})
            model_fn_kwargs = fn_kwargs | model.get("fn_kwargs", {})
            flow = SubchunkableFnFlowSchema(
                callable_fn=BaseCoarsener,
                model_path=model["path"],
                model_resolution=model["dst_resolution"],
                model_res_change_mult=model["res_change_mult"],
                src_path=src_path,
                src_layer=src_layer,
                dst_path=dst_path,
                dst_layer=dst_layer,
                dst_chunk_size=dst_chunk_size,
                dst_resolution_list=dst_resolution_list,
                processing_chunk_sizes=model_proc_chunk_sizes,
                crop_pad=crop_pad,
                subchunkable_kwargs=model_subchunkable_kwargs,
                op_kwargs=model_op_kwargs,
                fn_kwargs=model_fn_kwargs,
            )
            self.flows.append(flow)

    def flow(self, bbox):
        for i, flow in enumerate(self.flows):
            yield flow(bbox, i)


@builder.register("PairwiseAlignmentFlowSchema")
@mazepa.flow_schema_cls
@attrs.mutable
class PairwiseAlignmentFlowSchema:
    project_folder: str
    run_encoding: bool
    encoding_flow_schema: EncodingFlowSchema
    # run_defect: bool
    # defect_flow_schema: DefectFlowSchema

    def _resolve_path(self, path, default=None):
        if path is None:
            path = default
        if "gs://" in path:
            # is absolute path
            return path
        # is a relative path to `project_folder`
        return os.path.join(self.project_folder, path)

    def __init__(
        self,
        src_image_path: str,
        project_folder: str = "",
        run_encoding: bool = False,
        encoding_kwargs: dict[Any, Any] = None,
        # run_defect: bool = False,
        # defect_configs: dict = None,
    ):  # pylint: disable=too-many-statements
        self.project_folder = project_folder

        self.run_encoding = run_encoding
        if run_encoding:
            # Resolve encoding paths
            encoding_kwargs["src_path"] = self._resolve_path(encoding_kwargs.get("src_path", src_image_path))
            encoding_kwargs["dst_path"] = self._resolve_path(encoding_kwargs.get("dst_path", "encodings"))
            self.encoding_flow_schema = EncodingFlowSchema(**encoding_kwargs)

        # self.run_defect = run_defect
        # if run_defect:
        #     self.defect_schema = DefectFlowSchema(**defect_flow_kwargs)

    def flow(self, bbox):
        if self.run_encoding:
            yield self.encoding_flow_schema(bbox)

        # yield self.encoding_flow_schema.flows[1](bbox)
        # yield self.encoding_flow_schema.flows[2](bbox)


        # if self.run_defect:
        #     flow = build_subchunkable_apply_flow(
        #         bbox=bbox,
        #         op=VolumetricCallableOperation(
        #             fn=DefectDetector(**self.defect_fn_kwargs),
        #             **self.defect_op_kwargs,
        #         ),
        #         **self.defect_subchunkable_kwargs,
        #     )
        #     yield flow

        # yield mazepa.Dependency()


@builder.register("build_pairwise_alignment_flow")
def build_pairwise_alignment_flow(
    bbox: BBox3D,
    src_image_path: str,
    project_folder: str = "",
    run_encoding: bool = False,
    encoding_kwargs: dict | None = None,
    # run_defect: bool = False,
    # defect_configs: dict | None = None,
) -> mazepa.Flow:
    flow_schema = PairwiseAlignmentFlowSchema(
        src_image_path=src_image_path,
        project_folder=project_folder,
        run_encoding=run_encoding,
        encoding_kwargs=encoding_kwargs,
        # run_defect=run_defect,
        # defect_configs=defect_configs,
    )
    flow = flow_schema(bbox=bbox)
    return flow


def _make_dst_layer(path, info_ref, resolution_list, chunk_size=None):
    return build_cv_layer(
        path=path,
        info_reference_path=info_ref,
        info_add_scales=resolution_list,
        info_add_scales_mode="replace",
        info_field_overrides={
            "type": "image",
            "num_channels": 1,
            "data_type": "int8",
        },
        info_chunk_size=chunk_size,
        on_info_exists="overwrite",
    )
