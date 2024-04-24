
from __future__ import annotations

from functools import partial
import os
from typing import Any, Callable, Literal, Mapping, Sequence, TypedDict, Union

import attrs
from numpy import interp
import torch

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.builder.built_in_registrations import efficient_parse_lambda_str
from zetta_utils.common.partial import ComparablePartial
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.internal.alignment.base_coarsener import BaseCoarsener
from zetta_utils.internal.alignment.base_encoder import BaseEncoder
from zetta_utils.internal.alignment.defect_detector import DefectDetector
from zetta_utils.internal.alignment.online_finetuner import align_with_online_finetuner
from zetta_utils.layer.volumetric import (
    VolumetricLayer,
)
from zetta_utils.layer.volumetric.cloudvol.build import build_cv_layer
from zetta_utils.mazepa.flows import Dependency
from zetta_utils.mazepa.semaphores import SemaphoreType
from zetta_utils.mazepa_layer_processing.alignment.compute_field_multistage_flow import ComputeFieldMultistageFlowSchema, ComputeFieldStage
from zetta_utils.mazepa_layer_processing.common import build_subchunkable_apply_flow
from zetta_utils.mazepa_layer_processing.common.interpolate_flow import make_interpolate_operation
from zetta_utils.mazepa_layer_processing.common.volumetric_callable_operation import VolumetricCallableOperation

from zetta_utils.mazepa_layer_processing.operation_protocols import VolumetricOpProtocol
from zetta_utils.tensor_ops.common import compare
from zetta_utils.tensor_ops.convert import to_uint8
from zetta_utils.tensor_ops.mask import (
    filter_cc,
    kornia_closing,
    kornia_dilation,
    kornia_opening,
)


@builder.register("binarize_defect_prediction")
def binarize_defect_prediction(
    src: torch.Tensor,
    threshold: int = 100,
    kornia_opening_width: int = 11,
    kornia_dilation_width: int = 3,
    filter_cc_threshold: int = 240,
    kornia_closing_width: int = 30,
):
    pred = compare(src, mode=">=", value=threshold, binarize=True)

    mask = to_uint8(pred) 
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

class EncParams(TypedDict):
    model_path: str
    res_change_mult: Sequence[int] 
    
class ScaleSpec(TypedDict):
    enc_params: EncParams
    cf_fn: Callable 
    
@attrs.mutable
class Stages: 
    enc_model_path: str
    enc_res_change_mult: Vec3D[int]
    cf_fn: Callable 
    
    dst_resolution: Vec3D[float]
    processing_chunk_size_l0: Vec3D[int]

def build_subchunkable_with_auto_expansion(
    dst: VolumetricLayer,
    bbox: BBox3D,
    dst_resolution: Sequence[float],
    processing_chunk_sizes: Sequence[Sequence[int]],
    processing_crop_pads: Sequence[int] | Sequence[Sequence[int]] = (0, 0, 0),
    processing_blend_pads: Sequence[int] | Sequence[Sequence[int]] = (0, 0, 0),
    processing_blend_modes: Union[
        Literal["linear", "quadratic", "defer"], Sequence[Literal["linear", "quadratic", "defer"]]
    ] = "quadratic",
    fn: Callable | None = None,
    fn_semaphores: Sequence[SemaphoreType] | None = None,
    op: VolumetricOpProtocol | None = None,
    op_kwargs: Mapping[str, Any] | None = None,
): 
    if dst_resolution[0] > 512:
        expansion_args = {
            "expand_bbox_backend": True,
            "expand_bbox_resolution": True,
            "expand_bbox_processing": False,
            "shrink_processing_chunk": True,
        }
    else:
        expansion_args = {
           "expand_bbox_processing": True, 
        }
        
    return build_subchunkable_apply_flow(
        dst=dst,
        bbox=bbox,
        dst_resolution=dst_resolution,
        processing_chunk_sizes=processing_chunk_sizes,
        processing_crop_pads=processing_crop_pads,
        processing_blend_pads=processing_blend_pads,
        processing_blend_modes=processing_blend_modes,
        op=op,
        fn=fn,
        fn_semaphores=fn_semaphores,
        op_kwargs=op_kwargs,
        level_intermediaries_dirs=['file://~/.zetta_utils/tmp0/', 'file://~/.zetta_utils/tmp1/'],
        **expansion_args
    )
    
@builder.register("per_offset_pairwise_flow")
@mazepa.flow_schema
def per_offset_pairwise_flow(
    img_layer: VolumetricLayer,
    enc_masked_layer: VolumetricLayer, 
    bbox: BBox3D
):
    ...

@builder.register("PairwiseAlignFlowSchema")
@mazepa.flow_schema_cls
@attrs.mutable
class DataPreparationFlowSchema:
    img_path: str
    bbox: BBox3D
    dst_dir: str
    defect_model_path: str
    scale_specs: list[ScaleSpec]
    base_resolution: Vec3D[float]
    on_info_exists_mode: Literal["overwrite", "expect_same"] 
    l1_proc_chunk: Vec3D[int] 
    
    # Reference Resolutions for models 
    enc_base_reference_xy_resolution: int 
    defect_reference_xy_resolution: int 
   
    defect_binarization_fn: Callable[[torch.Tensor], torch.Tensor]
     
    # Proc Chunk Settings
    enc_max_l0_proc_chunk: Vec3D[int]
    defect_max_l0_proc_chunk: Vec3D[int]
   
    offsets: Sequence[int]
    
    # Skip settings 
    skip_enc: bool
    skip_defect_detect: bool 
    skip_defect_bin: bool
    skip_enc_mask: bool 
    skip_cf: bool
   
    # Layers 
    img_layer: VolumetricLayer = attrs.field(init=False)
    enc_layer: VolumetricLayer = attrs.field(init=False)
    defect_raw_layer: VolumetricLayer = attrs.field(init=False)
    defect_bin_layer: VolumetricLayer = attrs.field(init=False)
    enc_masked_layer: VolumetricLayer = attrs.field(init=False)
    pfield_layers: dict[int, VolumetricLayer] = attrs.field(init=False)
    pfield_inv_layers: dict[int, VolumetricLayer] = attrs.field(init=False)
    
    # Misc 
    defect_resolution: Vec3D[float] = attrs.field(init=False, factory=list)
    enc_stages: list[EncStage] = attrs.field(init=False, factory=list)
    
    def _setup_enc_details(self):
        curr_xy_resolution = self.base_resolution[0]
        while curr_xy_resolution * 2 <= self.enc_base_reference_xy_resolution:
            curr_xy_resolution *= 2
        
        for scale_spec in self.scale_specs: 
            enc_stage = EncStage(
                path=scale_spec["enc_params"]["model_path"],
                res_change_mult=Vec3D[int](*scale_spec["enc_params"]["res_change_mult"]), 
                dst_resolution=Vec3D[float](curr_xy_resolution, curr_xy_resolution, self.base_resolution[-1]),
                processing_chunk_size_l0=self.enc_max_l0_proc_chunk // Vec3D[int](*scale_spec["enc_params"]["res_change_mult"])
            )
            self.enc_stages.append(enc_stage)
            curr_xy_resolution *= 2
            
        self.enc_layer = build_cv_layer(
            path=os.path.join(self.dst_dir, "enc") ,
            info_reference_path=self.img_path,
            info_add_scales=[
                list(stage.dst_resolution) for stage in self.enc_stages
            ],
            info_add_scales_mode="replace",
            on_info_exists=self.on_info_exists_mode,
            info_field_overrides={
                'data_type': "int8"
            }
        )
        self.enc_masked_layer = build_cv_layer(
            path=os.path.join(self.dst_dir, "enc_masked") ,
            info_reference_path=self.img_path,
            info_add_scales=[
                list(stage.dst_resolution) for stage in self.enc_stages
            ],
            info_add_scales_mode="replace",
            on_info_exists=self.on_info_exists_mode,
            info_field_overrides={
                'data_type': "int8"
            }
        )
        
    def _set_up_defect_details(self):
        curr_xy_resolution = self.base_resolution[0]
        while curr_xy_resolution * 2 <= self.defect_reference_xy_resolution:
            curr_xy_resolution *= 2
        self.defect_resolution = Vec3D[float](curr_xy_resolution, curr_xy_resolution, self.base_resolution[-1])
        
        defect_raw_path = os.path.join(self.dst_dir, "defect_raw") 
        self.defect_raw_layer = build_cv_layer(
            path=defect_raw_path,
            info_reference_path=self.img_path,
            info_add_scales=[list(self.defect_resolution)],
            info_add_scales_mode="replace",
            on_info_exists=self.on_info_exists_mode,
            info_field_overrides={
                'data_type': "uint8"
            }
        ) 
        defect_bin_path = os.path.join(self.dst_dir, "defect_bin") 
        self.defect_bin_layer = build_cv_layer(
            path=defect_bin_path,
            info_reference_path=self.img_path,
            info_add_scales=[
                list(stage.dst_resolution) for stage in self.enc_stages
            ],
            info_add_scales_mode="replace",
            on_info_exists=self.on_info_exists_mode,
            info_field_overrides={
                'data_type': "uint8"
            },
            write_procs=[
                ComparablePartial(filter_cc, thr=6, mode='keep_large')
            ]
        ) 
        
    def _set_up_cf_details(self):
        for offset in self.offsets:
            build_field_layer_fn = partial(  
                build_cv_layer,
                info_reference_path=self.img_path,
                info_add_scales=[
                    {
                        "resolution": list(stage.dst_resolution) ,
                        "encoding": "zfpc",
                        "zfpc_correlated_dims": [True, True, False, False],
                        "zfpc_tolerance": 0.001953125,
                    }    
                    for stage in self.enc_stages
                ],
                info_add_scales_mode="replace",
                on_info_exists=self.on_info_exists_mode,
                info_field_overrides={
                    'data_type': "float32"
                },
            ) 
            self.pfield_layers[offset] = build_field_layer_fn(
                path=os.path.join(self.dst_dir, "pfields", str(offset)),
            )
            self.pfield_inv_layers[offset] = build_field_layer_fn(
                path=os.path.join(self.dst_dir, "pfields_inv", str(offset)),
            )
        
    def __attrs_post_init__(self):
        self.img_layer = build_cv_layer(path=self.img_path) 
        self._setup_enc_details()
        self._set_up_defect_details()
        
    def flow(self):
        if not self.skip_enc:
            for stage in self.enc_stages:    
                enc_flow = build_subchunkable_with_auto_expansion(
                    dst=self.enc_layer,
                    bbox=self.bbox,
                    dst_resolution=stage.dst_resolution,
                    processing_chunk_sizes=[self.l1_proc_chunk, stage.processing_chunk_size_l0],
                    processing_crop_pads=[[0, 0, 0], [32, 32, 0]],
                    op=VolumetricCallableOperation(
                        fn=BaseCoarsener(
                            model_path=stage.path,
                            ds_factor=stage.res_change_mult[0],
                            tile_size=None
                        ),
                        res_change_mult=stage.res_change_mult,
                    ),
                    op_kwargs={
                        'src': self.img_layer, 
                    },
                )
                yield enc_flow 
                
        if not self.skip_defect_detect:
            defect_detect_flow = build_subchunkable_with_auto_expansion(
                dst=self.defect_raw_layer,
                bbox=self.bbox,
                dst_resolution=self.defect_resolution,
                processing_chunk_sizes=[self.l1_proc_chunk, self.defect_max_l0_proc_chunk],
                processing_crop_pads=[[0, 0, 0], [512, 512, 0]],
                fn=DefectDetector(
                    model_path=self.defect_model_path,
                ),
                op_kwargs={
                    'src': self.img_layer, 
                },
            )
            yield defect_detect_flow  
        
        if not self.skip_defect_bin:
            if not self.skip_defect_detect: 
                yield Dependency(defect_detect_flow)
                
            defect_bin_flow = build_subchunkable_with_auto_expansion(
                dst=self.defect_bin_layer,
                bbox=self.bbox,
                dst_resolution=self.defect_resolution,
                processing_chunk_sizes=[self.l1_proc_chunk, self.defect_max_l0_proc_chunk],
                processing_crop_pads=[[0, 0, 0], [128, 128, 0]],
                fn=self.defect_binarization_fn,
                op_kwargs={
                    'src': self.defect_raw_layer, 
                },
            )
            yield defect_bin_flow  
            yield Dependency(defect_bin_flow)
            
            last_defect_interp_flow = None
            for enc_stage in self.enc_stages:
                if enc_stage.dst_resolution[0] > self.defect_resolution[0]:
                    defect_interp_flow = build_subchunkable_with_auto_expansion(
                        dst=self.defect_bin_layer,
                        bbox=self.bbox,
                        dst_resolution=enc_stage.dst_resolution,
                        processing_chunk_sizes=[self.l1_proc_chunk, self.defect_max_l0_proc_chunk],
                        processing_crop_pads=[[0, 0, 0], [16, 16, 0]],
                        op=make_interpolate_operation(
                            res_change_mult=[2, 2, 1],
                            mode="mask",
                            mask_value_thr=0.0,
                        ),
                        op_kwargs={
                            'src': self.defect_bin_layer, 
                        },
                    ) 
                    if last_defect_interp_flow is not None:
                        yield Dependency(last_defect_interp_flow)
                    yield defect_interp_flow 
                    last_defect_interp_flow = defect_interp_flow
                
        if not self.skip_enc_mask:
            if not self.skip_enc:
                yield Dependency(enc_flow)
            if not self.skip_defect_bin:
                yield Dependency(last_defect_interp_flow)
            enc_mask_flows = []
            defect_bin_layer_with_data_resolution = build_cv_layer(
                path=self.defect_bin_layer.name,
                data_resolution=self.defect_resolution,
                interpolation_mode="mask"
            )  
            
            for enc_stage in self.enc_stages:
                if enc_stage.dst_resolution[0] < self.defect_resolution[0]:
                    mask_layer = defect_bin_layer_with_data_resolution
                else:
                    mask_layer = self.defect_bin_layer
                self.defect_bin_layer.with_changes()
                enc_mask_flows.append(
                    build_subchunkable_with_auto_expansion(
                        dst=self.enc_masked_layer,
                        bbox=self.bbox,
                        dst_resolution=enc_stage.dst_resolution,
                        processing_chunk_sizes=[self.l1_proc_chunk, Vec3D[int](1024, 1024, 1)],
                        fn=efficient_parse_lambda_str(
                            lambda_str="lambda src, mask: torch.where(mask > 0, 0, src)",
                            name="Maksing Encs"
                        ),
                        op_kwargs={
                            'src': self.enc_layer, 
                            'mask': mask_layer, 
                        },
                    )
                )
            yield enc_mask_flows  
        if not self.skip_cf:
            for offset in self.offsets:
                cf_flow_schema = ComputeFieldMultistageFlowSchema(
                    stages=[
                    ComputeFieldStage(
                        dst_resolution=self.enc_stages[i].dst_resolution,
                        fn=ComparablePartial(
                            align_with_online_finetuner,
                            **stage_spec["cf_fn"]
                        ),
                    )
                    for i, stage_spec in reversed(enumerate(self.scale_specs))
                    ],
                    tmp_layer_dir=os.path.join(self.dst_dir, "field_tmp"),
                    tmp_layer_factory=ComparablePartial(
                        build_cv_layer,
                        info_reference_path=self.pfield_layers[-1].name,
                        on_info_exists=self.on_info_exists_mode,
                    )
                )
                cf_flow = cf_flow_schema(
                    
                ) 


@builder.register("build_pairwise_align_flow")    
def build_pairwise_align_flow(
    img_path: str,
    bbox: BBox3D,
    dst_dir: str,
    defect_model_path: str,
    scale_specs: list[ScaleSpec],
    base_resolution: Sequence[float],
    on_info_exists_mode: Literal["overwrite", "expect_same"] = "expect_same",
    l1_proc_chunk: Sequence[int] = (4096, 4096, 1),
    enc_base_reference_xy_resolution: int = 32,
    defect_reference_xy_resolution: int = 64,
    defect_binarization_fn: Callable[[torch.Tensor], torch.Tensor] = binarize_defect_prediction,
    enc_max_l0_proc_chunk: Sequence[int] = (2048, 2048, 1),
    defect_max_l0_proc_chunk: Sequence[int] = (512, 512, 1),
    offsets: Sequence[int] = (-1, -2),
    skip_enc: bool = False,
    skip_defect_detect: bool = False,
    skip_defect_bin: bool = False,
    skip_enc_mask: bool = False,
    skip_cf: bool = False,
):
    flow_schema = PairwiseAlignFlowSchema(
        img_path=img_path,
        bbox=bbox,
        dst_dir=dst_dir,
        defect_model_path=defect_model_path,
        scale_specs=scale_specs,
        base_resolution=Vec3D[float](*base_resolution),
        on_info_exists_mode=on_info_exists_mode, 
        l1_proc_chunk=l1_proc_chunk, 
        enc_base_reference_xy_resolution=enc_base_reference_xy_resolution,
        defect_reference_xy_resolution=defect_reference_xy_resolution,
        defect_binarization_fn=defect_binarization_fn,
        enc_max_l0_proc_chunk=Vec3D[int](*enc_max_l0_proc_chunk),
        defect_max_l0_proc_chunk=Vec3D[int](*defect_max_l0_proc_chunk),
        offsets=offsets,
        skip_enc=skip_enc,
        skip_defect_detect=skip_defect_detect,
        skip_defect_bin=skip_defect_bin,
        skip_enc_mask=skip_enc_mask,
        skip_cf=skip_cf,
    )
    
    return flow_schema()